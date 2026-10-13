import XCTest
@testable import SLMChat

/// Pins the Swift port of MediaPipe's Language Detector against reference output
/// captured from the extraction's NumPy implementation.
///
/// This exists for the same reason `GuardEngineTests` does, and the failure it
/// guards against is the same one: a drifted port does not crash, it returns a
/// confident answer about the wrong language, and the only symptom is that some
/// messages get screened by a classifier that never saw their language. Nothing
/// else in the suite would notice.
///
/// The preprocessing is where that drift comes from, not the arithmetic. It
/// depends on Plan9 Unicode tables that disagree with Foundation, on iterating
/// unicode scalars rather than Characters, and on faithfully reproducing a
/// buffer over-read in MediaPipe when lowercasing shortens the text. Each of
/// those is a plausible "cleanup" that silently changes the answer.
///
/// `testNGramIDsAreExact` is therefore the load-bearing one: an exact integer
/// comparison on the token ids catches every one of those, where a probability
/// comparison would only catch the large ones.
///
/// **If this goes red, do not regenerate the fixtures to make it green.** They
/// are captured from the reference implementation, not from this port; rebuilding
/// them from the port would make the test assert that the code equals itself and
/// delete the only check there is. The reference lives in `language-detection/`,
/// which is gitignored and ships via `$SLMCHAT_ASSETS` — fetch it, run
/// `scripts/build_langdetect_assets.py`, and find out which side actually moved.
/// Regenerate only when the model itself changes.
final class LangDetectConformanceTests: XCTestCase {

    private var detector: LangDetect { LangDetect.shared }

    override func setUpWithError() throws {
        try XCTSkipUnless(detector.isLoaded,
                          "langdetect-weights.{bin,json} not bundled — run scripts/bootstrap.sh")
    }

    /// FNV-1a-64, matching `build_langdetect_assets.py::fnv1a64`.
    private func digest(_ ids: [Int32]) -> UInt64 {
        var h: UInt64 = 0xCBF29CE484222325
        for v in ids {
            withUnsafeBytes(of: v.littleEndian) { bytes in
                for b in bytes {
                    h = (h ^ UInt64(b)) &* 0x100000001B3
                }
            }
        }
        return h
    }

    /// Exact integer equality on the n-gram ids, over the full fixture set.
    ///
    /// Any drift in the Plan9 lowercasing, the tokenization, the truncation rule
    /// or MurmurHash64 moves at least one id and fails here.
    func testNGramIDsAreExact() {
        var failures: [String] = []
        for v in LangDetectFixtures.vectors {
            let (ids, tokens) = detector.ngramIDs(v.text)
            guard tokens == v.tokens else {
                failures.append("token count \(tokens) != \(v.tokens) for \(v.text.debugDescription)")
                continue
            }
            let d = digest(ids)
            if d != v.idsDigest {
                failures.append("id digest \(d) != \(v.idsDigest) for \(v.text.debugDescription)")
            }
        }
        XCTAssertEqual(failures.count, 0,
                       "n-gram hashing drifted from the reference:\n"
                       + failures.prefix(8).joined(separator: "\n"))
    }

    /// The label the model actually returns must match the reference.
    func testTopLabelsMatchReference() {
        var failures: [String] = []
        for v in LangDetectFixtures.vectors {
            guard let top = detector.ranked(v.text).first else {
                if !v.text.isEmpty { failures.append("no hypotheses for \(v.text.debugDescription)") }
                continue
            }
            if top.code != v.top {
                failures.append("\(v.text.debugDescription): got \(top.code), expected \(v.top)")
            }
        }
        XCTAssertEqual(failures.count, 0,
                       "top-1 disagreements:\n" + failures.prefix(8).joined(separator: "\n"))
    }

    /// Probabilities must land within float32 accumulation noise of the reference.
    ///
    /// The bound is deliberately tight. It is NOT there to catch rounding — it is
    /// there to catch someone replacing the hybrid int8 FullyConnected with a
    /// plain float matmul, which looks equivalent, keeps every label the same on
    /// easy text, and drifts by ~1.3e-02 exactly where it hurts: short messages.
    func testProbabilitiesMatchReference() {
        var worst = 0.0
        var worstText = ""
        for v in LangDetectFixtures.vectors {
            guard let top = detector.ranked(v.text).first, top.code == v.top else { continue }
            let diff = abs(top.probability - v.prob)
            if diff > worst { worst = diff; worstText = v.text }
        }
        XCTAssertLessThan(worst, 1e-5,
                          "probabilities drifted (worst on \(worstText.debugDescription)) — "
                          + "check that hybridFC still runs the dynamic-range int8 kernel")
    }

    /// The detector must survive inputs that are not sentences.
    func testDegenerateInputsDoNotCrash() {
        for text in ["", " ", "\n\n", "?", "😀😀😀", String(repeating: "a", count: 5000),
                     "\u{0}\u{0}", "İ", "ẞ", "\u{FFFD}"] {
            _ = detector.ranked(text)          // must not trap
        }
    }

    /// Token count is capped by `max_splits`, however long the input.
    func testLongInputIsTruncatedNotUnbounded() {
        let (_, tokens) = detector.ngramIDs(String(repeating: "xin chào ", count: 400))
        XCTAssertLessThanOrEqual(tokens, 128)
    }

    /// Repeated calls must agree — the detector holds no state between them, which
    /// is what makes it safe to call from the main actor and from `ChatService`'s
    /// background work without synchronisation.
    func testRepeatedCallsAreDeterministic() {
        let texts = ["Tôi rất thích cái này", "İstanbul İİİ", "Guten Morgen", "ẞẞẞẞẞ"]
        for text in texts {
            let first = detector.ranked(text).prefix(5).map(\.code)
            for _ in 0..<20 {
                XCTAssertEqual(detector.ranked(text).prefix(5).map(\.code), first,
                               "non-deterministic for \(text.debugDescription)")
            }
        }
    }

    /// Concurrent use from several queues must not corrupt results.
    func testConcurrentCallsAgreeWithSerialOnes() {
        let texts = LangDetectFixtures.vectors.prefix(40).map(\.text)
        let expected = texts.map { detector.ranked($0).first?.code ?? "" }
        let results = UnsafeMutableBufferPointer<Int>.allocate(capacity: texts.count)
        defer { results.deallocate() }
        results.initialize(repeating: 0)
        DispatchQueue.concurrentPerform(iterations: texts.count) { i in
            results[i] = (self.detector.ranked(texts[i]).first?.code ?? "") == expected[i] ? 1 : 0
        }
        XCTAssertEqual(results.reduce(0, +), texts.count,
                       "concurrent results diverged from serial ones — scratch is being shared")
    }
}
