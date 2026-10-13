import XCTest
@testable import SLMChat

/// Checks the long-message windowing in `GuardEngine`.
///
/// The model reads 128 tokens and `GuardTokenizer.encode` truncates silently, so
/// before this a long message was screened on its opening and nothing else — a
/// violation in the tail returned a confident Clean. These pin that long text is
/// split into whole-sentence windows, that every window is screened, and that the
/// worst window wins.
final class GuardWindowTests: XCTestCase {

    private func tokenizer(maxLen: Int = 128) throws -> GuardTokenizer {
        let pack = try XCTUnwrap(GuardPackCatalog.pack(for: .vietnamese))
        let url = try XCTUnwrap(ResourceLookup.url(pack.tokenizerResource, ext: "json"))
        return try XCTUnwrap(GuardTokenizer(tokenizerJSONURL: url, maxLen: maxLen))
    }

    /// The common case. Anything that already fits must come back as one window,
    /// byte-identical, so short messages keep the exact path they always had.
    func testShortTextIsASingleWindow() throws {
        let tok = try tokenizer()
        for text in ["Xin chào bạn", "Let's meet tomorrow at 3pm.", "ok"] {
            XCTAssertEqual(GuardEngine.windows(text, tokenizer: tok), [text])
        }
    }

    /// Every window must fit the model's input, or the tokenizer silently
    /// truncates it and the split achieved nothing.
    func testEveryWindowFitsTheModelInput() throws {
        let tok = try tokenizer()
        let long = String(repeating: "Hôm nay trời đẹp quá và tôi muốn đi dạo công viên. ",
                          count: 40)
        let windows = GuardEngine.windows(long, tokenizer: tok)

        XCTAssertGreaterThan(windows.count, 1, "long text should split")
        for w in windows {
            XCTAssertLessThanOrEqual(tok.tokenCount(of: w), tok.maxLen,
                                     "window exceeds the model window: \(w.prefix(40))…")
        }
    }

    /// Windows are whole sentences, not fixed-offset cuts — a fragment classifies
    /// worse than a complete thought.
    func testWindowsBreakOnSentences() throws {
        let tok = try tokenizer()
        let long = String(repeating: "Câu này hoàn chỉnh và có dấu chấm câu rõ ràng. ", count: 40)
        for w in GuardEngine.windows(long, tokenizer: tok) {
            XCTAssertTrue(w.hasSuffix(".") || w.hasSuffix("!") || w.hasSuffix("?"),
                          "window did not end on a sentence boundary: …\(w.suffix(30))")
        }
    }

    /// One sentence per window is the whole point: a violation must dominate its
    /// own window or the model reads the window as its benign majority. Measured —
    /// packing this threat behind a single benign sentence returns Clean.
    func testEachSentenceGetsItsOwnWindow() throws {
        let tok = try tokenizer()
        let text = "Hôm nay trời đẹp quá. Tao sẽ giết mày."
        let ws = GuardEngine.windows(text, tokenizer: tok)

        XCTAssertTrue(ws.contains("Tao sẽ giết mày."),
                      "the threat must be screened on its own, got: \(ws)")
        XCTAssertTrue(ws.contains(text), "the whole message should also be screened for context")
    }

    /// A single sentence longer than the whole window still has to be split, or
    /// one runaway paragraph would be truncated exactly as before.
    func testOverlongSentenceIsSplit() throws {
        let tok = try tokenizer()
        let runOn = String(repeating: "từ ", count: 400)   // no terminator at all
        let windows = GuardEngine.windows(runOn, tokenizer: tok)

        XCTAssertGreaterThan(windows.count, 1)
        for w in windows {
            XCTAssertLessThanOrEqual(tok.tokenCount(of: w), tok.maxLen)
        }
    }

    /// Pathological input must stay bounded — the cost is one inference per
    /// window, so an unbounded split would be an unbounded stall.
    func testWindowCountIsCapped() throws {
        let tok = try tokenizer()
        let huge = String(repeating: "Đây là một câu tiếng Việt bình thường. ", count: 2_000)
        XCTAssertLessThanOrEqual(GuardEngine.windows(huge, tokenizer: tok).count,
                                 GuardEngine.maxWindows)
    }

    /// The point of the whole exercise: a violation past the 128-token mark is
    /// found. The filler is benign and long enough to push the threat out of the
    /// first window on its own.
    func testViolationAfterTheFirstWindowIsCaught() async throws {
        let tok = try tokenizer()
        let filler = String(repeating: "Hôm nay trời đẹp quá và tôi muốn đi dạo công viên. ",
                            count: 10)
        let text = filler + "Tao sẽ giết mày."

        XCTAssertGreaterThan(tok.tokenCount(of: filler), tok.maxLen,
                             "filler must exceed one window for this test to mean anything")
        XCTAssertLessThanOrEqual(GuardEngine.windows(text, tokenizer: tok).count,
                                 GuardEngine.maxWindows,
                                 "test text must stay under the cap, or it proves nothing")

        let engine = GuardEngine()
        let res = await engine.warmUp(language: .vietnamese, region: nil)
        XCTAssertTrue(res.ok, "warm-up failed: \(res.error ?? "unknown")")

        // Model-only: the rules scan the full text already and would catch this
        // without any windowing, which would make the test pass for the wrong reason.
        let verdict = await engine.classifyModelOnly(text, language: .vietnamese)
        XCTAssertTrue(verdict.isUnsafe,
                      "threat past the first window was missed — windowing is not working")
        XCTAssertEqual(verdict.label, .violence)
    }

    /// Truncated, the same text reads clean — this is the regression that
    /// windowing fixes, pinned so it can't quietly come back.
    func testSameTextIsMissedWithoutWindowing() throws {
        let tok = try tokenizer()
        let filler = String(repeating: "Hôm nay trời đẹp quá và tôi muốn đi dạo công viên. ",
                            count: 10)
        let text = filler + "Tao sẽ giết mày."

        // What the model would have seen before: the first 128 tokens only.
        let truncated = tok.encode(text)
        XCTAssertEqual(truncated.ids.count, tok.maxLen)
        XCTAssertFalse(truncated.ids.contains(where: { $0 == 0 }),
                       "input is fully packed, so the tail was dropped")

        XCTAssertGreaterThan(GuardEngine.windows(text, tokenizer: tok).count, 1,
                             "windowing is what recovers the dropped tail")
    }
}
