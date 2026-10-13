import XCTest
@testable import SLMChat

/// Guards the numbers in the `[SLMPerf]` line — the ones every latency decision
/// is made from.
///
/// These fail silently in the worst way: a wrong throughput figure still prints,
/// still looks like a measurement, and sends you optimizing the wrong stage. The
/// bug this was written for reported a 269-token prompt pass as
/// `prefill 0.000s (5977777.8 tok/s)` while the same turn's `ttft` was 3.450 s —
/// llama.cpp's `t_p_eval_ms` accumulates inside `synchronize()` and intermittently
/// lands near zero. `TOTAL` stayed correct throughout, so nothing looked broken;
/// the missing seconds were just silently reattributed to `overhead`.
///
/// Runs a real generation against the bundled checkpoint, so it's slower than the
/// rest of the suite. That is the point — the defect only appears in the timing
/// of an actual prompt pass and cannot be reproduced with a stub.
final class TurnMetricsTests: XCTestCase {

    /// The model this test measures against. Falls back across the catalog so the
    /// test skips cleanly rather than failing on a machine with a different subset
    /// of the (gitignored) weights.
    private func loadedEngine() throws -> ChatLM {
        let candidates = ModelCatalog.taskModels.map(\.resourceName)
        guard let name = candidates.first(where: { ResourceLookup.path($0, ext: "gguf") != nil }),
              let path = ResourceLookup.path(name, ext: "gguf") else {
            throw XCTSkip("No .gguf in the bundle — these are gitignored binaries.")
        }
        let lm = ChatLM()
        guard lm.load(modelPath: path, backend: SLMBackend.cpu, cacheDir: nil, loraPath: nil) else {
            throw XCTSkip("Model at \(name).gguf failed to load: \(lm.lastError)")
        }
        return lm
    }

    /// A prompt long enough that its pass is unambiguously measurable — the bug
    /// only showed on multi-token prefills, and a 1-token tail is a different path.
    private let longPrompt = String(
        repeating: "The quick brown fox jumps over the lazy dog near the river. ", count: 8)

    /// A multi-token prompt pass must report a real duration and a physically
    /// possible rate.
    ///
    /// The upper bound is the actual regression guard. A too-low `prefillSeconds`
    /// doesn't look wrong on its own — but divided into the token count it yields
    /// millions of tokens per second, which is unmistakable.
    func testPrefillIsMeasuredNotReportedAsZero() throws {
        let lm = try loadedEngine()
        _ = lm.generate(system: nil, history: [], user: longPrompt,
                        maxTokens: 8, temperature: 0.1, cacheable: false, adapterPath: nil)

        let m = try XCTUnwrap(lm.lastTurnMetrics)
        XCTAssertGreaterThan(m.prefillTokenCount, 1, "prompt should not have collapsed to one token")
        XCTAssertGreaterThan(m.prefillSeconds, 0,
                             "a multi-token prompt pass cannot take zero time")
        XCTAssertLessThan(m.prefillTokensPerSecond, 100_000,
                          "implausible prefill rate — prefillSeconds is being under-measured "
                          + "(observed 5,977,777 tok/s when this regressed)")
        XCTAssertGreaterThan(m.prefillTokensPerSecond, 0)
    }

    /// The breakdown must add up to the whole. `overhead` is derived by
    /// subtraction, so an under-measured stage doesn't shrink `TOTAL` — it inflates
    /// `overhead` instead, which is exactly how the bug hid.
    func testBreakdownAccountsForTheTotal() throws {
        let lm = try loadedEngine()
        _ = lm.generate(system: nil, history: [], user: longPrompt,
                        maxTokens: 8, temperature: 0.1, cacheable: false, adapterPath: nil)

        let m = try XCTUnwrap(lm.lastTurnMetrics)
        XCTAssertGreaterThan(m.totalSeconds, 0)
        XCTAssertEqual(m.prefillSeconds + m.decodeSeconds + m.overheadSeconds,
                       m.totalSeconds, accuracy: 0.001,
                       "prefill + decode + overhead must reconstruct TOTAL")
        // With the prompt pass measured, tokenizing and templating are all that's
        // left outside the model. If overhead dominates a turn again, a stage has
        // stopped being measured.
        XCTAssertLessThan(m.overheadSeconds, m.totalSeconds * 0.5,
                          "overhead should not dominate a turn — it absorbs whatever "
                          + "prefill/decode fail to account for")
    }

    /// TTFT covers the prompt pass plus the first token, so it can never be less
    /// than the prompt pass alone.
    func testTimeToFirstTokenCoversThePrefill() throws {
        let lm = try loadedEngine()
        _ = lm.generate(system: nil, history: [], user: longPrompt,
                        maxTokens: 8, temperature: 0.1, cacheable: false, adapterPath: nil)

        let m = try XCTUnwrap(lm.lastTurnMetrics)
        XCTAssertGreaterThanOrEqual(m.timeToFirstTokenSeconds, m.prefillSeconds)
        XCTAssertLessThanOrEqual(m.timeToFirstTokenSeconds, m.totalSeconds)
    }

    /// The adapter fields are always populated, so the `lora=` term in the log is
    /// one parseable shape rather than two. Loaded with `loraPath: nil`, so this
    /// pins the bare-weights spelling specifically.
    func testAdapterFieldsAreAlwaysReported() throws {
        let lm = try loadedEngine()
        _ = lm.generate(system: nil, history: [], user: longPrompt,
                        maxTokens: 4, temperature: 0.1, cacheable: false, adapterPath: nil)

        let m = try XCTUnwrap(lm.lastTurnMetrics)
        XCTAssertEqual(m.adapterName, "none")
        XCTAssertEqual(m.adapterState, "same")
        XCTAssertGreaterThanOrEqual(m.adapterSeconds, 0)
        XCTAssertTrue(m.summaryLine.contains("lora=none/same"), m.summaryLine)
        for field in ["prefill", "ttft", "tok/s", "decode", "overhead", "TOTAL", "stop="] {
            XCTAssertTrue(m.summaryLine.contains(field), "missing \(field): \(m.summaryLine)")
        }
    }
}

/// Prefill-throughput probe, run headless so it doesn't need the app's UI.
///
/// Exists to A/B a single engine setting: whether the BLAS (Accelerate) device is
/// left in the CPU device list. It reports the ratio rather than raw tok/s,
/// because the absolute numbers are simulator numbers and don't transfer to a
/// phone — but "is prefill actually batched?" does.
final class PrefillThroughputProbe: XCTestCase {

    /// Long enough that the prompt pass dominates and a per-token fallback path is
    /// obvious. ~350+ tokens, comfortably inside the 1024-token per-slot budget.
    private let prompt = String(
        repeating: "The quick brown fox jumps over the lazy dog near the wide river bank. ",
        count: 40)

    func testReportPrefillVersusDecode() throws {
        // EVERY bundled checkpoint, not just the first one available. The earlier
        // version measured whichever came first in the catalog (gemma-3 270M) and
        // reported it as if it described the app, which runs qwen3-0.6b-v6 — a
        // 2.2x larger model on a different quant family. A throughput probe that
        // silently measures a different model than the one shipping is worse than
        // no probe.
        let available = ModelCatalog.taskModels.filter(\.isAvailable)
        guard !available.isEmpty else { throw XCTSkip("No .gguf in the bundle.") }

        for option in available {
            guard let path = ResourceLookup.path(option.resourceName, ext: "gguf") else { continue }
            let pack = LoRACatalog.adapters(forBaseModel: option.id).first.flatMap {
                LoRACatalog.bundledPath(for: $0)
            }
            for adapter in [nil, pack] {
                if adapter == nil && pack != nil { } // measure bare first, then adapted
                let lm = ChatLM()
                guard lm.load(modelPath: path, backend: SLMBackend.cpu,
                              cacheDir: nil, loraPath: adapter) else { continue }
                _ = lm.generate(system: nil, history: [], user: prompt,
                                maxTokens: 24, temperature: 0.1,
                                cacheable: false, adapterPath: nil)
                guard let m = lm.lastTurnMetrics else { continue }
                let ratio = m.decodeTokensPerSecond > 0
                    ? m.prefillTokensPerSecond / m.decodeTokensPerSecond : 0
                print(String(format: "PROBE %@ lora=%@ prefill=%.1f decode=%.1f RATIO=%.2fx",
                             option.id, adapter == nil ? "no" : "yes",
                             m.prefillTokensPerSecond, m.decodeTokensPerSecond, ratio))
            }
        }
    }
}

/// Proves the prefix-reuse switch does what the log claims: with it off, a repeated
/// prompt must re-evaluate every token, so `prefill tok/s` is a true throughput
/// figure rather than a rate over whatever the cache happened to leave.
final class PrefixReuseSwitchTests: XCTestCase {

    private let prompt = String(
        repeating: "The quick brown fox jumps over the lazy dog near the wide river bank. ",
        count: 20)

    private func engine() throws -> ChatLM {
        guard let name = ModelCatalog.taskModels.first(where: { $0.isAvailable })?.resourceName,
              let path = ResourceLookup.path(name, ext: "gguf") else {
            throw XCTSkip("No .gguf in the bundle.")
        }
        let lm = ChatLM()
        guard lm.load(modelPath: path, backend: SLMBackend.cpu, cacheDir: nil, loraPath: nil) else {
            throw XCTSkip("load failed: \(lm.lastError)")
        }
        return lm
    }

    /// `cacheable: true` so the call is eligible for a slot — that is the only
    /// configuration where reuse could kick in, and therefore the only one where
    /// disabling it proves anything.
    private func run(_ lm: ChatLM) -> ChatLMTurnMetrics? {
        _ = lm.generate(system: nil, history: [], user: prompt,
                        maxTokens: 4, temperature: 0.1, cacheable: true, adapterPath: nil)
        return lm.lastTurnMetrics
    }

    func testReuseOnSecondCallSkipsMostOfThePrompt() throws {
        let lm = try engine()
        lm.prefixReuseEnabled = true
        _ = run(lm)                                  // populate the slot
        let second = try XCTUnwrap(run(lm))          // identical prompt
        XCTAssertGreaterThan(second.cachedPrefixTokenCount, 0,
                             "an identical repeated prompt should hit the cache")
        XCTAssertLessThan(second.prefillTokenCount, second.promptTokenCount / 2,
                          "most of the prompt should have been served from cache")
    }

    func testReuseOffReEvaluatesEveryToken() throws {
        let lm = try engine()
        lm.prefixReuseEnabled = false
        _ = run(lm)
        let second = try XCTUnwrap(run(lm))
        XCTAssertEqual(second.cachedPrefixTokenCount, 0, "nothing may be reused")
        XCTAssertEqual(second.prefillTokenCount, second.promptTokenCount,
                       "every prompt token must be evaluated")
        // The match is still REPORTED — that's the point of measuring this way:
        // one run gives both the raw rate and the hit rate you gave up to get it.
        XCTAssertGreaterThan(second.prefixMatchTokenCount, 0,
                             "the match that WOULD have been reused should still be reported")
        print(String(format: "RAWPREFILL %d tok in %.3fs = %.1f tok/s (reuse off, would have reused %d)",
                     second.prefillTokenCount, second.prefillSeconds,
                     second.prefillTokensPerSecond, second.prefixMatchTokenCount))
    }
}
