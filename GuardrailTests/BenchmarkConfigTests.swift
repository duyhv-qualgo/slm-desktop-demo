import XCTest
@testable import SLMChat

// The benchmark harness's own configuration.
//
// Its numbers are only as good as its inputs: prompt length decides whether
// prefill is measured where throughput has plateaued or in the noisy
// short-prompt regime, and prompt + forced decode must fit one sequence.

/// What the benchmark harness actually measures, in tokens.
///
/// `BenchmarkHarness.fixedPrompt` is the single prompt behind every published
/// number in the Benchmark screen, so its length decides whether those numbers
/// describe batched prefill throughput or mostly fixed per-call overhead.
final class BenchmarkPromptTests: XCTestCase {

    func testFixedPromptTokenCount() throws {
        var any = false
        for option in ModelCatalog.taskModels where option.isAvailable {
            guard let path = ResourceLookup.path(option.resourceName, ext: "gguf") else { continue }
            let lm = ChatLM()
            guard lm.load(modelPath: path, backend: SLMBackend.cpu, cacheDir: nil, loraPath: nil)
            else { continue }
            any = true
            let raw = lm.tokenCount(of: BenchmarkHarness.fixedPrompt)

            // What the model really sees: the harness passes system:nil, history:[],
            // so the chat template wraps this one user turn. Measure the delta by
            // running it and reading the engine's own prompt token count.
            _ = lm.generate(system: nil, history: [], user: BenchmarkHarness.fixedPrompt,
                            maxTokens: 1, temperature: 0.1, cacheable: false, adapterPath: nil)
            let templated = lm.lastTurnMetrics?.promptTokenCount ?? -1

            print("BMPROMPT [\(option.id)] raw=\(raw) tok | after chat template=\(templated) tok")

            // Pin the length. The prompt is calibrated to ~512 tokens so prefill is
            // measured where throughput has plateaued (see fixedPrompt's doc), and
            // an innocuous copy edit would otherwise silently move the benchmark's
            // operating point. Vocabularies differ, so this is a band, not a value.
            XCTAssertGreaterThan(templated, 480,
                                 "\(option.id): benchmark prompt got shorter — prefill "
                                 + "would be measured in the noisy short-prompt regime")
            XCTAssertLessThan(templated, 560,
                              "\(option.id): benchmark prompt got longer — with a 512-token "
                              + "forced decode this must stay well inside kBenchmarkCtx (2048)")
        }
        try XCTSkipUnless(any, "No .gguf in the bundle.")
    }
}

/// The benchmark's own configuration must actually run: ~512 prompt tokens plus a
/// 512-token FORCED decode (EOS ignored) has to fit one sequence.
///
/// This is the case that motivated `kBenchmarkCtx`. Under the app's 1024-token
/// slots, 512 + 512 lands exactly on the ceiling — which would corrupt a
/// benchmark rather than fail it, since the harness reports whatever it got.
final class BenchmarkCapacityTests: XCTestCase {

    func testFixedModeFitsPromptPlusForcedDecode() throws {
        guard let name = ModelCatalog.taskModels.first(where: { $0.isAvailable })?.resourceName,
              let path = ResourceLookup.path(name, ext: "gguf") else {
            throw XCTSkip("No .gguf in the bundle.")
        }
        let cfg = BenchmarkConfig.fixed
        let lm = ChatLM()
        guard lm.load(modelPath: path, backend: SLMBackend.cpu, cacheDir: nil,
                      loraPath: nil, enableBenchmark: true,
                      forcedDecodeTokens: cfg.engineOptions.forcedDecodeTokens,
                      activation: .auto) else {
            throw XCTSkip("benchmark load failed: \(lm.lastError)")
        }

        let info = lm.benchmarkStream(system: nil, history: [],
                                      user: BenchmarkHarness.fixedPrompt,
                                      maxTokens: cfg.maxTokens,
                                      temperature: cfg.temperature) { _, _ in }
        let m = try XCTUnwrap(lm.lastTurnMetrics)
        print("""
        BMCAP in=\(m.promptTokenCount) tok + out=\(m.decodeTokenCount) tok \
        = \(m.promptTokenCount + m.decodeTokenCount) | prefill \
        \(String(format: "%.1f", m.prefillTokensPerSecond)) tok/s | decode \
        \(String(format: "%.1f", m.decodeTokensPerSecond)) tok/s | stop=\(m.stopReason)
        """)

        // The harness reads ChatLMBenchmarkInfo, the per-turn line reads
        // ChatLMTurnMetrics. They must agree — they described the same generation,
        // and they disagreed for as long as only one of them had the prefill-timer
        // fix.
        let bi = try XCTUnwrap(info)
        XCTAssertEqual(bi.prefillTokenCount, m.prefillTokenCount)
        XCTAssertEqual(bi.prefillTokensPerSecond, m.prefillTokensPerSecond, accuracy: 0.5,
                       "benchmark and per-turn prefill rates must match")
        XCTAssertLessThan(bi.prefillTokensPerSecond, 100_000,
                          "implausible benchmark prefill rate — the unreliable "
                          + "t_p_eval_ms counter is back")
        XCTAssertEqual(Int32(m.decodeTokenCount), cfg.maxTokens,
                       "forced decode must produce exactly maxTokens, EOS ignored")
        XCTAssertNotEqual(m.stopReason, "error", lm.lastError)
        XCTAssertGreaterThan(m.prefillTokensPerSecond, 0)
    }
}

/// Does prefill throughput depend on how long the prompt is?
///
/// It must, if any fixed per-call cost exists: a short prompt divides that cost
/// over few tokens. This decides whether a ~40-token benchmark prompt reports
/// batched throughput or mostly overhead.
final class PrefillLengthSweepTests: XCTestCase {

    func testThroughputVersusPromptLength() throws {
        var any = false
        for option in ModelCatalog.taskModels where option.isAvailable {
            guard let path = ResourceLookup.path(option.resourceName, ext: "gguf") else { continue }
            let lm = ChatLM()
            guard lm.load(modelPath: path, backend: SLMBackend.cpu, cacheDir: nil, loraPath: nil)
            else { continue }
            any = true
            lm.prefixReuseEnabled = false   // every run pays a full prompt pass

            let unit = "The quick brown fox jumps over the lazy dog near the river bank. "
            for reps in [1, 4, 10, 20, 40, 70] {   // 70 keeps us under kCtxPerSlot
                let prompt = String(repeating: unit, count: reps)
                _ = lm.generate(system: nil, history: [], user: prompt,
                                maxTokens: 2, temperature: 0.1,
                                cacheable: false, adapterPath: nil)
                guard let m = lm.lastTurnMetrics else { continue }
                print(String(format: "SWEEP [%@] in=%4d tok  prefill %.3fs  %7.1f tok/s",
                             option.id, m.prefillTokenCount, m.prefillSeconds,
                             m.prefillTokensPerSecond))
            }
        }
        try XCTSkipUnless(any, "No .gguf in the bundle.")
    }
}
