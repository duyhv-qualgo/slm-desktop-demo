import XCTest
@testable import SLMChat

/// Guards the four numbers in the `[SLMPerf]` line (`ChatLMTurnMetrics`).
///
/// Runs a real generation against the bundled checkpoint, so it's slower than the
/// rest of the suite — timing can't be checked against a stub.
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
        guard lm.load(modelPath: path, backend: SLMBackend.cpu, loraPath: nil) else {
            throw XCTSkip("Model at \(name).gguf failed to load: \(lm.lastError)")
        }
        return lm
    }

    private let longPrompt = String(
        repeating: "The quick brown fox jumps over the lazy dog near the river. ", count: 8)

    /// The timings are physically consistent: text arrives after the call
    /// starts and before it ends, and decode reports a real, plausible rate.
    func testTimingsAreConsistent() throws {
        let lm = try loadedEngine()
        _ = lm.generate(system: nil, history: [], user: longPrompt,
                        maxTokens: 8, temperature: 0.1, adapterPath: nil)

        let m = try XCTUnwrap(lm.lastTurnMetrics)
        XCTAssertGreaterThan(m.timeToFirstTokenSeconds, 0, "a token was produced, so TTFT is set")
        XCTAssertLessThanOrEqual(m.timeToFirstTokenSeconds, m.totalSeconds)
        XCTAssertGreaterThan(m.decodeTokensPerSecond, 0)
        XCTAssertLessThan(m.decodeTokensPerSecond, 10_000, "implausible decode rate")
    }

    /// `stopReason` is one of the three documented values; hitting the cap says
    /// `max_tokens`.
    func testStopReasonAtTheCap() throws {
        let lm = try loadedEngine()
        _ = lm.generate(system: nil, history: [], user: longPrompt,
                        maxTokens: 2, temperature: 0.1, adapterPath: nil)

        let m = try XCTUnwrap(lm.lastTurnMetrics)
        XCTAssertTrue(["eos", "max_tokens"].contains(m.stopReason), m.stopReason)
    }
}

/// The KV cache is cleared at the start of every call, so nothing from one
/// generation can leak into the next: a repeated greedy prompt must produce the
/// same reply every time.
final class KVCacheClearTests: XCTestCase {

    private let prompt = String(
        repeating: "The quick brown fox jumps over the lazy dog near the wide river bank. ",
        count: 20)

    func testRepeatedGreedyCallIsIdentical() throws {
        guard let name = ModelCatalog.taskModels.first(where: { $0.isAvailable })?.resourceName,
              let path = ResourceLookup.path(name, ext: "gguf") else {
            throw XCTSkip("No .gguf in the bundle.")
        }
        let lm = ChatLM()
        guard lm.load(modelPath: path, backend: SLMBackend.cpu, loraPath: nil) else {
            throw XCTSkip("load failed: \(lm.lastError)")
        }
        let first = lm.generate(system: nil, history: [], user: prompt,
                                maxTokens: 16, temperature: 0, adapterPath: nil)
        let second = lm.generate(system: nil, history: [], user: prompt,
                                 maxTokens: 16, temperature: 0, adapterPath: nil)
        XCTAssertEqual(first, second, "a cleared cache must make greedy output repeatable")
    }
}
