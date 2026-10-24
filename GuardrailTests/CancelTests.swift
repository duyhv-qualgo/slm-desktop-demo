import XCTest
@testable import SLMChat

/// `ChatLM.cancel` stops a running generation early, from another thread, and
/// leaves the engine usable — and a cancel with nothing running is ignored.
final class CancelTests: XCTestCase {

    private let longAsk = "Write a long, detailed story about a lighthouse keeper. "
                        + "Keep going for many paragraphs."

    private func engine() throws -> ChatLM {
        guard let name = ModelCatalog.taskModels.first(where: { $0.isAvailable })?.resourceName,
              let path = ResourceLookup.path(name, ext: "gguf") else {
            throw XCTSkip("No .gguf in the bundle.")
        }
        let lm = ChatLM()
        guard lm.load(modelPath: path, backend: SLMBackend.cpu, loraPath: nil) else {
            throw XCTSkip("load failed: \(lm.lastError)")
        }
        return lm
    }

    func testCancelFromAnotherThreadStopsEarly() throws {
        let lm = try engine()
        let cap: Int32 = 400
        var chunks = 0
        var sawDone = false
        lm.stream(system: nil, history: [], user: longAsk,
                  maxTokens: cap, temperature: 0) { chunk, done in
            if done { sawDone = true; return }
            chunks += 1
            // Cancel from a different thread, as a UI would, once text is flowing.
            if chunks == 3 { DispatchQueue.global().async { lm.cancel() } }
        }
        let m = try XCTUnwrap(lm.lastTurnMetrics)
        XCTAssertEqual(m.stopReason, "cancelled")
        XCTAssertTrue(sawDone, "a cancelled stream must still deliver done=YES")
        XCTAssertLessThan(chunks, 100, "generation should stop soon after the cancel")
        XCTAssertEqual(lm.lastError, "", "a cancel is not an error")

        // The engine is still usable, and the cancel did not stick.
        let next = lm.generate(system: nil, history: [], user: "Say hi.",
                               maxTokens: 8, temperature: 0, adapterPath: nil)
        XCTAssertNotNil(next)
        XCTAssertNotEqual(lm.lastTurnMetrics?.stopReason, "cancelled")
    }

    func testCancelWhileIdleDoesNotCancelTheNextCall() throws {
        let lm = try engine()
        lm.cancel()
        _ = lm.generate(system: nil, history: [], user: "Say hi.",
                        maxTokens: 8, temperature: 0, adapterPath: nil)
        XCTAssertNotEqual(lm.lastTurnMetrics?.stopReason, "cancelled")
    }
}
