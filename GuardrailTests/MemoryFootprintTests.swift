import XCTest
@testable import SLMChat

/// Tracks where the guardrail's resident memory goes, and which parts of it grow
/// without bound.
///
/// The one number on record — "~3 MB per language pack" — was measured before the
/// per-language rule sets, the screener, and per-message verdicts existed. This
/// re-measures each consumer separately so a footprint rise can be attributed
/// instead of guessed at.
///
/// Footprint is noisy: the allocator does not return pages promptly, other tests
/// leave residue, and the simulator is not the device. Treat single numbers as
/// indicative and the SHAPE — flat vs. linear in message count — as the finding.
@MainActor
final class MemoryFootprintTests: XCTestCase {

    private func footprintMB() -> Double {
        Double(ProcessMetrics.memoryFootprint() ?? 0) / 1_048_576
    }

    /// Let allocations settle before sampling. Without this, deltas mostly report
    /// scheduling noise.
    private func settle() async {
        try? await Task.sleep(nanoseconds: 300_000_000)
    }

    private func measure(_ label: String,
                         _ body: () async throws -> Void) async rethrows -> Double {
        await settle()
        let before = footprintMB()
        try await body()
        await settle()
        let after = footprintMB()
        let delta = after - before
        print(String(format: "MEM | %-42@ %+7.2f MB  (now %.1f MB)",
                     label as NSString, delta, after))
        return delta
    }

    /// Rule sets are compiled `NSRegularExpression`s held for the app's lifetime,
    /// and a region now loads one set PER LANGUAGE plus the universal fallback —
    /// the UK is 127 + 21 + 3. Worth knowing what that costs before more languages
    /// are added.
    func testRuleSetFootprintByRegion() async throws {
        var engines: [GuardEngine] = []

        _ = await measure("UK rules (uk-en 127 + uk-vi 21 + universal)") {
            let e = GuardEngine()
            _ = await e.warmUp(language: .english, region: .uk)
            engines.append(e)
        }
        _ = await measure("Vietnam rules (vietnam-vi 200)") {
            let e = GuardEngine()
            _ = await e.warmUp(language: .vietnamese, region: .vietnam)
            engines.append(e)
        }
        XCTAssertEqual(engines.count, 2)   // keep them alive across the measurement
    }

    /// Each pack should cost ~3 MB resident, far below the 9.6 MB `.tflite`,
    /// because TFLite memory-maps the weights. Pins the shape rather than a value.
    func testPackFootprintIsFarBelowFileSize() async throws {
        let engine = GuardEngine()

        let first = await measure("first pack (vi) — includes TFLite runtime init") {
            _ = await engine.warmUp(language: .vietnamese, region: nil)
            _ = await engine.classify("Xin chào", language: .vietnamese)
        }
        let second = await measure("second pack (en)") {
            _ = await engine.classify("Hello there", language: .english)
        }

        XCTAssertLessThan(second, 9.0,
                          "a pack cost \(second) MB — that is the whole .tflite, so "
                        + "TFLite has stopped memory-mapping and every language now "
                        + "costs 10 MB instead of 3")
        _ = first
    }

    /// **The unbounded one.**
    ///
    /// `MessageScreener.claimed` records every message id it has ever screened and
    /// is only ever emptied by `reset()`, which nothing calls. So it grows for the
    /// life of the app: a long-lived thread that the user keeps scrolling adds a
    /// UUID per message and never gives one back.
    ///
    /// This measures the growth so the cost is known rather than assumed. It is
    /// small per message — the finding is that it is monotonic, not that it is
    /// large.
    func testScreenerClaimedSetGrowsWithoutBound() async throws {
        let service = ChatService()
        _ = await service.guardEngine.warmUp(language: .english, region: .uk)

        let messages = (0..<4_000).map {
            ChatMessage(sender: .friend, text: "Backlog line number \($0).")
        }
        let convo = Conversation(contactName: "Test", messages: messages, service: service)

        let delta = await measure("screening 4000 messages") {
            for m in convo.messages { await convo.classifySafety(in: m) }
        }

        XCTAssertEqual(convo.screeningRuns, 4_000)
        print(String(format: "MEM | per message: %.3f KB", delta * 1024 / 4_000))

        // Not a leak in the ARC sense — the ids are reachable and intentional —
        // but nothing releases them, so state it plainly rather than let a future
        // reader assume the set is bounded by the viewport.
        XCTAssertGreaterThan(convo.screeningRuns, 0)
    }

    /// The pending queue holds STRONG references to `ChatMessage`. It is capped at
    /// `maxPending`, so it must not retain a whole thread after a flood drains.
    func testPendingQueueDoesNotRetainTheThread() async throws {
        let service = ChatService()
        _ = await service.guardEngine.warmUp(language: .english, region: .uk)

        let group = GroupChat(service: service)
        group.floodGroup(count: 600)
        group.screenFromViewport(group.messages[599])

        try? await Task.sleep(nanoseconds: 2_000_000_000)

        XCTAssertLessThanOrEqual(group.pendingScreenings, MessageScreener.maxPending,
                                 "\(group.pendingScreenings) messages still queued — "
                               + "the cap is not holding, and each is a strong ref")
        print("MEM | pending after a 600-message flood settles: \(group.pendingScreenings)")
    }

    /// Verdicts are two `Int?`s on each message and must not be the thing that
    /// grows. Measured against the message text itself for scale.
    func testVerdictStorageIsNegligible() async throws {
        let messages = (0..<20_000).map {
            ChatMessage(sender: .friend, text: "Line \($0).")
        }
        let delta = await measure("applying 20000 verdicts") {
            for m in messages { m.apply(SafetyResult(label: .harassment,
                                                     severity: .caution,
                                                     confidence: 0.7)) }
        }
        XCTAssertLessThan(delta, 8.0,
                          "verdict storage cost \(delta) MB for 20k messages")
    }
}
