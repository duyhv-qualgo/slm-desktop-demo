import XCTest
@testable import SLMChat

/// Measures the MAIN-ACTOR work that follows a classification.
///
/// Written after a demo run reported two very different numbers for the same
/// message: `[Guard] … 39.1ms` and `[GuardDemo] … 935.3ms`, with a 0.91 s hang
/// logged between them. The gap is not the guardrail. `[Guard]` times
/// `guardEngine.classify`; `[GuardDemo]` times all of `classifySafety`, which
/// also runs `message.apply(result)` and `persist()` on the main actor.
///
/// Every other test in this suite leaves `onPersist` nil, so persistence is free
/// and this cost is invisible. That is exactly why it went unnoticed: the tests
/// were measuring a code path the app doesn't have.
@MainActor
final class ScreeningTailTests: XCTestCase {

    private var service: ChatService!

    override func setUp() async throws {
        try await super.setUp()
        service = ChatService()
        let res = await service.guardEngine.warmUp(language: .english,
                                                   region: .uk)
        XCTAssertTrue(res.ok, "guard warm-up failed: \(res.error ?? "unknown")")
    }

    override func tearDown() {
        service = nil
        super.tearDown()
    }

    /// Persist cost scales with the WHOLE INBOX, not with the message.
    ///
    /// `ConversationStore.save` encodes every conversation and rewrites the file
    /// atomically, and `Conversation.persist` calls it once per verdict. So the
    /// price of classifying one short message is set by how much history the user
    /// has — which is the wrong thing for it to depend on, and why a long thread
    /// turns a 10 ms classification into a visible hang.
    func testPersistCostScalesWithTheInboxNotTheMessage() async throws {
        func measure(historyCount: Int) async -> (classify: Double, tail: Double) {
            let msgs = (0..<historyCount).map {
                ChatMessage(sender: .friend, text: "History message number \($0).")
            }
            let convo = Conversation(contactName: "Test", messages: msgs, service: service)

            // Wired the way the app wires it: a verdict triggers a full re-encode.
            var snapshots: [StoredConversation] = []
            convo.onPersist = { snapshots = [convo.snapshot] }

            let fresh = ChatMessage(sender: .friend, text: "Let's meet tomorrow at 3pm.")
            convo.messages.append(fresh)

            let t0 = DispatchTime.now().uptimeNanoseconds
            let verdict = await service.classify(fresh.text, language: .english)
            let t1 = DispatchTime.now().uptimeNanoseconds
            fresh.apply(verdict)
            convo.onPersist?()
            let t2 = DispatchTime.now().uptimeNanoseconds

            _ = snapshots
            return (Double(t1 - t0) / 1_000_000, Double(t2 - t1) / 1_000_000)
        }

        let small = await measure(historyCount: 10)
        let large = await measure(historyCount: 1_000)

        print("TAIL | 10 msgs   → classify \(String(format: "%.1f", small.classify))ms, "
            + "tail \(String(format: "%.1f", small.tail))ms")
        print("TAIL | 1000 msgs → classify \(String(format: "%.1f", large.classify))ms, "
            + "tail \(String(format: "%.1f", large.tail))ms")

        // Classification is indifferent to thread length; the tail is not. Stated
        // as a ratio because absolute numbers move with the machine.
        XCTAssertLessThan(small.tail, large.tail,
                          "snapshot cost should grow with history — if it doesn't, "
                        + "this test is no longer measuring what it claims")
    }

    /// The demo path end to end, so the number in the `[GuardDemo]` log has a
    /// test behind it. Records rather than asserts: the point is that the tail is
    /// visible, not that it is any particular size on this machine.
    func testDemoLineCostIsDominatedByTheTailOnALongThread() async throws {
        let msgs = (0..<800).map {
            ChatMessage(sender: .friend, text: "Earlier group traffic, line \($0).")
        }
        let convo = Conversation(contactName: "Test", messages: msgs, service: service)
        convo.onPersist = { _ = convo.snapshot }        // encode, as the store does

        let flagged = ChatMessage(
            sender: .friend,
            text: "I've got your nudes. Pay me £500 or I'll send them to all your contacts.")
        convo.messages.append(flagged)

        let started = DispatchTime.now().uptimeNanoseconds
        await convo.classifySafety(in: flagged)
        let total = Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000

        XCTAssertEqual(flagged.safety, .sexual)
        XCTAssertEqual(flagged.severity, .critical)
        print("TAIL | classifySafety on an 800-message thread: "
            + "\(String(format: "%.1f", total))ms total")
    }

    /// The coalesced path exists for this reason: a look-ahead sweep must not pay
    /// the snapshot cost once per message.
    func testCoalescedPersistDoesNotWritePerMessage() async throws {
        let msgs = (0..<200).map { ChatMessage(sender: .friend, text: "Backlog line \($0).") }
        let convo = Conversation(contactName: "Test", messages: msgs, service: service)

        var writes = 0
        convo.onPersist = { writes += 1 }

        convo.screenFromViewport(convo.messages[convo.messages.count - 1])

        // Long enough for the sweep to finish and the 500 ms debounce to fire.
        try? await Task.sleep(nanoseconds: 1_500_000_000)

        print("TAIL | \(convo.screeningRuns) verdicts → \(writes) persist calls")
        XCTAssertLessThan(writes, convo.screeningRuns,
                          "\(writes) writes for \(convo.screeningRuns) verdicts — "
                        + "the sweep is persisting per message")
    }
}
