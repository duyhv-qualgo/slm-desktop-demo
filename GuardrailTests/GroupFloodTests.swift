import XCTest
@testable import SLMChat

/// Load tests for the screener under a community flood.
///
/// The scripted scenarios check that the guardrail says the right thing. These
/// check that it keeps up — a different failure, and the one that actually shows
/// on a busy group: verdicts arrive, but too late to be worth anything, because
/// the reader has already scrolled past.
@MainActor
final class GroupFloodTests: XCTestCase {

    private var service: ChatService!

    override func setUp() async throws {
        try await super.setUp()
        service = ChatService()
        // Load the UK rules explicitly. Without this the engine runs MODEL-ONLY:
        // the pack still loads lazily, so messages are screened and verdicts come
        // back — they are just missing every rule-based detection, and a test
        // that asserts on rule hits fails for a reason that has nothing to do
        // with what it is testing. (`ChatService.warmUp` would do this, but it
        // also loads the 8-second SLM, which no test here needs.)
        let res = await service.guardEngine.warmUp(language: .english,
                                                   region: .uk)
        XCTAssertTrue(res.ok, "guard warm-up failed: \(res.error ?? "unknown")")
        XCTAssertGreaterThan(service.guardEngine.ruleCount, 0, "UK rules did not load")
    }

    override func tearDown() {
        service = nil
        super.tearDown()
    }

    private func floodedGroup(_ count: Int) -> GroupChat {
        let g = GroupChat(service: service)
        g.floodGroup(count: count)
        return g
    }

    private func settle(_ g: GroupChat, timeout: TimeInterval = 30) async {
        let deadline = Date().addingTimeInterval(timeout)
        var last = -1
        while Date() < deadline {
            let runs = g.screeningRuns
            let done = g.messages.filter { $0.safetyLabel != nil }.count
            if runs == last && runs > 0 { return }      // no progress for a tick = drained
            last = runs
            _ = done
            try? await Task.sleep(nanoseconds: 200_000_000)
        }
    }

    /// The core claim of the whole split: a flood that nobody looks at costs
    /// nothing. 500 messages arrive and not one is classified.
    func testFloodCostsNothingUntilItIsRead() async throws {
        let g = floodedGroup(500)
        try? await Task.sleep(nanoseconds: 500_000_000)

        XCTAssertEqual(g.screeningRuns, 0,
                       "\(g.screeningRuns) classifications for a backlog nobody opened")
        XCTAssertEqual(g.messages.count, 500)
    }

    /// Opening the group screens the bottom of it and nothing else.
    func testOpeningAFloodedGroupScreensOnlyTheViewport() async throws {
        let g = floodedGroup(500)
        g.screenFromViewport(g.messages[g.messages.count - 1])
        await settle(g)

        XCTAssertLessThanOrEqual(g.screeningRuns, GroupChat.screenLookAhead + 1,
                                 "opened a 500-message group and ran \(g.screeningRuns) "
                               + "classifications")
    }

    /// **The failure mode this file exists to expose.**
    ///
    /// A fast scroll fires `onAppear` for row after row, and every one enqueues
    /// itself plus its look-ahead. The screener is serial at ~4.5 ms a message,
    /// so a thumb moving through 200 rows outruns it: the queue grows, and the
    /// verdicts for what is on screen NOW arrive behind verdicts for rows the
    /// reader has already passed.
    ///
    /// This test doesn't assert that the backlog is small — it measures how deep
    /// it gets, so the number is on record and a fix can be shown to change it.
    /// It fails only if the work is unbounded, i.e. if a scroll can enqueue the
    /// same message more than once.
    func testFastScrollBuildsABacklog() async throws {
        let g = floodedGroup(300)

        // A flick through 200 rows, as fast as the runloop will deliver them.
        for i in stride(from: g.messages.count - 1, through: 100, by: -1) {
            g.screenFromViewport(g.messages[i])
        }

        // Measured on `GroupChat`, not on the engine. The engine reports 0 here:
        // every enqueue is a Swift Task that has not reached its dispatch queue
        // yet, because this loop never yielded the main actor. The backlog is one
        // layer up, and that is where a flood control would have to read it.
        let depthAfterScroll = g.pendingScreenings
        let engineDepth = service.guardEngine.backlogDepth()
        await settle(g)

        XCTAssertLessThanOrEqual(g.screeningRuns, g.messages.count,
                                 "enqueued more work than there are messages — "
                               + "the in-flight gate is leaking under scroll")
        XCTAssertGreaterThan(depthAfterScroll, GroupChat.screenLookAhead,
                             "a 200-row flick queued only \(depthAfterScroll) messages — "
                           + "if this ever drops to viewport size, something is "
                           + "shedding work and the test below is measuring nothing")

        // Recorded, not asserted: this is the number a flood control would target.
        print("FLOOD | scrolled 200 rows in a 300-message group")
        print("FLOOD | pending screenings straight after the scroll: \(depthAfterScroll)")
        print("FLOOD | engine-level depth at that moment: \(engineDepth.pending) "
            + "(0 is expected — the work has not been submitted yet)")
        print("FLOOD | classifications run: \(g.screeningRuns)")
        print("FLOOD | screened: \(g.messages.filter { $0.safetyLabel != nil }.count)/300")
    }

    /// Throughput under sustained load, so the per-message cost is known for a
    /// flood rather than extrapolated from one quiet message.
    func testSustainedThroughput() async throws {
        let g = floodedGroup(120)

        let started = Date()
        for m in g.messages { await g.classifySafety(in: m) }
        let elapsed = Date().timeIntervalSince(started)

        let screened = g.messages.filter { $0.safetyLabel != nil }.count
        XCTAssertEqual(screened, 120, "not everything was screened")

        let perMessage = elapsed / 120 * 1000
        print("FLOOD | 120 messages in \(String(format: "%.2f", elapsed))s "
            + "= \(String(format: "%.1f", perMessage))ms each, "
            + "\(String(format: "%.0f", 120 / elapsed)) msg/s")

        // Generous — this is a floor on sanity, not a performance target. The
        // simulator runs on the Mac's CPU; a real iPhone XS will be slower.
        XCTAssertLessThan(perMessage, 100,
                          "\(perMessage)ms per message under sustained load")
    }

    /// A flood must not change what the guardrail SAYS. Load is allowed to make
    /// verdicts late; it is not allowed to make them wrong.
    func testFloodDoesNotChangeVerdicts() async throws {
        let g = floodedGroup(60)
        for m in g.messages { await g.classifySafety(in: m) }

        let lines = GuardrailDemo.flood(count: 60)
        for (msg, line) in zip(g.messages, lines) {
            XCTAssertEqual(msg.safety?.isUnsafe, line.shouldFlag,
                           "\(line.text.debugDescription) under load → "
                         + "\(msg.safety?.displayName ?? "nil")")
        }
    }
}
