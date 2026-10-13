import XCTest
@testable import SLMChat

/// Checks the split between the two safety-screening paths.
///
/// Messages used to be classified on arrival, all of them. That is right for a
/// message arriving at human typing speed — the verdict lands a frame or two
/// after the bubble renders — and wrong for a backlog, where a 200-message sync
/// meant 200 inferences for the handful anyone scrolls to, competing with the
/// decryption those same messages need.
///
/// So: live messages screen on arrival, bulk history screens when it reaches the
/// viewport. These pin that the bulk path really does defer, that scrolling
/// really does screen, and that the deferral can't leak unscreened text onto a
/// screen the user hasn't opened.
@MainActor
final class ViewportScreeningTests: XCTestCase {

    /// `Conversation` holds its service `unowned` — the store owns it for the
    /// app's lifetime. A test that passes a temporary crashes the moment
    /// screening touches it, so the service is retained here for the test's life.
    private var service: ChatService!

    override func setUp() {
        super.setUp()
        service = ChatService()
    }

    override func tearDown() {
        service = nil
        super.tearDown()
    }

    private func conversation(_ texts: [String] = []) -> Conversation {
        let msgs = texts.map { ChatMessage(sender: .friend, text: $0) }
        return Conversation(contactName: "Test", messages: msgs, service: service)
    }

    /// Wait for screening to settle. The work runs on the guard engine's serial
    /// `.utility` queue, so this polls rather than assuming a fixed duration —
    /// a fixed sleep would be flaky on a loaded machine and slow on an idle one.
    private func waitForScreening(_ convo: Conversation,
                                  count: Int,
                                  timeout: TimeInterval = 10) async -> Int {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            let done = convo.messages.filter { $0.safetyLabel != nil }.count
            if done >= count { return done }
            try? await Task.sleep(nanoseconds: 50_000_000)
        }
        return convo.messages.filter { $0.safetyLabel != nil }.count
    }

    /// A message that scrolled into view gets screened.
    func testViewportScreeningClassifiesTheVisibleMessage() async throws {
        let convo = conversation(["Let's meet tomorrow at 3pm."])
        XCTAssertNil(convo.messages[0].safetyLabel, "should start unscreened")

        convo.screenFromViewport(convo.messages[0])
        let done = await waitForScreening(convo, count: 1)

        XCTAssertEqual(done, 1, "the visible message was never screened")
    }

    /// The saving. A long backlog must stay unscreened until someone looks at it —
    /// this is the whole reason the bulk path exists, and if look-ahead ever grows
    /// to cover the entire thread the compute win is gone.
    func testBacklogBeyondLookAheadStaysUnscreened() async throws {
        let texts = (0..<60).map { "Backlog message number \($0)." }
        let convo = conversation(texts)

        // The user opens the thread at the bottom, as ChatView does.
        convo.screenFromViewport(convo.messages[convo.messages.count - 1])
        _ = await waitForScreening(convo, count: Conversation.screenLookAhead + 1)

        let screened = convo.messages.filter { $0.safetyLabel != nil }.count
        XCTAssertLessThanOrEqual(screened, Conversation.screenLookAhead + 1,
                                 "\(screened) of 60 screened — the bulk path is not deferring")
        XCTAssertGreaterThan(convo.messages.filter { $0.safetyLabel == nil }.count, 0,
                             "nothing was deferred")
    }

    /// Look-ahead runs in BOTH directions around the visible row.
    ///
    /// It used to run older-only, which was right while chats opened at the
    /// bottom — every unscreened message was above you. Now the thread opens at
    /// the first unread message, so the unread ones are BELOW and the reader
    /// travels downward. Older-only look-ahead would race away from the direction
    /// of travel and hand the reader unscreened rows.
    func testLookAheadCoversBothDirections() async throws {
        let texts = (0..<40).map { "Message number \($0)." }
        let convo = conversation(texts)

        let anchor = 20
        convo.screenFromViewport(convo.messages[anchor])
        _ = await waitForScreening(convo, count: 3)
        try? await Task.sleep(nanoseconds: 600_000_000)

        XCTAssertNotNil(convo.messages[anchor].safetyLabel, "the anchor itself")
        XCTAssertNotNil(convo.messages[anchor - 1].safetyLabel, "older neighbour")
        XCTAssertNotNil(convo.messages[anchor + 1].safetyLabel,
                        "NEWER neighbour — the direction a last-read-anchored "
                      + "reader actually scrolls")
        XCTAssertNil(convo.messages[39].safetyLabel,
                     "far messages must still wait — look-ahead is bounded")
    }

    /// `onAppear` fires per row, and the initial layout materializes several rows
    /// in one runloop tick — so the same message is asked for repeatedly. The
    /// in-flight gate must absorb that: `runScreening`'s own `safetyLabel == nil`
    /// check happens before an await and cannot.
    func testRepeatedViewportCallsDoNotDoubleScreen() async throws {
        let convo = conversation(["Let's meet tomorrow at 3pm."])

        for _ in 0..<20 { convo.screenFromViewport(convo.messages[0]) }
        _ = await waitForScreening(convo, count: 1)

        XCTAssertEqual(convo.messages.filter { $0.safetyLabel != nil }.count, 1)
        XCTAssertEqual(convo.screeningRuns, 1,
                       "classified \(convo.screeningRuns) times — the in-flight gate leaked")
    }

    /// The live and viewport paths overlap on every message that arrives while the
    /// chat is open: `classifySafety` runs on arrival, and the row's `onAppear`
    /// asks for the same message moments later. Both must not classify it.
    ///
    /// This is exactly what the seeded-burst path does — it screens the newest
    /// message eagerly (the inbox preview needs it) into a view that is about to
    /// materialize that same row.
    func testLiveAndViewportPathsDoNotDoubleScreen() async throws {
        let convo = conversation(["Let's meet tomorrow at 3pm."])
        let message = convo.messages[0]

        // Both paths fired for the same message, same runloop tick.
        Task { await convo.classifySafety(in: message) }
        convo.screenFromViewport(message)

        _ = await waitForScreening(convo, count: 1)
        try? await Task.sleep(nanoseconds: 300_000_000)

        XCTAssertEqual(convo.screeningRuns, 1,
                       "classified \(convo.screeningRuns) times — the live path is "
                     + "bypassing the in-flight gate")
    }

    /// The saving, stated as work done rather than as state reached: opening a
    /// long thread must not classify the whole thing.
    func testOpeningALongThreadDoesBoundedWork() async throws {
        let convo = conversation((0..<60).map { "Backlog message number \($0)." })

        convo.screenFromViewport(convo.messages[convo.messages.count - 1])
        _ = await waitForScreening(convo, count: Conversation.screenLookAhead + 1)
        try? await Task.sleep(nanoseconds: 300_000_000)

        XCTAssertLessThanOrEqual(convo.screeningRuns, Conversation.screenLookAhead + 1,
                                 "ran \(convo.screeningRuns) classifications for one viewport")
    }

    /// Screening is idempotent — an already-classified message is left alone, so
    /// scrolling back over history costs nothing.
    func testAlreadyScreenedMessagesAreNotReclassified() async throws {
        let convo = conversation(["Let's meet tomorrow at 3pm."])
        convo.screenFromViewport(convo.messages[0])
        _ = await waitForScreening(convo, count: 1)

        let verdict = convo.messages[0].safetyLabel
        convo.screenFromViewport(convo.messages[0])
        try? await Task.sleep(nanoseconds: 200_000_000)

        XCTAssertEqual(convo.messages[0].safetyLabel, verdict, "verdict changed on rescreen")
    }

    /// An empty message is skipped outright rather than queued — a reply
    /// placeholder is empty until the first token arrives, and every row that
    /// appears would otherwise enqueue work for text that isn't there yet.
    ///
    /// Whitespace-only text is NOT skipped here; it reaches `GuardEngine`, which
    /// trims and returns Clean before the tokenizer or the model run. That costs
    /// a queue hop and nothing else, and it is the behaviour the arrival path
    /// always had — pinned so the difference is deliberate rather than noticed
    /// later as a bug.
    func testEmptyMessagesAreNotScreened() async throws {
        let convo = conversation(["", "   "])
        convo.screenFromViewport(convo.messages[0])
        convo.screenFromViewport(convo.messages[1])
        try? await Task.sleep(nanoseconds: 300_000_000)

        XCTAssertNil(convo.messages[0].safetyLabel, "empty text should never be queued")
        XCTAssertEqual(convo.messages[1].safety, .clean,
                       "whitespace reaches the engine and comes back clean")
    }
}
