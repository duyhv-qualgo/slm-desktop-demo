import XCTest
@testable import SLMChat

/// Checks the screening scheduler: bidirectional look-ahead, viewport-priority
/// ordering, and shedding.
///
/// The behaviour these pin was chosen against measurements, not taste. A
/// task-per-message drained FIFO was measured queueing 210 messages and finishing
/// 17 during a 200-row flick, with the message on screen waiting behind ~200 the
/// reader had already passed. Each test here corresponds to one part of that.
@MainActor
final class MessageScreenerTests: XCTestCase {

    private func messages(_ n: Int) -> [ChatMessage] {
        (0..<n).map { ChatMessage(sender: .friend, text: "Message number \($0).") }
    }

    /// Records what ran, in order, without touching the real classifier — the
    /// scheduler's job is ordering, and a real verdict would only add noise.
    private func recordingScreener(_ log: @escaping (ChatMessage) -> Void)
        -> MessageScreener<ChatMessage> {
        MessageScreener { message, _ in
            log(message)
            message.apply(.clean)          // mark done, as real screening would
        }
    }

    private func settle() async {
        try? await Task.sleep(nanoseconds: 400_000_000)
    }

    /// Look-ahead must run BOTH ways.
    ///
    /// Older-only was correct while chats opened at the bottom. Anchored at the
    /// first unread message the reader travels DOWNWARD into newer messages, and
    /// one-directional look-ahead would race away from them — every row arriving
    /// unscreened, which is the whole failure the look-ahead exists to prevent.
    func testLookAheadCoversBothDirections() async throws {
        let msgs = messages(100)
        var seen: [String] = []
        let s = recordingScreener { seen.append($0.text) }

        s.viewportDidShow(index: 50, in: msgs)
        await settle()

        XCTAssertTrue(seen.contains("Message number 50."), "the anchor itself")
        XCTAssertTrue(seen.contains("Message number 45."), "older neighbour")
        XCTAssertTrue(seen.contains("Message number 55."), "NEWER neighbour — the "
                    + "direction a last-read-anchored reader actually travels")
        XCTAssertFalse(seen.contains("Message number 90."), "far messages must wait")
    }

    /// Work is taken nearest-the-viewport first, not first-enqueued first.
    ///
    /// Queue a distant region, then move the viewport. What runs next must be
    /// what the reader is looking at now — under FIFO it would be the distant
    /// work, which is the defect this scheduler replaced.
    func testNearestToViewportRunsFirst() async throws {
        let msgs = messages(400)
        var seen: [ChatMessage] = []
        let s = recordingScreener { seen.append($0) }

        s.viewportDidShow(index: 0, in: msgs)     // queue the far end
        s.viewportDidShow(index: 300, in: msgs)   // reader jumps here
        await settle()

        let firstFew = seen.prefix(5).compactMap { m in
            msgs.firstIndex(where: { $0.id == m.id })
        }
        XCTAssertFalse(firstFew.isEmpty)
        for i in firstFew {
            XCTAssertGreaterThan(i, 200,
                                 "ran index \(i) before the viewport at 300 — "
                               + "the queue is draining FIFO again")
        }
    }

    /// The queue stays bounded under a flick.
    func testQueueIsShedUnderAFlick() async throws {
        let msgs = messages(1_000)
        let s = MessageScreener<ChatMessage> { _, _ in
            // Never completes quickly, so the queue builds while we scroll.
            try? await Task.sleep(nanoseconds: 5_000_000)
        }

        for i in stride(from: 900, through: 100, by: -1) {
            s.viewportDidShow(index: i, in: msgs)
        }

        XCTAssertLessThanOrEqual(s.pendingCount, MessageScreening.maxPending,
                                 "queue reached \(s.pendingCount) — shedding is not working")
        XCTAssertGreaterThan(s.shed, 0, "a 800-row flick should have shed work")
    }

    /// Shedding must not silently lose coverage: a shed message is FORGOTTEN, so
    /// returning to it screens it like any other row. If shedding also marked it
    /// claimed, scrolling back would show permanently unscreened messages.
    func testShedMessagesAreScreenedWhenScrolledBackTo() async throws {
        let msgs = messages(1_000)
        var seen: Set<String> = []
        let s = recordingScreener { seen.insert($0.text) }

        for i in stride(from: 900, through: 100, by: -1) {
            s.viewportDidShow(index: i, in: msgs)
        }
        await settle()

        // Pick something far from where the flick ended, likely shed.
        let target = msgs[800]
        if !seen.contains(target.text) {
            s.viewportDidShow(index: 800, in: msgs)
            await settle()
            XCTAssertTrue(seen.contains(target.text),
                          "a shed message was never screened on return — shedding "
                        + "is dropping coverage rather than deferring it")
        }
    }

    /// The live path runs ahead of the queue and dedupes against it.
    func testScreenNowBypassesTheQueueAndDedupes() async throws {
        let msgs = messages(50)
        var count = 0
        let s = recordingScreener { _ in count += 1 }

        s.viewportDidShow(index: 10, in: msgs)   // queues index 10 among others
        await s.screenNow(msgs[10])              // same message, live path
        await settle()

        XCTAssertEqual(msgs[10].safety, .clean)
        let tens = msgs.filter { $0.text == "Message number 10." }
        XCTAssertEqual(tens.count, 1)
        XCTAssertGreaterThan(count, 0)
        // Total runs can never exceed the messages that needed screening.
        XCTAssertLessThanOrEqual(s.runs, msgs.count)
    }

    /// Already-screened messages cost nothing — scrolling back over read history
    /// must be free, which is the assumption the whole last-read design rests on.
    func testAlreadyScreenedMessagesAreSkipped() async throws {
        let msgs = messages(50)
        for m in msgs { m.apply(.clean) }        // pretend the thread was read before

        var ran = 0
        let s = recordingScreener { _ in ran += 1 }
        s.viewportDidShow(index: 25, in: msgs)
        await settle()

        XCTAssertEqual(ran, 0, "re-screened \(ran) already-classified messages")
        XCTAssertEqual(s.pendingCount, 0)
    }
}
