import XCTest
@testable import SLMChat

/// Measures how much of each auto-suggest prompt is a byte-identical prefix of the
/// previous one — i.e. how much of it `ChatLM`'s KV cache can skip.
///
/// This is the cheapest latency lever in the app and the only free one: a reused
/// token costs nothing, while an evaluated one costs ~1/55 s on an A12. A device
/// turn showed `in=257 tok (cached 32, eval 225)` — 88% of the prompt re-prefilled,
/// 4.05 s of a 6.01 s turn, against a 3 s budget. Whether that was a bad turn or
/// the normal pattern is exactly what this pins down.
///
/// Prefix length is measured in characters, not tokens. Tokenization is
/// deterministic and prefix-preserving, so the character ratio tracks the token
/// ratio closely enough to compare window settings against each other.
final class SuggestReuseTests: XCTestCase {

    /// A chat grows by ~2 messages per turn: the friend says something, you reply.
    /// Auto-suggest fires on the friend's message, so suggests happen every 2.
    private func conversation(messages n: Int) -> [ChatMessage] {
        (0..<n).map { i in
            ChatMessage(sender: i % 2 == 0 ? .friend : .you,
                        text: i % 2 == 0
                            ? "Message \(i) from the friend, roughly the length of a real chat line."
                            : "Reply \(i), also about as long as someone actually types.")
        }
    }

    private func sharedPrefix(_ a: String, _ b: String) -> Int {
        var n = 0
        var i = a.startIndex, j = b.startIndex
        while i < a.endIndex, j < b.endIndex, a[i] == b[j] {
            n += 1; i = a.index(after: i); j = b.index(after: j)
        }
        return n
    }

    /// Report the reuse profile over a conversation's lifetime. Not an assertion
    /// about a target — it prints the pattern so the window/stride pair can be
    /// chosen from data instead of intuition.
    func testReuseProfileAcrossATypicalConversation() {
        var previous: String?
        var totals: (reused: Int, total: Int) = (0, 0)
        var fullReprefills = 0
        var suggests = 0

        // Suggest fires on each friend message, i.e. at even counts.
        for count in stride(from: 4, through: 40, by: 2) {
            let history = conversation(messages: count)
            let prompt = Prompts.suggestReplies(history: history, language: .vietnamese)
            if let previous {
                suggests += 1
                let shared = sharedPrefix(previous, prompt)
                totals.reused += shared
                totals.total += prompt.count
                if shared < prompt.count / 2 { fullReprefills += 1 }
                print(String(format: "REUSE n=%2d prompt=%4d chars shared=%4d (%.0f%%)%@",
                             count, prompt.count, shared,
                             100.0 * Double(shared) / Double(prompt.count),
                             shared < prompt.count / 2 ? "  <-- re-prefill" : ""))
            }
            previous = prompt
        }

        let pct = 100.0 * Double(totals.reused) / Double(totals.total)
        print(String(format: "REUSE SUMMARY avg=%.0f%% reused, %d of %d suggests re-prefilled",
                     pct, fullReprefills, suggests))
    }

    /// The window must not move on every turn. That is the failure `windowStride`
    /// exists to prevent, and the one that makes prefill dominate every turn
    /// instead of every few — worth an assertion, not just a printout.
    func testWindowHoldsStillBetweenStrideSteps() {
        var starts: [Int] = []
        for count in 10...30 {
            starts.append(Prompts.windowStart(messageCount: count, window: Prompts.suggestWindow))
        }
        let moves = zip(starts, starts.dropFirst()).filter { $0 != $1 }.count
        // Over 20 message arrivals a stride-5 window should step ~4 times, not 20.
        XCTAssertLessThanOrEqual(moves, 6,
                                 "window is sliding nearly every message — stride isn't holding "
                                 + "(starts: \(starts))")
    }
}
