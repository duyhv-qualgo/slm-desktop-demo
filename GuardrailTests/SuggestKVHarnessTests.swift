import XCTest
@testable import SLMChat

/// End-to-end KV-reuse harness for auto-suggest: plays a scripted chat through a
/// real `ChatLM` with prefix reuse ON and records what each suggest actually
/// evaluated.
///
/// Why this exists alongside the others: `BenchmarkView` clears the cache every
/// run by design, and `SuggestReuseTests` only diffs prompt TEXT. Neither sees
/// slot choice, LRU eviction, the chat template's tail, tokenizer boundaries or
/// the scratch-slot traffic other features add — all of which decide the real
/// `eval` count. This does.
///
/// It mirrors the app's call pattern (see `Conversation.send` / `.suggest`):
///   - on each message YOU send: event detection (non-cacheable, scratch slot),
///     and — with pre-warm on — `prefill` of the next suggest's known prefix;
///   - on each friend message that ends a burst: the suggest itself.
///
/// `eval` (prompt tokens actually run through the model) is the metric. It is a
/// token count, so it does not depend on the device or backend: simulator
/// numbers transfer, seconds do not.
///
/// Run: xcodebuild test -scheme SLMChat -destination 'platform=iOS Simulator,name=iPhone 17 Pro' \
///        -only-testing:GuardrailTests/SuggestKVHarnessTests
/// then grep the console for `KVH`.
final class SuggestKVHarnessTests: XCTestCase {

    // MARK: - Script

    /// ~24 tokens per line, like a real chat. Includes a friend burst (two in a
    /// row → one suggest), a user burst (two sends → two pre-warms, the second
    /// supersedes), and a switch to Vietnamese near the end, which changes the
    /// prompt's first line and so is a full miss by construction. That switch is
    /// only exercised when `LangDetect`'s assets are bundled; without them every
    /// language resolves to native and the `lang` row stays empty.
    private static let script: [(Sender, String)] = [
        (.friend, "Hey! Are you still up for dinner on Friday? I found a new ramen place downtown."),
        (.you,    "Yes definitely, I've been craving ramen all week. What time works for you?"),
        (.friend, "How about 7pm? I can book a table, they said weekends get pretty busy."),
        (.you,    "7 is perfect. Should we invite Linh too? She mentioned wanting to catch up."),
        (.friend, "Good idea, I'll text her. Do you know if she's back from Da Nang yet?"),
        (.you,    "I think she got back on Tuesday, she posted photos from the beach yesterday."),
        (.friend, "Nice, then I'll book for three. Any dietary stuff I should mention to them?"),
        (.you,    "Not for me, but I think Linh stopped eating pork last month, worth checking."),
        (.friend, "Okay I'll ask her directly so we don't guess wrong. Thanks for the heads up!"),
        (.you,    "No problem. By the way, did you finish that report for your manager?"),
        (.friend, "Barely, I submitted it at midnight. She wants a follow-up meeting on Monday."),
        (.friend, "Honestly I'm a bit nervous, the numbers from last quarter didn't look great."),
        (.you,    "You'll be fine, you always prepare well. Want to go over the slides together?"),
        (.friend, "That would help a lot. Maybe Sunday afternoon at the coffee shop near me?"),
        (.you,    "Sunday works. Around 3pm? I have a family lunch before that, should be done by then."),
        (.friend, "3pm is great. I'll bring my laptop and the printed charts, just in case."),
        (.you,    "Perfect. Also, did you hear back from the apartment you applied for?"),
        (.you,    "The one with the big balcony you showed me last week, I mean."),
        (.friend, "Not yet! The agent said they'd decide by Thursday. Fingers crossed honestly."),
        (.you,    "I really hope you get it, that place was so much nicer than your current one."),
        (.friend, "Right? The commute would be half as long too. I'd finally have time for the gym."),
        (.you,    "Ha, you say that every time. But seriously, it would be good for you."),
        (.friend, "I know I know. Speaking of which, do you still go to that yoga class on Wednesdays?"),
        (.you,    "Yes, every week. The teacher is great, you should come try a session with me."),
        (.friend, "Maybe next week? This week is chaos with the report and the apartment stuff."),
        (.you,    "Sure, no pressure. I'll send you the schedule so you can pick a day."),
        (.friend, "Thanks! Oh, Linh just replied, she's in for Friday and she's fine with ramen."),
        (.you,    "Great, did she say anything about the pork thing? Just want to be sure."),
        (.friend, "She said she's okay with it now, she only stopped for a few weeks. All good."),
        (.you,    "Perfect, then we're set. Should we meet there or go together from the station?"),
        (.friend, "Let's meet at the station at 6:40, it's a short walk and we can chat on the way."),
        (.you,    "Sounds good. I'll be the one in the green jacket, as usual haha."),
        (.friend, "Classic. Hey, random question, do you remember the name of that bakery in District 3?"),
        (.you,    "The one with the croissants? I think it was called Maison something, let me check."),
        (.friend, "Yes that one! My mom's birthday is next week and she loves their almond cake."),
        (.you,    "Found it, it's Maison Marou. They take pre-orders on their website I think."),
        (.friend, "Tuyệt quá, cảm ơn bạn nhiều nha. Mình sẽ đặt bánh ngay tối nay luôn."),
        (.you,    "Không có gì, nhớ đặt sớm vì cuối tuần họ hay hết bánh lắm đó."),
        (.friend, "Ừ mình biết rồi, lần trước mình đến trễ là hết sạch luôn, tiếc ghê."),
        (.you,    "Vậy tối nay đặt luôn cho chắc nha, có gì nhắn mình."),
        (.friend, "Oke bạn, mà thứ Sáu nhớ đúng giờ nha, đừng để tụi mình chờ như lần trước."),
    ]

    // MARK: - Engine

    /// Every bundled task model that is present. The .gguf files are gitignored,
    /// so a checkout without them skips instead of failing.
    private func availableModels() -> [String] {
        ModelCatalog.taskModels.filter(\.isAvailable).map(\.resourceName)
    }

    /// A fresh engine per configuration, so one config's slots can't flatter the
    /// next. Loaded bare (no LoRA): the adapter changes which K/V values are
    /// cached, not how many tokens are.
    private func engine(_ name: String) throws -> ChatLM {
        guard let path = ResourceLookup.path(name, ext: "gguf") else {
            throw XCTSkip("\(name).gguf not in bundle")
        }
        let lm = ChatLM()
        guard lm.load(modelPath: path, backend: SLMBackend.cpu, cacheDir: nil, loraPath: nil) else {
            throw XCTSkip("\(name) failed to load: \(lm.lastError)")
        }
        lm.prefixReuseEnabled = true
        return lm
    }

    // MARK: - One run

    enum Kind: String { case first, steady, reanchor, lang }

    struct Turn {
        let index: Int
        let kind: Kind
        let promptTokens: Int
        let cached: Int
        let eval: Int
        /// nil = no pre-warm was attempted for this suggest.
        let prewarmTokens: Int?
        var prewarmHit: Bool? { prewarmTokens.map { cached >= $0 } }
    }

    struct Run {
        let model: String
        let stride: Int
        let prewarm: Bool
        var turns: [Turn] = []
        var prewarmEval: [Int] = []

        func evals(_ kind: Kind) -> [Int] { turns.filter { $0.kind == kind }.map(\.eval) }
    }

    private func play(model: String, stride: Int, prewarm: Bool) throws -> Run {
        let lm = try engine(model)
        var run = Run(model: model, stride: stride, prewarm: prewarm)
        var messages: [ChatMessage] = []
        var lastStart: Int?
        var lastLang: TranslateLanguage?
        var pendingPrewarm: Int?

        let script = Self.script
        for (i, (sender, text)) in script.enumerated() {
            let msg = ChatMessage(sender: sender, text: text)
            messages.append(msg)
            let nextIsFriend = i + 1 < script.count && script[i + 1].0 == .friend

            if sender == .you {
                // Event detection runs on every send in the app — non-cacheable,
                // so it lives in the scratch slot and must not disturb suggest's.
                _ = lm.generate(system: nil, history: [],
                                user: "Extract any calendar event from this message, or reply NONE:\n\(text)",
                                maxTokens: 4, temperature: 0.1, cacheable: false, adapterPath: nil)
                if prewarm && nextIsFriend {
                    let lang = Prompts.conversationLanguage(messages, native: .english)
                    let prefix = Prompts.suggestPrewarmPrefix(history: messages, language: lang,
                                                              stride: stride)
                    let n = lm.prefill(system: nil, history: [], user: prefix, adapterPath: nil)
                    pendingPrewarm = n > 0 ? Int(n) : nil
                    if let m = lm.lastTurnMetrics, n > 0 { run.prewarmEval.append(Int(m.prefillTokenCount)) }
                }
                continue
            }

            // Friend message: suggest fires once a burst has settled.
            guard !nextIsFriend else { continue }
            let lang = Prompts.conversationLanguage(messages, native: .english)
            let prompt = Prompts.suggestReplies(history: messages, language: lang, stride: stride)
            _ = lm.generate(system: nil, history: [], user: prompt,
                            maxTokens: 8, temperature: 0.8, cacheable: true, adapterPath: nil)
            let m = try XCTUnwrap(lm.lastTurnMetrics, "no metrics: \(lm.lastError)")

            let start = Prompts.windowStart(messageCount: messages.count,
                                            window: Prompts.suggestWindow, stride: stride)
            let kind: Kind = lastStart == nil ? .first
                : lang != lastLang ? .lang
                : start != lastStart ? .reanchor
                : .steady
            run.turns.append(Turn(index: i, kind: kind,
                                  promptTokens: Int(m.promptTokenCount),
                                  cached: Int(m.cachedPrefixTokenCount),
                                  eval: Int(m.prefillTokenCount),
                                  prewarmTokens: pendingPrewarm))
            lastStart = start
            lastLang = lang
            pendingPrewarm = nil
        }
        return run
    }

    // MARK: - Reporting

    private func stat(_ xs: [Int]) -> String {
        guard !xs.isEmpty else { return "      -" }
        let s = xs.sorted()
        let median = s[s.count / 2]
        let p95 = s[min(s.count - 1, Int((Double(s.count) * 0.95).rounded(.up)) - 1)]
        return String(format: "med %3d p95 %3d max %3d (n=%d)", median, p95, s.last!, s.count)
    }

    private func report(_ r: Run, perTurn: Bool) {
        if perTurn {
            for t in r.turns {
                let pw = t.prewarmHit.map { $0 ? "hit(\(t.prewarmTokens!))" : "MISS(\(t.prewarmTokens!))" } ?? "-"
                print(String(format: "KVH   %@ s=%d pw=%@ msg#%2d %-8@ in=%3d cached=%3d eval=%3d prewarm=%@",
                             r.model, r.stride, r.prewarm ? "on" : "off", t.index,
                             t.kind.rawValue, t.promptTokens, t.cached, t.eval, pw))
            }
        }
        let attempted = r.turns.compactMap(\.prewarmHit)
        let hits = attempted.filter { $0 }.count
        let reanchorServed = r.turns.filter { $0.kind == .reanchor && $0.prewarmHit == true }.count
        let reanchors = r.turns.filter { $0.kind == .reanchor }.count
        print(String(format: "KVH SUMMARY %@ stride=%d prewarm=%@ | steady %@ | reanchor %@ | "
                     + "reanchors %d/%d suggests (served from prewarm %d) | prewarm hit %d/%d | "
                     + "total suggest eval %d | background prewarm eval %d",
                     r.model, r.stride, r.prewarm ? "on " : "off",
                     stat(r.evals(.steady)), stat(r.evals(.reanchor)),
                     reanchors, r.turns.count, reanchorServed, hits, attempted.count,
                     r.turns.map(\.eval).reduce(0, +), r.prewarmEval.reduce(0, +)))
    }

    // MARK: - Tests

    /// The sweep behind `Prompts.suggestStride`: every stride 2…6, pre-warm off
    /// and on, every bundled model. Prints only; the decision is made from the
    /// summary lines.
    func testStrideAndPrewarmSweep() throws {
        let models = availableModels()
        guard !models.isEmpty else { throw XCTSkip("No .gguf in the bundle.") }
        for model in models {
            for prewarm in [false, true] {
                for stride in 2...6 {
                    report(try play(model: model, stride: stride, prewarm: prewarm), perTurn: false)
                }
            }
        }
    }

    /// The shipping configuration, per turn, with the success criterion as an
    /// assertion: at most 60 evaluated tokens on every steady (non-re-anchor,
    /// same-language) suggest.
    ///
    /// Qwen3.5 is included and passes, for a narrower reason than the others:
    /// its recurrent layers make `seq_rm` refuse any partial removal, so every
    /// PRE-WARM clears its slot and pays a full prefill in the background (see
    /// its much larger background eval). The suggest then extends that slot at
    /// exactly its end — removing nothing, the one case recurrent memory
    /// accepts. A pre-warm miss on Qwen3.5 therefore costs a full prefill.
    func testShippingConfigMeetsEvalBudget() throws {
        let models = availableModels()
        guard !models.isEmpty else { throw XCTSkip("No .gguf in the bundle.") }
        for model in models {
            let run = try play(model: model, stride: Prompts.suggestStride, prewarm: true)
            report(run, perTurn: true)
            let steady = run.evals(.steady)
            XCTAssertFalse(steady.isEmpty, "\(model): script produced no steady turns")
            XCTAssertLessThanOrEqual(steady.max() ?? 0, 60,
                                     "\(model): a steady suggest evaluated more than 60 tokens")
        }
    }
}
