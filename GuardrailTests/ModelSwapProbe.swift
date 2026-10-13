import XCTest
@testable import SLMChat

/// LFM2.5-230M as a candidate replacement for Gemma-3 270M on low-tier phones.
/// Same harness shape as `PrefillThreadsProbe`, and like it, meaningful only
/// ON A DEVICE (the iPhone XS is the device the swap is for):
///
///   xcodebuild test -scheme SLMChat -destination 'platform=iOS,id=<udid>' \
///     -only-testing:GuardrailTests/ModelSwapProbe DEVELOPMENT_TEAM=<team>
///
/// then grep `SWAP`. Needs `Resources/lfm2.5-230m.gguf` (Q8_0) and optionally
/// `Resources/lfm2.5-230m-q4.gguf` (QAD Q4_0) from
/// huggingface.co/LiquidAI/LFM2.5-230M-GGUF; missing files are skipped.
///
/// Every model runs on its bare weights. LFM2.5 has no task pack yet, so this
/// measures speed and memory only — the outputs are printed as a template
/// sanity check, not as a quality comparison.
final class ModelSwapProbe: XCTestCase {

    private let candidates = ["lfm2.5-230m", "model"]

    /// The same ~500-token suggest-shaped prompt `PrefillThreadsProbe` uses, so
    /// the two probes' prefill numbers line up.
    private var englishPrompt: String {
        let lines = [
            "<Mai>: Hey! Are you still up for dinner on Friday? I found a new ramen place downtown.",
            "<Duy>: Yes definitely, I've been craving ramen all week. What time works for you?",
            "<Mai>: How about 7pm? I can book a table, they said weekends get pretty busy there.",
            "<Duy>: 7 is perfect. Should we invite Linh too? She mentioned wanting to catch up.",
        ]
        let body = (0..<4).flatMap { _ in lines }.joined(separator: "\n")
        return "Suggest a few short smart-reply options in English for <Duy>, who replies next in the conversation below.\n\nConversation:\n" + body
    }

    /// The same conversation in Vietnamese. LFM2.5 does not list Vietnamese,
    /// and a tokenizer that splits it finely
    /// makes every Vietnamese prompt proportionally slower to prefill.
    private let vietnameseSample = """
    <Mai>: Ê! Tối thứ Sáu cậu vẫn đi ăn tối chứ? Mình mới tìm được một quán ramen ở trung tâm.
    <Duy>: Đi chứ, cả tuần nay mình thèm ramen lắm. Mấy giờ thì tiện cho cậu?
    <Mai>: 7 giờ được không? Mình đặt bàn trước, họ bảo cuối tuần quán khá đông.
    <Duy>: 7 giờ là ổn. Có nên rủ Linh đi cùng không? Bạn ấy bảo muốn gặp mọi người.
    """

    /// The Vietnamese counterpart of `englishPrompt`: same conversation, same
    /// shape, so prefill SECONDS (not tok/s) compare directly across languages.
    ///
    /// At this full length LFM2.5 tokenizes it to ~1109 tokens (Gemma: 462), past
    /// the app's 1024-token slot, so it fails on LFM2.5 — kept to show that.
    /// `repeats: 2` is the version both models fit, for timing.
    private func vietnamesePrompt(repeats: Int = 4) -> String {
        let body = (0..<repeats).map { _ in vietnameseSample }.joined(separator: "\n")
        return "Gợi ý vài câu trả lời ngắn bằng tiếng Việt cho <Duy>, người trả lời tiếp theo trong đoạn hội thoại dưới đây.\n\nHội thoại:\n" + body
    }

    private struct Result {
        var promptTokens = 0
        var prefillSeconds = 0.0
        var decodeTokens = 0
        var decodeTokensPerSecond = 0.0
        var decodeCharsPerSecond = 0.0
        var outputChars = 0
        var output = ""
    }

    /// Tokens/sec is NOT comparable across tokenizers — LFM2.5 splits Vietnamese
    /// into ~2.4x the tokens Gemma does — so the figures that matter here are
    /// prefill seconds for the same text and output characters per second.
    private func measure(_ lm: ChatLM, _ prompt: String) -> Result {
        var r = Result()
        // Prefill: one warm-up to page in weights, then the median of three.
        var prefill: [Double] = []
        for i in 0..<4 {
            _ = lm.generate(system: nil, history: [], user: prompt, maxTokens: 1,
                            temperature: 0.1, cacheable: true, adapterPath: nil)
            guard let m = lm.lastTurnMetrics else { continue }
            r.promptTokens = Int(m.prefillTokenCount)
            if i > 0 { prefill.append(m.prefillSeconds) }
        }
        r.prefillSeconds = prefill.sorted()[prefill.count / 2]

        // Decode: a natural generation, EOS respected, so the output can be
        // eyeballed too. Median of three, ranked by chars/s.
        var runs: [Result] = []
        for _ in 0..<3 {
            let out = lm.generate(system: nil, history: [], user: prompt, maxTokens: 128,
                                  temperature: 0.1, cacheable: true, adapterPath: nil) ?? "<nil>"
            guard let m = lm.lastTurnMetrics, m.decodeSeconds > 0 else { continue }
            var d = r
            d.output = out
            d.outputChars = out.count
            d.decodeTokens = Int(m.decodeTokenCount)
            d.decodeTokensPerSecond = m.decodeTokensPerSecond
            d.decodeCharsPerSecond = Double(out.count) / m.decodeSeconds
            runs.append(d)
        }
        guard !runs.isEmpty else { return r }
        return runs.sorted { $0.decodeCharsPerSecond < $1.decodeCharsPerSecond }[runs.count / 2]
    }

    private func footprintMB() -> Double {
        Double(ProcessMetrics.memoryFootprint() ?? 0) / 1_048_576
    }

    func testGemmaVersusLFM() throws {
        let present = candidates.compactMap { name in
            ResourceLookup.path(name, ext: "gguf").map { (name, $0) }
        }
        guard present.count > 1 else { throw XCTSkip("Need model.gguf plus at least one LFM2.5 .gguf in the bundle.") }

        for (name, path) in present {
            let base = footprintMB()
            let lm = ChatLM()
            let t0 = Date()
            guard lm.load(modelPath: path, backend: SLMBackend.cpu, cacheDir: nil, loraPath: nil) else {
                XCTFail("\(name) failed to load: \(lm.lastError)"); continue
            }
            let loadS = Date().timeIntervalSince(t0)
            let loadedMB = footprintMB() - base
            lm.prefixReuseEnabled = false

            for (lang, prompt) in [("en", englishPrompt), ("vi", vietnamesePrompt()), ("vi-half", vietnamesePrompt(repeats: 2))] {
                let r = measure(lm, prompt)
                guard r.promptTokens > 0 else {
                    print("SWAP \(name) \(lang) | FAILED — prompt is \(lm.tokenCount(of: prompt)) tokens, error: \(lm.lastError)")
                    continue
                }
                print(String(format: "SWAP %@ %@ | prompt %ld tok, %ld chars | prefill %.3fs (%.0f tok/s) | decode %.0f chars/s (%.1f tok/s, %ld tok, %.1f chars/tok)",
                             name, lang, r.promptTokens, prompt.count, r.prefillSeconds,
                             Double(r.promptTokens) / r.prefillSeconds,
                             r.decodeCharsPerSecond, r.decodeTokensPerSecond, r.decodeTokens,
                             r.decodeTokens > 0 ? Double(r.outputChars) / Double(r.decodeTokens) : 0))
                print("SWAP \(name) \(lang) output: \(r.output.prefix(300).replacingOccurrences(of: "\n", with: " ⏎ "))")
                XCTAssertFalse(r.output.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                               "\(name) produced no text — check the chat template path")
            }
            print(String(format: "SWAP %@ | load %.2fs, +%.0f MB loaded, +%.0f MB after gen",
                         name, loadS, loadedMB, footprintMB() - base))
        }
    }
}
