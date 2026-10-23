import XCTest
@testable import SLMChat

/// Q4_0 vs the shipping Q8_0, before the team invests in QAT: what does the
/// format alone buy in speed and memory? Same harness as `ModelSwapProbe` and
/// `PrefillThreadsProbe`, so the numbers line up with PERFORMANCE.md.
///
///   xcodebuild test -scheme SLMChat -destination '<sim or device>' \
///     -only-testing:GuardrailTests/QuantFormatProbe DEVELOPMENT_TEAM=<team>
///
/// then grep `QUANT`. Needs `Resources/<model>-q4_0.gguf` next to each Q8_0
/// file, made with
/// `llama-quantize --allow-requantize <model>.gguf <model>-q4_0.gguf Q4_0`.
/// Requantized from Q8_0, so the Q4 outputs are NOT a quality preview of QAT —
/// they are printed only as a template sanity check. Missing files are skipped.
///
/// The iPhone XS runs only qwen2.5 and qwen3; qwen3.5-0.8b is simulator-only.
final class QuantFormatProbe: XCTestCase {

    /// (Q8_0 resource, matching LoRA pack prefix)
    private var models: [(name: String, lora: String)] {
        let all = [("qwen2.5-0.5b-v6.1", "lora-qwen2.5-0.5b-v6.1"),
                   ("qwen3-0.6b-v6", "lora-qwen3-0.6b-v6"),
                   ("qwen3.5-0.8b", "lora-qwen3.5-0.8b")]
        // `TEST_RUNNER_QUANT_MODELS=qwen3-0.6b-v6` (comma-separated) picks a
        // subset — handy when a long device run dies halfway. Without it,
        // qwen3.5 runs only in the simulator (the XS doesn't need it).
        if let only = ProcessInfo.processInfo.environment["QUANT_MODELS"] {
            let keep = Set(only.split(separator: ",").map(String.init))
            return all.filter { keep.contains($0.0) }
        }
        #if targetEnvironment(simulator)
        return all
        #else
        return all.filter { $0.0 != "qwen3.5-0.8b" }
        #endif
    }

    /// `TEST_RUNNER_QUANT_BACKEND=gpu` runs every model on Metal instead of the
    /// CPU (iPhone 14 Pro and later; the A12 must stay on CPU).
    private var backend: String {
        ProcessInfo.processInfo.environment["QUANT_BACKEND"] == "gpu" ? SLMBackend.gpu : SLMBackend.cpu
    }

    /// `Q4_0+q8head`: Q4_0 everywhere except the tied embedding/output head,
    /// which stays Q8_0 (`--token-embedding-type q8_0 --output-tensor-type q8_0`).
    /// The default recipe makes it Q6_K, a K-quant the CPU decodes slowly.
    ///
    /// `TEST_RUNNER_QUANT_REVERSE=1` runs them last-to-first. On the XS a long
    /// run heats the phone (qwen3 Q8_0 prefill: 52 tok/s cool vs 28 after ~20
    /// min), so whatever runs first gets the advantage; compare both orders.
    private var formats: [(String, String)] {
        var all = [("Q8_0", ""), ("Q4_0", "-q4_0"), ("Q4_0+q8head", "-q4_0-q8head"),
                   // English pack merged into the Q4_0 weights (llama-export-lora): no LoRA at runtime
                   ("Q4_0-merged", "-q4_0-merged-en")]
        // `TEST_RUNNER_QUANT_FORMATS=Q8_0,Q4_0` picks a subset.
        if let only = ProcessInfo.processInfo.environment["QUANT_FORMATS"] {
            let keep = Set(only.split(separator: ",").map(String.init))
            all = all.filter { keep.contains($0.0) }
        }
        return ProcessInfo.processInfo.environment["QUANT_REVERSE"] == "1" ? all.reversed() : all
    }

    /// The ~500-token suggest-shaped prompt `PrefillThreadsProbe` uses.
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

    /// The same conversation in Vietnamese (the `ModelSwapProbe` text).
    private var vietnamesePrompt: String {
        let sample = """
        <Mai>: Ê! Tối thứ Sáu cậu vẫn đi ăn tối chứ? Mình mới tìm được một quán ramen ở trung tâm.
        <Duy>: Đi chứ, cả tuần nay mình thèm ramen lắm. Mấy giờ thì tiện cho cậu?
        <Mai>: 7 giờ được không? Mình đặt bàn trước, họ bảo cuối tuần quán khá đông.
        <Duy>: 7 giờ là ổn. Có nên rủ Linh đi cùng không? Bạn ấy bảo muốn gặp mọi người.
        """
        let body = (0..<4).map { _ in sample }.joined(separator: "\n")
        return "Gợi ý vài câu trả lời ngắn bằng tiếng Việt cho <Duy>, người trả lời tiếp theo trong đoạn hội thoại dưới đây.\n\nHội thoại:\n" + body
    }

    private struct Result {
        var promptTokens = 0
        var prefillSeconds = 0.0
        var decodeTokens = 0
        var decodeTokensPerSecond = 0.0
        var totalSeconds = 0.0
        var output = ""
    }

    /// Prefill: a warm-up, then the median of three 1-token passes. Decode: the
    /// median of three natural 128-token-capped generations. Both formats share a
    /// tokenizer, so tok/s compares directly within a model.
    private func measure(_ lm: ChatLM, _ prompt: String) -> Result {
        var r = Result()
        var prefill: [Double] = []
        for i in 0..<4 {
            _ = lm.generate(system: nil, history: [], user: prompt, maxTokens: 1,
                            temperature: 0.1, adapterPath: nil)
            guard let m = lm.lastTurnMetrics else { continue }
            r.promptTokens = Int(m.promptTokenCount)
            if i > 0 { prefill.append(m.prefillSeconds) }
        }
        guard !prefill.isEmpty else { return r }
        r.prefillSeconds = prefill.sorted()[prefill.count / 2]

        var runs: [Result] = []
        for _ in 0..<3 {
            let out = lm.generate(system: nil, history: [], user: prompt, maxTokens: 128,
                                  temperature: 0.1, adapterPath: nil) ?? "<nil>"
            guard let m = lm.lastTurnMetrics, m.decodeSeconds > 0 else { continue }
            var d = r
            d.output = out
            d.decodeTokens = Int(m.decodeTokenCount)
            d.decodeTokensPerSecond = m.decodeTokensPerSecond
            d.totalSeconds = m.totalSeconds
            runs.append(d)
        }
        guard !runs.isEmpty else { return r }
        return runs.sorted { $0.decodeTokensPerSecond < $1.decodeTokensPerSecond }[runs.count / 2]
    }

    private func footprintMB() -> Double {
        Double(ProcessMetrics.memoryFootprint() ?? 0) / 1_048_576
    }

    private func fileMB(_ path: String) -> Double {
        let size = (try? FileManager.default.attributesOfItem(atPath: path)[.size] as? NSNumber)?.doubleValue ?? 0
        return size / 1_048_576
    }

    func testQ8VersusQ4() throws {
        var ran = 0
        for model in models {
            for (format, suffix) in formats {
                guard let path = ResourceLookup.path(model.name + suffix, ext: "gguf") else {
                    print("QUANT \(model.name) \(format) | SKIPPED — not in bundle"); continue
                }
                // Bare weights on both prompts, then the shipping shape: the
                // English pack (rank 64 since 2026-10-05) applied at load.
                let loraPath = ResourceLookup.path(model.lora + "-english", ext: "gguf")
                // Optional `<pack>-r32.gguf`: the original rank-32 pack, to
                // re-measure the Metal small-K penalty (models-lab/lora-rank32/).
                let lora32Path = ResourceLookup.path(model.lora + "-english-r32", ext: "gguf")
                let merged = format.hasSuffix("-merged")
                let configs: [(String, String?, [(String, String)])] = merged ? [
                    ("merged-en", nil, [("en", englishPrompt)]),
                ] : [
                    ("bare", nil, [("en", englishPrompt), ("vi", vietnamesePrompt)]),
                    ("lora-en", loraPath, loraPath == nil ? [] : [("en", englishPrompt)]),
                    ("lora-en-r32", lora32Path, lora32Path == nil ? [] : [("en", englishPrompt)]),
                ]
                for (label, lora, prompts) in configs where !prompts.isEmpty {
                    let base = footprintMB()
                    let lm = ChatLM()
                    let t0 = Date()
                    guard lm.load(modelPath: path, backend: backend, cacheDir: nil, loraPath: lora) else {
                        XCTFail("\(model.name) \(format) \(label) failed to load: \(lm.lastError)"); continue
                    }
                    let loadS = Date().timeIntervalSince(t0)
                    if backend == SLMBackend.gpu && lm.activeBackend != SLMBackend.gpu {
                        XCTFail("\(model.name) \(format) asked for gpu, got \(lm.activeBackend)"); continue
                    }
                    let loadedMB = footprintMB() - base
                    ran += 1

                    for (lang, prompt) in prompts {
                        let r = measure(lm, prompt)
                        guard r.promptTokens > 0 else {
                            print("QUANT \(model.name) \(format) \(label) \(lang) | FAILED — \(lm.tokenCount(of: prompt)) tok, error: \(lm.lastError)")
                            continue
                        }
                        print(String(format: "QUANT %@ %@ %@ %@ | prompt %ld tok | prefill %.3fs (%.1f tok/s) | decode %.1f tok/s (%ld tok) | gen total %.2fs",
                                     model.name, format, label, lang, r.promptTokens, r.prefillSeconds,
                                     Double(r.promptTokens) / r.prefillSeconds,
                                     r.decodeTokensPerSecond, r.decodeTokens, r.totalSeconds))
                        print("QUANT \(model.name) \(format) \(label) \(lang) output: \(r.output.prefix(200).replacingOccurrences(of: "\n", with: " ⏎ "))")
                        XCTAssertFalse(r.output.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                                       "\(model.name) \(format) produced no text")
                    }
                    print(String(format: "QUANT %@ %@ %@ [%@] | file %.0f MB | load %.2fs | +%.0f MB loaded | +%.0f MB after gen",
                                 model.name, format, label, lm.activeBackend, fileMB(path), loadS, loadedMB, footprintMB() - base))
                }
            }
        }
        if ran == 0 { throw XCTSkip("No Qwen .gguf in the bundle.") }
    }
}
