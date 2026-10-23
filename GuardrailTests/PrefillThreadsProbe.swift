import XCTest
@testable import SLMChat

/// Prefill throughput vs `n_threads_batch`, for the open question in
/// PERFORMANCE.md §5: ChatLM gives the prompt pass only the P-cores, a choice
/// justified for decode (memory-bound) and never measured for prefill
/// (compute-bound). Run it ON A DEVICE — the simulator's core layout says
/// nothing about an A12:
///
///   xcodebuild test -scheme SLMChat -destination 'platform=iOS,id=<udid>' \
///     -only-testing:GuardrailTests/PrefillThreadsProbe DEVELOPMENT_TEAM=<team>
///
/// then grep `PTHREADS`. Reuse is off so every run pays the full prompt.
final class PrefillThreadsProbe: XCTestCase {

    /// ~500 tokens of chat-shaped text in the auto-suggest format, close to the
    /// 450–600-token windows measured on the XS.
    private var prompt: String {
        let lines = [
            "<Mai>: Hey! Are you still up for dinner on Friday? I found a new ramen place downtown.",
            "<Duy>: Yes definitely, I've been craving ramen all week. What time works for you?",
            "<Mai>: How about 7pm? I can book a table, they said weekends get pretty busy there.",
            "<Duy>: 7 is perfect. Should we invite Linh too? She mentioned wanting to catch up.",
        ]
        let body = (0..<4).flatMap { _ in lines }.joined(separator: "\n")
        return "Suggest a few short smart-reply options in English for <Duy>, who replies next in the conversation below.\n\nConversation:\n" + body
    }

    func testPrefillVersusBatchThreads() throws {
        let hw = ProcessInfo.processInfo.activeProcessorCount
        let names = ModelCatalog.taskModels.filter(\.isAvailable).map(\.resourceName)
            .filter { ["qwen2.5-0.5b-v6.1", "qwen3-0.6b-v6", "model"].contains($0) }
        guard !names.isEmpty else { throw XCTSkip("No .gguf in the bundle.") }
        for name in names {
            guard let path = ResourceLookup.path(name, ext: "gguf") else { continue }
            for threads in Array(Set([2, 4, hw])).sorted() where threads <= hw {
                let lm = ChatLM()
                lm.batchThreads = Int32(threads)
                guard lm.load(modelPath: path, backend: SLMBackend.cpu, cacheDir: nil, loraPath: nil) else {
                    XCTFail("\(name) failed to load: \(lm.lastError)"); continue
                }
                // One warm-up (page in weights), then three measured passes.
                var rates: [Double] = []
                var tokens = 0
                for i in 0..<4 {
                    _ = lm.generate(system: nil, history: [], user: prompt, maxTokens: 1,
                                    temperature: 0.1, adapterPath: nil)
                    guard let m = lm.lastTurnMetrics else { continue }
                    tokens = Int(m.promptTokenCount)
                    if i > 0 { rates.append(m.prefillTokensPerSecond) }
                }
                let med = rates.sorted()[rates.count / 2]
                print(String(format: "PTHREADS %@ threads_batch=%d (of %d cores) | %d tok | prefill %.1f tok/s (runs %@) | %.2fs per 500 tok",
                             name, threads, hw, tokens, med,
                             rates.map { String(format: "%.1f", $0) }.joined(separator: ", "),
                             500.0 / med))
            }
        }
    }
}
