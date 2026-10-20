import XCTest
@testable import SLMChat

/// What the running app's footprint is made of: each component loaded in the
/// order the app loads it, with `phys_footprint` sampled after each step (the
/// number Activity Monitor shows). Prints a table; asserts nothing beyond the
/// components loading, since footprint is noisy run to run.
///
/// Opt-in — it loads every model and takes ~30 s:
///   TEST_RUNNER_FOOTPRINT_BREAKDOWN=1 xcodebuild test … -only-testing:GuardrailTests/AppFootprintBreakdownTests
@MainActor
final class AppFootprintBreakdownTests: XCTestCase {

    private var rows: [(String, Double, Double)] = []
    private var last = 0.0

    private func mb() -> Double { Double(ProcessMetrics.memoryFootprint() ?? 0) / 1_048_576 }

    private func step(_ label: String, _ body: () async throws -> Void) async rethrows {
        try await body()
        try? await Task.sleep(nanoseconds: 500_000_000)   // let the allocator settle
        let now = mb()
        rows.append((label, now - last, now))
        last = now
    }

    func testBreakdown() async throws {
        guard ProcessInfo.processInfo.environment["FOOTPRINT_BREAKDOWN"] != nil else {
            throw XCTSkip("set TEST_RUNNER_FOOTPRINT_BREAKDOWN=1 to run")
        }
        try? await Task.sleep(nanoseconds: 500_000_000)
        last = mb()
        rows.append(("baseline (app process + test host, no models)", 0, last))

        let model = DesktopModels.defaultTaskModel
        let slm = SLMEngine(namespace: "footprint")
        await step("chat model load: \(model.displayName), GPU") {
            let r = await slm.warmUp(modelName: model.resourceName, backend: SLMBackend.gpu)
            XCTAssertTrue(r.ok, r.error ?? "")
        }
        await step("chat model first generation (KV + compute buffers)") {
            _ = await slm.complete(system: "You are helpful.", history: [],
                                   user: "Say hello in one short sentence.",
                                   maxTokens: 24, temperature: 0)
        }
        if let pack = LoRACatalog.installedPath(baseModelID: model.id, language: .vietnamese) {
            await step("chat model + Vietnamese language pack (LoRA), generation") {
                _ = await slm.complete(system: "Bạn là trợ lý.", history: [],
                                       user: "Chào bạn, hôm nay thế nào?",
                                       maxTokens: 24, temperature: 0, adapterPath: pack)
            }
        }
        await step("chat model, realistic long prompt (~1.5k tokens)") {
            let history = String(repeating: "Friend: Are we still on for dinner on Saturday at the new phở place? ", count: 90)
            _ = await slm.complete(system: "Suggest three short replies.", history: [],
                                   user: history, maxTokens: 64, temperature: 0)
        }
        let guardEngine = GuardEngine()
        await step("guardrail: Vietnamese classifier + rules") {
            _ = await guardEngine.warmUp(language: .vietnamese, region: .vietnam)
        }
        await step("guardrail: first classification") {
            _ = await guardEngine.classify("Tối nay đi ăn không?", language: .vietnamese)
        }
        let tagger = ImageTagEngine()
        await step("image tagger load") {
            _ = await tagger.warmUp(modelName: ChatService.imageTagModelResource,
                                    tagListName: ChatService.imageTagListResource)
        }
        await step("language detector (first use)") {
            _ = LangDetect.shared.ranked("Can we move the meeting to Thursday?")
        }
        let embedder = MessageEmbedder()
        try await step("embedder load (EmbeddingGemma 2 text+vision, CPU)") {
            try await embedder.load()
        }
        try await step("embedder: first text embedding") {
            _ = try await embedder.embed(documents: ["Can we move the meeting to Thursday?"])
        }
        let photo = "/Library/User Pictures/Animals/Eagle.heic"
        if let data = FileManager.default.contents(atPath: photo) {
            try await step("embedder: first photo embedding (vision encoder)") {
                _ = try await embedder.embed(image: data)
            }
        }

        print("\nFOOTPRINT | step                                                  delta      total")
        for (label, delta, total) in rows {
            print(String(format: "FOOTPRINT | %-52@ %+7.1f MB %7.1f MB", label as NSString, delta, total))
        }
        await embedder.unload()
        await slm.unload()
    }
}
