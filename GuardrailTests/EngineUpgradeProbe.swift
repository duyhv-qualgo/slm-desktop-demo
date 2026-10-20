import XCTest
@testable import SLMChat

/// Probe for the llama.cpp framework upgrade (b11669 + the Hadamard LoRA patch,
/// see scripts/build_llama_macos.sh): every desktop checkpoint, bare and under
/// its language packs, through `ChatLM` on Metal — plus Ternary Bonsai 2, which
/// only loads on the new build.
///
/// Prints `UPGRADE` lines (grep for them) so the same probe can be run before
/// and after a framework swap and the numbers compared. Asserts only what must
/// hold on any build: it loads on the GPU, the pack actually applies, and the
/// output is non-empty and in the target language.
final class EngineUpgradeProbe: XCTestCase {

    /// Long enough (~250 tokens) that prefill throughput is measured, not noise.
    private let source = """
        They moved the meeting with the client to tomorrow at 10am, so I need the slides tonight. \
        Please finish the pricing page first, then check the timeline on page six because the dates \
        changed after last week's call. Alex will join from Da Nang, so we should test the video link \
        before nine. If the projector in room three is still broken, book room five instead and tell \
        the front desk. I will bring the printed contracts and the signed NDA, and Lan will take notes. \
        After the meeting we can have lunch at the pho place on Le Loi street, my treat this time. \
        Let me know tonight if anything is missing so I can fix it before I go to bed.
        """

    /// Not bundled: Bonsai is evaluated from the iOS checkout's models-lab.
    private static let bonsaiPath: String = {
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        return repo.appendingPathComponent("../demo/models-lab/bonsai2-2b/Ternary-Bonsai-2-2B-Q2_0.gguf")
            .standardizedFileURL.path
    }()

    private static let vietnameseMarks = try! NSRegularExpression(pattern: "[ăâđêôơưạảấầẩẫậắằẳẵặẹẻẽếềểễệọỏốồổỗộớờởỡợụủứừửữự]",
                                                                  options: .caseInsensitive)

    private func isVietnamese(_ s: String) -> Bool {
        Self.vietnameseMarks.firstMatch(in: s, range: NSRange(s.startIndex..., in: s)) != nil
    }

    /// One load + one translation; returns the cleaned output.
    @discardableResult
    private func run(_ label: String, model path: String, pack: String?) throws -> String {
        let lm = ChatLM()
        guard lm.load(modelPath: path, backend: SLMBackend.gpu, cacheDir: nil, loraPath: pack) else {
            XCTFail("\(label): load failed: \(lm.lastError)")
            return ""
        }
        XCTAssertEqual(lm.activeBackend, SLMBackend.gpu, "\(label): not on Metal")
        XCTAssertEqual(lm.activeLoRAPath, pack, "\(label): pack did not apply")

        let prompt = Prompts.translate(source, from: .english, to: .vietnamese)
        // Warm-up on a DIFFERENT prompt: compiles the Metal pipelines without
        // leaving the measured prompt's prefix in the KV cache.
        _ = lm.generate(system: nil, history: [], user: Prompts.translate("Good morning.", from: .english, to: .vietnamese), maxTokens: 8,
                        temperature: 0.1, cacheable: false, adapterPath: pack)
        let raw = lm.generate(system: nil, history: [], user: prompt, maxTokens: 160,
                              temperature: 0.1, cacheable: false, adapterPath: pack) ?? ""
        let out = Prompts.parseTranslation(raw)
        let m = try XCTUnwrap(lm.lastTurnMetrics, label)
        print(String(format: "UPGRADE %@ pack=%@ prefill=%.0f tok/s (%d tok) decode=%.1f tok/s ttft=%.3fs out=%@",
                     label, pack.map { ($0 as NSString).lastPathComponent } ?? "none",
                     m.prefillTokensPerSecond, m.prefillTokenCount, m.decodeTokensPerSecond,
                     m.timeToFirstTokenSeconds, out))
        XCTAssertFalse(out.isEmpty, "\(label): empty output")
        return out
    }

    /// Each shipped checkpoint bare, then under its Vietnamese and English packs —
    /// the reload `ChatService` does when the language changes.
    func testShippedModelsAndPackSwitchOnMetal() throws {
        let models = DesktopModels.taskModels.filter(\.isAvailable)
        guard !models.isEmpty else { throw XCTSkip("No desktop .gguf bundled — run scripts/bootstrap.sh") }
        for option in models {
            let path = try XCTUnwrap(ResourceLookup.path(option.resourceName, ext: "gguf"))
            try run(option.id, model: path, pack: nil)
            for language in [TranslateLanguage.vietnamese, .english] {
                guard let adapter = LoRACatalog.adapter(baseModelID: option.id, language: language),
                      let pack = LoRACatalog.bundledPath(for: adapter) else { continue }
                try run("\(option.id)/\(language.rawValue)", model: path, pack: pack)
            }
        }
    }

    /// Bonsai 2 needs b11556+ (prism.hadamard runtime): an older build either
    /// refuses the Q2_0 type or loads it and emits gibberish, hence the language check.
    func testBonsaiLoadsOnMetalAndStaysCoherent() throws {
        guard FileManager.default.fileExists(atPath: Self.bonsaiPath) else {
            throw XCTSkip("Bonsai not found at \(Self.bonsaiPath)")
        }
        let out = try run("bonsai2-2b", model: Self.bonsaiPath, pack: nil)
        XCTAssertTrue(isVietnamese(out), "Bonsai output is not Vietnamese — Hadamard runtime missing? \(out)")
    }
    /// Hybrid attention/SSM checkpoints (`qwen35`): llama.cpp can't truncate their
    /// recurrent state, so a partial `seq_rm` is refused and every turn prefills
    /// cold. The app turns reuse off for them (`ChatService.prefixReuseActive`).
    private static let recurrentIDs: Set<String> = ["qwen3.5-0.8b", "bonsai2-2b"]

    /// Prefix reuse per checkpoint: a second suggest on the same conversation
    /// should re-read only the new tail — except on the recurrent checkpoints,
    /// where this pins the cold prefill so a llama.cpp that fixes it shows up here.
    func testPromptPrefixReuse() throws {
        var paths = DesktopModels.taskModels.filter(\.isAvailable).compactMap { option in
            ResourceLookup.path(option.resourceName, ext: "gguf").map { (option.id, $0) }
        }
        if !paths.contains(where: { $0.0 == "bonsai2-2b" }), FileManager.default.fileExists(atPath: Self.bonsaiPath) {
            paths.append(("bonsai2-2b", Self.bonsaiPath))
        }
        for (id, path) in paths {
            let lm = ChatLM()
            guard lm.load(modelPath: path, backend: SLMBackend.gpu, cacheDir: nil, loraPath: nil) else {
                XCTFail("\(id): load failed: \(lm.lastError)"); continue
            }
            var history = [ChatMessage(sender: .friend, text: "Hey, are you free this weekend?"),
                           ChatMessage(sender: .you, text: "Mostly, why?"),
                           ChatMessage(sender: .friend, text: "Want to grab dinner Saturday at 7?")]
            _ = lm.generate(system: nil, history: [], user: Prompts.suggestReplies(history: history, language: .english),
                            maxTokens: 32, temperature: 0.8, cacheable: true, adapterPath: nil)
            let first = try XCTUnwrap(lm.lastTurnMetrics)
            history.append(ChatMessage(sender: .you, text: "Sure, where?"))
            history.append(ChatMessage(sender: .friend, text: "The new pho place on Le Loi."))
            _ = lm.generate(system: nil, history: [], user: Prompts.suggestReplies(history: history, language: .english),
                            maxTokens: 32, temperature: 0.8, cacheable: true, adapterPath: nil)
            let second = try XCTUnwrap(lm.lastTurnMetrics)
            print("UPGRADE \(id) prefix-reuse first: prompt=\(first.promptTokenCount) prefilled=\(first.prefillTokenCount)"
                  + " | second: prompt=\(second.promptTokenCount) match=\(second.prefixMatchTokenCount)"
                  + " cached=\(second.cachedPrefixTokenCount) prefilled=\(second.prefillTokenCount) slot=\(second.cacheSlotIndex)")
            if Self.recurrentIDs.contains(id) {
                XCTAssertEqual(second.cachedPrefixTokenCount, 0, "\(id): recurrent prefix reuse works now — revisit ChatService.prefixReuseActive")
            } else {
                XCTAssertGreaterThan(second.cachedPrefixTokenCount, 0, "\(id): no prefix reuse")
            }
        }
    }
    /// The same Vietnamese pack at rank 32 and zero-padded to rank 64 (identical
    /// delta, see scripts/peft_lora_to_gguf.py --pad-rank) through the app's own
    /// engine on Metal: what the pack's rank costs in prefill and decode.
    func testBonsaiPackRankSpeed() throws {
        let dir = (Self.bonsaiPath as NSString).deletingLastPathComponent
        let packs = ["r32": "lora-bonsai2-2b-vietnamese-r32.gguf", "r64": "lora-bonsai2-2b-vietnamese-r64pad.gguf"]
            .mapValues { (dir as NSString).appendingPathComponent($0) }
        guard FileManager.default.fileExists(atPath: Self.bonsaiPath),
              packs.values.allSatisfy(FileManager.default.fileExists) else {
            throw XCTSkip("Bonsai or its rank-32/64 packs not found in \(dir)")
        }
        let long = Prompts.summarize([ChatMessage(sender: .friend, text: String(repeating: source + " ", count: 6))],
                                     language: "Vietnamese")
        for label in ["none", "r32", "r64"] {
            let pack = packs[label]
            let lm = ChatLM()
            guard lm.load(modelPath: Self.bonsaiPath, backend: SLMBackend.gpu, cacheDir: nil, loraPath: pack) else {
                XCTFail("\(label): load failed: \(lm.lastError)"); continue
            }
            XCTAssertEqual(lm.activeLoRAPath, pack, "\(label): pack did not apply")
            _ = lm.generate(system: nil, history: [], user: "Xin chào", maxTokens: 8,
                            temperature: 0.1, cacheable: false, adapterPath: pack)
            var prefill: [Double] = [], decode: [Double] = [], ttft: [Double] = [], tokens = 0
            for _ in 0..<3 {
                _ = lm.generate(system: nil, history: [], user: long, maxTokens: 128,
                                temperature: 0.1, cacheable: false, adapterPath: pack)
                let m = try XCTUnwrap(lm.lastTurnMetrics)
                prefill.append(m.prefillTokensPerSecond); decode.append(m.decodeTokensPerSecond)
                ttft.append(m.timeToFirstTokenSeconds); tokens = Int(m.prefillTokenCount)
            }
            let med = { (xs: [Double]) in xs.sorted()[xs.count / 2] }
            print(String(format: "UPGRADE rank-speed pack=%@ prompt=%d tok prefill=%.0f tok/s decode=%.1f tok/s ttft=%.3fs",
                         label, tokens, med(prefill), med(decode), med(ttft)))
        }
    }
}
