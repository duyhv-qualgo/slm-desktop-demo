import Foundation

/// The checkpoints the macOS app ships and offers: a desktop-only subset of
/// `ModelCatalog`. The catalogs belong to the AI engineer and stay identical to
/// iOS, so the subset is applied here rather than by editing them.
///
/// Qwen2.5 0.5B and Qwen3.5 0.8B. Gemma-3 270M and Qwen3 0.6B are left out,
/// along with their LoRA packs (`scripts/bootstrap.sh` no longer copies them).
/// Plus `evaluationModels`, which the catalog doesn't have at all.
enum DesktopModels {
    static let ids: [String] = ["qwen3.5-0.8b", "qwen2.5-0.5b-v6.1"]

    /// Desktop-only checkpoints under evaluation, NOT in `ModelCatalog`. Offered
    /// only when the file is present (bootstrap copies it if the iOS checkout has
    /// it), and never the default.
    ///
    /// Ternary Bonsai 2 2B: the base release, no finetune and no language packs
    /// yet, so it runs the app's prompts zero-shot and only Vietnamese/English are
    /// offered (`TranslateLanguage.isSupported`). Needs the llama.xcframework from
    /// scripts/build_llama_macos.sh — llama.cpp b11556+ for the folded Hadamard
    /// weights, plus the patch that lets its LoRA packs apply.
    static let evaluationModels: [SLMModelOption] = [
        SLMModelOption(id: "bonsai2-2b",
                       displayName: "Bonsai 2 2B",
                       subtitle: "Ternary · evaluation · Vietnamese pack",
                       resourceName: "Ternary-Bonsai-2-2B-Q2_0"),
    ]

    /// Language packs for `evaluationModels`, merged into `LoRACatalog`'s lookups
    /// (`LoRACatalog.allAdapters`) so the catalog's own rows stay identical to iOS.
    ///
    /// Bonsai 2 2B Vietnamese: the team's first pack (PEFT r=32, alpha 64),
    /// converted with scripts/peft_lora_to_gguf.py and zero-padded to rank 64
    /// (`--pad-rank 64`, same delta): rank 32 runs Metal prefill at about half
    /// speed — 552 vs 958 tok/s in ChatLM on an 884-token prompt. Bundled, so it
    /// applies on first launch. Needs the patched llama.xcframework: official
    /// llama.cpp refuses LoRA on Bonsai's Hadamard-folded weights.
    static let evaluationAdapters: [LoRAAdapter] = [
        LoRAAdapter(id: "bonsai2-2b-vi",
                    baseModelID: "bonsai2-2b",
                    displayName: "Vietnamese pack",
                    languages: [.vietnamese],
                    fileName: "lora-bonsai2-2b-vietnamese.gguf",
                    sizeBytes: 125_063_328,
                    sha256: "55ad4aabaea2fcc9a9447669cf3b6d6efb75f80e8e925f6ff92db1fa13705917"),
    ]

    /// Qwen3.5 — the strongest of the two. The catalog's own default is
    /// Gemma-3 270M, which the desktop build doesn't ship.
    static var defaultTaskModel: SLMModelOption { ModelCatalog.taskModel(for: ids[0]) }

    /// In catalog order, so Settings lists them the same way as on iOS, then the
    /// evaluation models that are present.
    static var taskModels: [SLMModelOption] {
        ModelCatalog.taskModels.filter { ids.contains($0.id) } + evaluationModels.filter(\.isAvailable)
    }

    /// Look up an offered option by id, falling back to the desktop default —
    /// `ModelCatalog.taskModel(for:)` doesn't know the evaluation models.
    static func taskModel(for id: String) -> SLMModelOption {
        taskModels.first { $0.id == id } ?? defaultTaskModel
    }

    /// `id` if the desktop app offers it, else the desktop default — for a
    /// saved choice from before the subset (e.g. Qwen3 0.6B).
    static func resolve(_ id: String?) -> SLMModelOption {
        let option = id.flatMap { id in taskModels.first { $0.id == id } } ?? defaultTaskModel
        return option.isAvailable ? option : defaultTaskModel
    }
}
