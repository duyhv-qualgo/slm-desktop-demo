import Foundation

/// One selectable on-device SLM checkpoint. `resourceName` is the basename of the
/// `.gguf` file in the app bundle (`<resourceName>.gguf`); the llama.cpp runtime
/// loads it the same way regardless of which model it is.
struct SLMModelOption: Identifiable, Equatable {
    /// Stable key persisted in UserDefaults (independent of display name).
    let id: String
    let displayName: String
    let subtitle: String
    /// Basename of the `.gguf` in Resources/ (no extension).
    let resourceName: String

    /// Whether the model's file is actually bundled. Alt models are dropped into
    /// Resources/ later, so this is checked at runtime to grey out missing ones.
    var isAvailable: Bool { ResourceLookup.path(resourceName, ext: "gguf") != nil }
}

/// The SLMs the user can switch the on-device feature engine between (translate /
/// suggest / summarize / event / guardrail). All run on llama.cpp; the turn
/// builder in `Prompts` is template-agnostic (Gemma + Qwen/ChatML).
///
/// Every entry ships language packs that were trained against it. That is the
/// bar for being listed: stock versions of these same Qwen checkpoints used to
/// appear here and were greyed out on every machine — their weights are in
/// `legacy/`, none had a pack, and an unselectable row that can only ever run
/// bare weights is a worse demo than no row. The finetuned qwen2.5 and qwen3.5
/// rows below are those checkpoints earning their place back, not the old bare
/// entries restored.
enum ModelCatalog {

    /// The bundled gemma-3 270m finetune is the default and is always present.
    ///
    /// `displayName` carries no finetune version — Settings shows "Qwen3 0.6B",
    /// not "Qwen3 0.6B v6". The version still lives in `id` and `resourceName`,
    /// which is where it matters: `id` is the persisted key and the join the LoRA
    /// packs resolve against, `resourceName` is the actual filename. Retitling a
    /// row is therefore a display-only edit; changing its `id` is not, and would
    /// silently reset every user's selection and orphan their installed packs.
    static let taskModels: [SLMModelOption] = [
        SLMModelOption(id: "gemma3-270m",
                       displayName: "Gemma-3 270M",
                       subtitle: "Default · on-device finetune",
                       resourceName: "model"),
        // The qwen2.5 v6.1 finetune. Smallest and oldest of the three Qwen rows,
        // and the only plain `qwen2` architecture among them — no SSM blocks, no
        // reasoning template (its ChatML template has no `<think>`, so `ChatLM`'s
        // empty-think-block suppression is inert here). It is listed because it
        // is the cheapest checkpoint that still has a full seven-language pack
        // set, which makes it the useful floor to benchmark the other two against.
        //
        // Q8_0 (~506 MB) over the ~379 MB Q4_K_M build, for the prefill reason
        // documented on the qwen3-0.6b-v6 row below; both builds are in
        // `out/qwen2.5-v6.1/`.
        SLMModelOption(id: "qwen2.5-0.5b-v6.1",
                       displayName: "Qwen2.5 0.5B",
                       subtitle: "Multilingual · 7 language packs",
                       resourceName: "qwen2.5-0.5b-v6.1"),
        // The multilingual finetune: seven LoRA language packs, one per language
        // (see LoRACatalog).
        //
        // Bundled at Q8_0 (~639 MB), NOT the ~397 MB Q4_K_M build of the same
        // weights, and the reason is speed rather than quality. Measured on the
        // same prompt, same code path, identical weights:
        //
        //     Q4_K_M   prefill 102 tok/s   decode 66   ratio 1.6x
        //     Q8_0     prefill 554 tok/s   decode 88   ratio 6.3x
        //
        // 5.4x prefill for +242 MB. The cause is that llama.cpp's batched GEMM
        // (llamafile_sgemm) covers Q8_0/Q4_0 but not the K-quants, so Q4_K_M runs
        // per-row dot products for the prompt pass as well as for decode — the
        // 1.6x ratio is the tell. Prefill is ~73% of an auto-suggest turn, so this
        // is the single biggest latency lever in the app. Re-check this before
        // switching quant family again; it is a property of the build, not of the
        // model.
        SLMModelOption(id: "qwen3-0.6b-v6",
                       displayName: "Qwen3 0.6B",
                       subtitle: "Multilingual · 7 language packs",
                       resourceName: "qwen3-0.6b-v6"),
        // The newer multilingual finetune. Same shape as the v6 row above —
        // one LoRA language pack per language, seven of them — on a larger and
        // architecturally different base.
        //
        // `general.architecture` is `qwen35`, a hybrid attention/SSM model, not
        // the plain transformer `qwen3` of the row above: only every 4th block
        // is full attention (`qwen35.full_attention_interval`), the rest are
        // SSM. The bundled llama.xcframework supports it (the build carries the
        // `qwen35` arch); an older llama.cpp will fail to load this file with an
        // unknown-architecture error rather than misbehave, so if this row stops
        // loading after a framework change, that is the first thing to check.
        //
        // Its embedded chat template is Qwen3-VL's, which contains `<|im_start|>`
        // and so is rendered by llama.cpp's built-in ChatML path — the same route
        // the v6 checkpoint takes. It also contains `<think>`, so `ChatLM`'s
        // empty-think-block suppression applies here too.
        //
        // Bundled at Q8_0 (~774 MB) rather than the ~505 MB Q4_K_M build of the
        // same weights, for the prefill reason documented on the v6 row: the
        // K-quants miss llama.cpp's batched GEMM and pay per-row dot products on
        // the prompt pass. Both builds are in `qwen3.5-multilingual/` if that
        // trade needs re-measuring on this architecture.
        SLMModelOption(id: "qwen3.5-0.8b",
                       displayName: "Qwen3.5 0.8B",
                       subtitle: "Multilingual · 7 language packs",
                       resourceName: "qwen3.5-0.8b"),
    ]

    static let defaultTaskModel = taskModels[0]

    /// Look up an option by id, falling back to the default for unknown ids.
    static func taskModel(for id: String) -> SLMModelOption {
        taskModels.first { $0.id == id } ?? defaultTaskModel
    }
}
