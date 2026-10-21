import Foundation

/// The checkpoints the macOS app ships and offers: a desktop-only subset of
/// `ModelCatalog`. The catalogs belong to the AI engineer and stay identical to
/// iOS, so the subset is applied here rather than by editing them.
///
/// Qwen2.5 0.5B and Qwen3.5 0.8B. Gemma-3 270M and Qwen3 0.6B are left out,
/// along with their LoRA packs (`scripts/bootstrap.sh` no longer copies them).
enum DesktopModels {
    static let ids: [String] = ["qwen3.5-0.8b", "qwen2.5-0.5b-v6.1"]

    /// Qwen3.5 — the strongest of the two. The catalog's own default is
    /// Gemma-3 270M, which the desktop build doesn't ship.
    static var defaultTaskModel: SLMModelOption { ModelCatalog.taskModel(for: ids[0]) }

    /// In catalog order, so Settings lists them the same way as on iOS.
    static var taskModels: [SLMModelOption] {
        ModelCatalog.taskModels.filter { ids.contains($0.id) }
    }

    /// Look up an offered option by id, falling back to the desktop default
    /// rather than the catalog's.
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
