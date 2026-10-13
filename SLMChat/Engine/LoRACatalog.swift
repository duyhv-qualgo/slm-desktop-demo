import Foundation

/// One LoRA adapter: the language pack for ONE base model.
///
/// An adapter is bound to the exact checkpoint it was trained against — the same
/// `.gguf` weights, not merely the same architecture — so it is always resolved
/// by (base model, language), never by language alone. Applying a Qwen3 adapter to
/// Gemma-3 does not fail loudly; it produces fluent nonsense.
///
/// A pack may be task-trained (the shipping Gemma-3 pack covers all six features)
/// or style-only, and may cover one language or several. Both shapes load
/// identically — the difference is only in what it was trained on, and in whether
/// editing `Prompts` invalidates it: a task-trained pack learns the prompt
/// templates, so retuning one is a reason to retrain the pack.
struct LoRAAdapter: Identifiable, Equatable {
    /// Stable key, also the UserDefaults-safe identity of the pack.
    let id: String
    /// Base model this adapter was trained against (`SLMModelOption.id`).
    let baseModelID: String
    let displayName: String

    /// Native languages this pack serves. A SET, not a single language: a pack
    /// trained on several languages at once is one file covering all of them, and
    /// splitting it into per-language rows would mean shipping duplicates of the
    /// same bytes and reloading the model on a switch that changes nothing.
    let languages: Set<TranslateLanguage>

    /// On-disk name, unique across the catalog so every pack shares one directory.
    let fileName: String
    /// Download size, for the Settings row. 0 when unknown (side-load only).
    let sizeBytes: Int64
    /// Expected SHA-256 of the `.gguf`, lowercase hex. Nil skips verification —
    /// acceptable for a bundled or hand-placed file, never for one off a URL.
    let sha256: String?

    /// Demo affordance: the pack ships inside the app, but is presented as if it
    /// had to be fetched. `LoRACatalog.installedPath` ignores the bundled copy
    /// until an "install" has run, and `LoRAStore` copies it out of the bundle
    /// behind a progress bar instead of hitting the network.
    ///
    /// This exists so the full Settings flow — available → downloading → installed
    /// → applied — can be shown without hosting anything. A shipping pack sets it
    /// false and is fetched from `remoteURL` for real; nothing else differs, so
    /// the path being demonstrated is the one that will actually run.
    var simulatedDownload: Bool = false

    /// The languages this pack covers, in `TranslateLanguage` order, for display.
    var languageList: String {
        TranslateLanguage.allCases
            .filter { languages.contains($0) }
            .map(\.name)
            .joined(separator: " · ")
    }

    /// Where this adapter is published, or nil if it has no home yet — see
    /// `LoRACatalog.remoteBaseURL`. A nil URL means side-load only: the row still
    /// works, it just can't download.
    var remoteURL: URL? {
        LoRACatalog.remoteBaseURL?.appendingPathComponent(fileName)
    }

    var displaySize: String {
        guard sizeBytes > 0 else { return "—" }
        return ByteCountFormatter.string(fromByteCount: sizeBytes, countStyle: .file)
    }
}

/// The language packs the app knows about, and where they live on disk.
///
/// Adapters are NOT bundled (except a default, see `bundledPath`): they are
/// per-language, and shipping every language to every user would defeat the point
/// of using adapters at all. The reason the app ships base + adapter rather than a
/// merged per-language model is update economics — a retuned language pack is a
/// ~15 MB download instead of re-shipping the whole ~600 MB checkpoint, and one
/// language can be fixed or rolled back without touching the others.
enum LoRACatalog {

    /// Where published adapters are fetched from. Set this to the CDN/bucket
    /// prefix once the packs are hosted; until then every adapter resolves to a
    /// nil `remoteURL` and Settings offers side-loading only.
    ///
    /// Adapters MUST be rebuilt whenever the base `.gguf` they target changes.
    /// Bumping a checkpoint without re-publishing its packs is the one failure
    /// this design can't detect from inside the app — hence `sha256`, which at
    /// least catches a truncated or swapped file.
    ///
    /// To exercise the download path before anything is hosted, serve the pack
    /// directory and point this at it:
    ///
    ///     cd gguf/gemma3 && python3 -m http.server 8000
    ///     static let remoteBaseURL = URL(string: "http://localhost:8000")
    ///
    /// Plain HTTP needs an ATS exemption (`NSAllowsLocalNetworking`) in
    /// Info.plist — add it for local testing only, never for a shipping build.
    static let remoteBaseURL: URL? = nil

    /// Every (model, language) pack that exists or is planned.
    ///
    /// Deliberately NOT the full cross-product of `ModelCatalog.taskModels` ×
    /// `TranslateLanguage`: models × languages grows multiplicatively and each
    /// cell is a real training run to produce, evaluate and version. Only list a
    /// pack that is actually built. A missing pair is not an error — the engine
    /// runs on bare weights, which is the correct configuration for any language
    /// the base already handles well.
    // Multi-task packs for the bundled Gemma-3 270M checkpoint: all six on-device
    // features, targeting every projection (attn q/k/v/o + ffn gate/up/down).
    // Trained against `gemma-3-270m-it`, which is what ships as
    // Resources/model.gguf — they are valid against those exact weights and no
    // others.
    //
    // NOTE: both entries currently point at copies of the SAME v5 adapter, which
    // was trained on Vietnamese and English together. They are split into one pack
    // per language so the app exercises the shape it will really ship — separate
    // rows, independent install state, and a genuine model reload when the
    // language changes. Replace each file as the per-language packs are trained;
    // nothing here changes but the checksums.
    static let adapters: [LoRAAdapter] = [
        // Bundled, so the default language works on first launch and offline.
        LoRAAdapter(id: "gemma3-270m-vi-v5",
                    baseModelID: "gemma3-270m",
                    displayName: "Vietnamese pack v5",
                    languages: [.vietnamese],
                    fileName: "lora-gemma3-270m-vietnamese.gguf",
                    sizeBytes: 15_205_856,
                    sha256: "4b1968c9637322248f1247bd969250d50804240185f5a689d89db1887633e3dd"),
        // The on-demand case. Bundled like the one above, but flagged as a
        // simulated download so the demo can show a pack being acquired: English
        // starts uninstalled and runs on the bare checkpoint until "Get" is
        // tapped. Removing it returns the row to that state, so it replays.
        LoRAAdapter(id: "gemma3-270m-en-v5",
                    baseModelID: "gemma3-270m",
                    displayName: "English pack v5",
                    languages: [.english],
                    fileName: "lora-gemma3-270m-english.gguf",
                    sizeBytes: 15_205_856,
                    sha256: "4b1968c9637322248f1247bd969250d50804240185f5a689d89db1887633e3dd",
                    simulatedDownload: true),

        // Every Qwen pack below is rank 64: trained at rank 32, then zero-padded
        // (`scripts/pad_lora_rank.py`, alpha doubled, identical output). Metal
        // tiles a matmul only when its inner dimension is >= 64, and in
        // lora_b that dimension is the rank, so rank-32 packs ran GPU prefill
        // ~3x slower. Keep new packs at rank >= 64. See docs/PERFORMANCE.md §11.
        // The Gemma packs stay rank 32: Gemma runs on the CPU, where rank
        // doesn't pick the kernel.
        //
        // Qwen2.5 0.5B v6.1 — the same seven-language set on the smallest and
        // oldest of the three Qwen bases. Plain `qwen2` attention, so these
        // adapters target the classic seven projections (attn q/k/v/output +
        // ffn gate/up/down) and share no tensor names with the hybrid qwen35
        // packs further down. Install shape matches the other sets: Vietnamese
        // bundled, the rest simulated downloads. No German row, for the reason
        // given on the qwen3.5 set.
        LoRAAdapter(id: "qwen2.5-0.5b-v6.1-vi",
                    baseModelID: "qwen2.5-0.5b-v6.1",
                    displayName: "Vietnamese pack v6.1",
                    languages: [.vietnamese],
                    fileName: "lora-qwen2.5-0.5b-v6.1-vietnamese.gguf",
                    sizeBytes: 70_408_736,
                    sha256: "f17087051e4358590616ce9064774b5ea0912febd7e12395195274f8d1e78a50"),
        LoRAAdapter(id: "qwen2.5-0.5b-v6.1-en",
                    baseModelID: "qwen2.5-0.5b-v6.1",
                    displayName: "English pack v6.1",
                    languages: [.english],
                    fileName: "lora-qwen2.5-0.5b-v6.1-english.gguf",
                    sizeBytes: 70_408_736,
                    sha256: "c9336a716de104bcc38b24cb4de7f66128cd688d02f3ceb5399783bfb2f91597",
                    simulatedDownload: true),
        LoRAAdapter(id: "qwen2.5-0.5b-v6.1-ar",
                    baseModelID: "qwen2.5-0.5b-v6.1",
                    displayName: "Arabic pack v6.1",
                    languages: [.arabic],
                    fileName: "lora-qwen2.5-0.5b-v6.1-arabic.gguf",
                    sizeBytes: 70_408_736,
                    sha256: "7d8d9ce29be620b43bc63bd859f006844b1976a420dda9dc137af95bb316eea4",
                    simulatedDownload: true),
        LoRAAdapter(id: "qwen2.5-0.5b-v6.1-es",
                    baseModelID: "qwen2.5-0.5b-v6.1",
                    displayName: "Spanish pack v6.1",
                    languages: [.spanish],
                    fileName: "lora-qwen2.5-0.5b-v6.1-spanish.gguf",
                    sizeBytes: 70_408_736,
                    sha256: "93e6aebbe31e4d4acb90d6a6013031d5da5a9f5538708d70dd5d9f9f1593c8d4",
                    simulatedDownload: true),
        LoRAAdapter(id: "qwen2.5-0.5b-v6.1-fr",
                    baseModelID: "qwen2.5-0.5b-v6.1",
                    displayName: "French pack v6.1",
                    languages: [.french],
                    fileName: "lora-qwen2.5-0.5b-v6.1-french.gguf",
                    sizeBytes: 70_408_736,
                    sha256: "28a68d10f0d88bba09c6673721ff8e25353392185505ad4eab01afeb70b5f891",
                    simulatedDownload: true),
        LoRAAdapter(id: "qwen2.5-0.5b-v6.1-ru",
                    baseModelID: "qwen2.5-0.5b-v6.1",
                    displayName: "Russian pack v6.1",
                    languages: [.russian],
                    fileName: "lora-qwen2.5-0.5b-v6.1-russian.gguf",
                    sizeBytes: 70_408_736,
                    sha256: "1c312023c114c509029f25de48105e8316a055a1bc4f7af9ebb871a7adf5604c",
                    simulatedDownload: true),
        LoRAAdapter(id: "qwen2.5-0.5b-v6.1-zh",
                    baseModelID: "qwen2.5-0.5b-v6.1",
                    displayName: "Chinese pack v6.1",
                    languages: [.chinese],
                    fileName: "lora-qwen2.5-0.5b-v6.1-chinese.gguf",
                    sizeBytes: 70_408_736,
                    sha256: "c0921227954ffd57fe8462a79069555d847c878e454774cd6545ae7e0e1aaffb",
                    simulatedDownload: true),

        // Qwen3 0.6B v6 — the multilingual set. One pack per language, all seven
        // trained against the same v6 base, and all genuinely distinct files
        // (their checksums differ, unlike the two Gemma-3 rows above, which are
        // still copies of one bilingual adapter).
        //
        // These are what make the extra languages selectable at all:
        // `TranslateLanguage.isSupported(forBaseModel:)` treats Vietnamese and
        // English as the baseline every checkpoint handles and derives the rest
        // from this list, so Arabic … Chinese light up on this model and stay
        // greyed out on the others.
        //
        // Vietnamese is a plain bundled pack so the default language works on
        // first launch and offline. The other six are flagged as simulated
        // downloads: each starts uninstalled and runs on the bare checkpoint
        // until "Get" is tapped, which is both the demo of the acquire flow and
        // an honest picture of the shipping shape — a user fetches the one or two
        // languages they speak, not 275 MB of all seven.
        LoRAAdapter(id: "qwen3-0.6b-v6-vi",
                    baseModelID: "qwen3-0.6b-v6",
                    displayName: "Vietnamese pack v6",
                    languages: [.vietnamese],
                    fileName: "lora-qwen3-0.6b-v6-vietnamese.gguf",
                    sizeBytes: 80_767_264,
                    sha256: "a08a6234fb76bfd0d63ac29b469cf2b6f6a95dc204a0217c8f7ef272fb22c62a"),
        LoRAAdapter(id: "qwen3-0.6b-v6-en",
                    baseModelID: "qwen3-0.6b-v6",
                    displayName: "English pack v6",
                    languages: [.english],
                    fileName: "lora-qwen3-0.6b-v6-english.gguf",
                    sizeBytes: 80_767_264,
                    sha256: "f85a343dbfffdb4e3c33b93fb17c45fb3acb94b8a1c073624237a1d0e4fc2987",
                    simulatedDownload: true),
        LoRAAdapter(id: "qwen3-0.6b-v6-ar",
                    baseModelID: "qwen3-0.6b-v6",
                    displayName: "Arabic pack v6",
                    languages: [.arabic],
                    fileName: "lora-qwen3-0.6b-v6-arabic.gguf",
                    sizeBytes: 80_767_264,
                    sha256: "04925b5069b079166e01fe052a8b5397edf9c218eb88f5f3cb78d5443e6c6dac",
                    simulatedDownload: true),
        LoRAAdapter(id: "qwen3-0.6b-v6-es",
                    baseModelID: "qwen3-0.6b-v6",
                    displayName: "Spanish pack v6",
                    languages: [.spanish],
                    fileName: "lora-qwen3-0.6b-v6-spanish.gguf",
                    sizeBytes: 80_767_264,
                    sha256: "f4784906b217f9e7a350412c595264651c088b33bdf316fb54bb40097aba64ca",
                    simulatedDownload: true),
        LoRAAdapter(id: "qwen3-0.6b-v6-fr",
                    baseModelID: "qwen3-0.6b-v6",
                    displayName: "French pack v6",
                    languages: [.french],
                    fileName: "lora-qwen3-0.6b-v6-french.gguf",
                    sizeBytes: 80_767_264,
                    sha256: "1244610e715e5a3dc02928e5689ef631c411069ba05e4a4f36ad408b3287f8f2",
                    simulatedDownload: true),
        LoRAAdapter(id: "qwen3-0.6b-v6-ru",
                    baseModelID: "qwen3-0.6b-v6",
                    displayName: "Russian pack v6",
                    languages: [.russian],
                    fileName: "lora-qwen3-0.6b-v6-russian.gguf",
                    sizeBytes: 80_767_264,
                    sha256: "1876f20d747a689b01d5c792f2c4d3bb272665eebdd636e6ef3810cdb5293235",
                    simulatedDownload: true),
        LoRAAdapter(id: "qwen3-0.6b-v6-zh",
                    baseModelID: "qwen3-0.6b-v6",
                    displayName: "Chinese pack v6",
                    languages: [.chinese],
                    fileName: "lora-qwen3-0.6b-v6-chinese.gguf",
                    sizeBytes: 80_767_264,
                    sha256: "1acc6c57d82cae5bb8d87bed39cbd88d91a071334b2db1df9da6dcdd6b6cf087",
                    simulatedDownload: true),

        // Qwen3.5 0.8B — the same seven-language set on the newer base. These
        // adapters target the hybrid `qwen35` blocks (attn_qkv/attn_gate,
        // ssm_alpha/ssm_beta/ssm_out, ffn_*), so they are structurally
        // incompatible with the v6 packs above and cannot be swapped between the
        // two models — another reason adapters resolve by (base model, language).
        //
        // Same install shape as v6: Vietnamese is a plain bundled pack so the
        // default language works on first launch, the other six are simulated
        // downloads.
        //
        // The training run also produced a German pack. It is deliberately NOT
        // listed: `TranslateLanguage` has no `.german` case, so an entry here
        // would not compile, and adding one is not a catalog change — it also
        // requires the `NLLanguage` mapping and the `GuardRegion.germany` rule
        // routing that its own comment flags as waiting on exactly this. The file
        // stays in `qwen3.5-multilingual/lora/` until then.
        LoRAAdapter(id: "qwen3.5-0.8b-vi",
                    baseModelID: "qwen3.5-0.8b",
                    displayName: "Vietnamese pack",
                    languages: [.vietnamese],
                    fileName: "lora-qwen3.5-0.8b-vietnamese.gguf",
                    sizeBytes: 86_606_944,
                    sha256: "fe9e123615fe4514f9ea1a503805c78c9a3977ae5b079f303657ad68940ceda2"),
        LoRAAdapter(id: "qwen3.5-0.8b-en",
                    baseModelID: "qwen3.5-0.8b",
                    displayName: "English pack",
                    languages: [.english],
                    fileName: "lora-qwen3.5-0.8b-english.gguf",
                    sizeBytes: 86_606_944,
                    sha256: "ff19b7c2689285a42698cca23b2ac6b5278e54607b54ec080efdbaf1817e2e6c",
                    simulatedDownload: true),
        LoRAAdapter(id: "qwen3.5-0.8b-ar",
                    baseModelID: "qwen3.5-0.8b",
                    displayName: "Arabic pack",
                    languages: [.arabic],
                    fileName: "lora-qwen3.5-0.8b-arabic.gguf",
                    sizeBytes: 86_606_944,
                    sha256: "62faabf6ef022745d1b176b04c1f5ed547ae6c8d697291fc94f66c4bf1f91222",
                    simulatedDownload: true),
        LoRAAdapter(id: "qwen3.5-0.8b-es",
                    baseModelID: "qwen3.5-0.8b",
                    displayName: "Spanish pack",
                    languages: [.spanish],
                    fileName: "lora-qwen3.5-0.8b-spanish.gguf",
                    sizeBytes: 86_606_944,
                    sha256: "8443e18836264115f942eebd745cbea12af8d29f41dde3697f38f50443622416",
                    simulatedDownload: true),
        LoRAAdapter(id: "qwen3.5-0.8b-fr",
                    baseModelID: "qwen3.5-0.8b",
                    displayName: "French pack",
                    languages: [.french],
                    fileName: "lora-qwen3.5-0.8b-french.gguf",
                    sizeBytes: 86_606_944,
                    sha256: "a88a18b6490ac206115d28188ed8e0bd8be8092473e06f4fb80ce115265fadee",
                    simulatedDownload: true),
        LoRAAdapter(id: "qwen3.5-0.8b-ru",
                    baseModelID: "qwen3.5-0.8b",
                    displayName: "Russian pack",
                    languages: [.russian],
                    fileName: "lora-qwen3.5-0.8b-russian.gguf",
                    sizeBytes: 86_606_944,
                    sha256: "687b45047606a9fc48e67799367132eeaf2c2c65cb9d4fb2b16643078dd61a10",
                    simulatedDownload: true),
        LoRAAdapter(id: "qwen3.5-0.8b-zh",
                    baseModelID: "qwen3.5-0.8b",
                    displayName: "Chinese pack",
                    languages: [.chinese],
                    fileName: "lora-qwen3.5-0.8b-chinese.gguf",
                    sizeBytes: 86_606_944,
                    sha256: "5551861e100665099297ce68cfa820c35a2aefdf0ae3982b58785710de36cac8",
                    simulatedDownload: true),
    ]

    /// The pack serving this (model, language) pair, or nil to run on bare
    /// weights.
    static func adapter(baseModelID: String, language: TranslateLanguage) -> LoRAAdapter? {
        adapters.first { $0.baseModelID == baseModelID && $0.languages.contains(language) }
    }

    /// Packs for one base model, in catalog order — what Settings lists once a
    /// model is selected.
    static func adapters(forBaseModel id: String) -> [LoRAAdapter] {
        adapters.filter { $0.baseModelID == id }
    }

    // MARK: - Locations

    /// Directory downloaded adapters are installed into. Application Support is
    /// the right home: it is backed up, not user-visible, and not purged the way
    /// Caches can be mid-session.
    static var installDirectory: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory,
                                            in: .userDomainMask)[0]
            .appendingPathComponent("LoRA", isDirectory: true)
        try? FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return base
    }

    static func installURL(for adapter: LoRAAdapter) -> URL {
        installDirectory.appendingPathComponent(adapter.fileName)
    }

    /// A copy dropped into Documents by hand (Finder file sharing, or Xcode's
    /// container download) — how a pack gets tested before anything is hosted.
    static func sideloadURL(for adapter: LoRAAdapter) -> URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent(adapter.fileName)
    }

    /// A pack shipped inside the app, so the default language works on first run
    /// and offline, before anything has been downloaded.
    static func bundledPath(for adapter: LoRAAdapter) -> String? {
        let stem = (adapter.fileName as NSString).deletingPathExtension
        return ResourceLookup.path(stem, ext: "gguf")
    }

    /// Resolve the adapter file to hand `ChatLM`, or nil to load bare weights.
    ///
    /// Order is downloaded → side-loaded → bundled, so a freshly downloaded pack
    /// always wins over a stale bundled one, and a hand-placed file beats the
    /// bundle during development without needing a rebuild.
    ///
    /// A `simulatedDownload` pack stops before the bundle: its bundled copy is the
    /// stand-in for a server, not an installed pack, so it must not resolve until
    /// `LoRAStore` has copied it into place.
    static func installedPath(for adapter: LoRAAdapter) -> String? {
        let fm = FileManager.default
        let installed = installURL(for: adapter)
        if fm.fileExists(atPath: installed.path) { return installed.path }
        let sideload = sideloadURL(for: adapter)
        if fm.fileExists(atPath: sideload.path) { return sideload.path }
        guard !adapter.simulatedDownload else { return nil }
        return bundledPath(for: adapter)
    }

    /// Adapter path for a (model, language) pair, or nil for bare weights —
    /// the one call the engine layer needs.
    static func installedPath(baseModelID: String, language: TranslateLanguage) -> String? {
        guard let a = adapter(baseModelID: baseModelID, language: language) else { return nil }
        return installedPath(for: a)
    }

    static func isInstalled(_ adapter: LoRAAdapter) -> Bool {
        installedPath(for: adapter) != nil
    }

    /// Remove a downloaded pack. Side-loaded and bundled copies are left alone —
    /// the first isn't ours to delete, the second can't be.
    static func removeDownload(_ adapter: LoRAAdapter) throws {
        let url = installURL(for: adapter)
        if FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.removeItem(at: url)
        }
    }
}
