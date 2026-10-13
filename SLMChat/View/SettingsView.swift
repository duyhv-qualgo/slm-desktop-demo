import SwiftUI

/// User-selectable app appearance. `.system` follows the device setting; the
/// others force a scheme. Persisted in the store as the raw string and applied
/// at the app root via `.preferredColorScheme`.
enum AppearanceMode: String, CaseIterable, Identifiable {
    case system, light, dark

    var id: String { rawValue }

    var label: String {
        switch self {
        case .system: return "System"
        case .light:  return "Light"
        case .dark:   return "Dark"
        }
    }

    /// Scheme to force, or nil to follow the system (for `.preferredColorScheme`).
    var colorScheme: ColorScheme? {
        switch self {
        case .system: return nil
        case .light:  return .light
        case .dark:   return .dark
        }
    }
}

/// Settings sheet reached from the home-screen gear. Currently one section: the
/// on-device SLM the feature engine (translate / suggest / summarize / event /
/// guardrail) runs on. Tapping a model reloads the engine live via the store;
/// the active model shows a checkmark, the one being loaded shows a spinner, and
/// models whose `.litertlm` isn't bundled yet are disabled.
struct SettingsView: View {
    @ObservedObject var store: ConversationStore
    /// Observed separately: `loraStore` is a plain `let` on the store, so its
    /// download progress wouldn't reach this view through `store` alone.
    @ObservedObject private var lora: LoRAStore
    @Environment(\.dismiss) private var dismiss

    init(store: ConversationStore) {
        self.store = store
        _lora = ObservedObject(wrappedValue: store.loraStore)
    }

    /// Two-way binding over the store's private(set) native language: reads the
    /// published value, writes through `setNativeLanguage` (which persists +
    /// updates the shared service). Async because changing language reloads the
    /// task engine when the two languages have different LoRA packs installed.
    /// Diagnostic switch — see `ConversationStore.setPrefixReuse`. Takes effect on
    /// the next generation; no reload, so a measurement can be started and stopped
    /// without disturbing anything else.
    private var prefixReuseBinding: Binding<Bool> {
        Binding(get: { store.prefixReuseEnabled },
                set: { store.setPrefixReuse($0) })
    }

    private var languageBinding: Binding<TranslateLanguage> {
        Binding(get: { store.nativeLanguage },
                set: { lang in Task { await store.setNativeLanguage(lang) } })
    }

    /// Two-way binding over the store's private(set) appearance, writing through
    /// `setAppearance` (which persists + republishes the root color scheme).
    private var appearanceBinding: Binding<AppearanceMode> {
        Binding(get: { store.appearance },
                set: { store.setAppearance($0) })
    }

    /// Two-way binding over the store's private(set) auto-suggest flag, writing
    /// through `setAutoSuggest` (which persists + updates the shared service).
    private var autoSuggestBinding: Binding<Bool> {
        Binding(get: { store.autoSuggestEnabled },
                set: { store.setAutoSuggest($0) })
    }

    /// Two-way binding over the store's private(set) backend. Writing reloads both
    /// SLMs, so it kicks off an async switch rather than setting a flag.
    private var backendBinding: Binding<SLMBackendOption> {
        Binding(get: { store.backendOption },
                set: { option in Task { await store.setBackend(option) } })
    }

    /// Explains what the picker can and can't deliver on this build. The GPU path
    /// is a request, not a guarantee — saying so here saves a debugging session
    /// when the log keeps reporting `cpu`.
    private var backendFooter: String {
        var text = "Metal (GPU) mainly speeds up PREFILL — reading your prompt — "
                 + "which is most of the wait for a reply suggestion. Generation "
                 + "speed changes little. Switching reloads both models.\n\n"
                 + "GPU needs an A13 (iPhone 11) or later. Older GPUs lack the "
                 + "kernels llama.cpp uses for softmax and normalisation, so those "
                 + "layers fall back to the CPU mid-graph and end up slower — on "
                 + "those devices this stays on CPU whatever you pick. The full "
                 + "prefill speed-up needs an A14 (iPhone 12) or later."
        #if targetEnvironment(simulator)
        text += "\n\nThe simulator has no Metal compute device, so GPU is ignored "
              + "here and everything runs on CPU. Measure on a real device."
        #endif
        if store.service.backendDemotedToCPU {
            text += "\n\nThis device's Metal support couldn't allocate the model's "
                  + "buffers (see the console for the ggml error), so the models "
                  + "were loaded on CPU instead. Selecting GPU again retries."
        }
        if SLMBackend.gpuProbeTripped {
            text += "\n\nA previous GPU load crashed during Metal init, so Auto is "
                  + "pinned to CPU. Pick GPU explicitly to try again."
        }
        return text
    }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Picker("Appearance", selection: appearanceBinding) {
                        ForEach(AppearanceMode.allCases) { mode in
                            Text(mode.label).tag(mode)
                        }
                    }
                    .pickerStyle(.segmented)
                } header: {
                    Text("Appearance")
                } footer: {
                    Text("Choose Light or Dark, or follow the system setting.")
                }

                languageSection

                modelSection

                safetySection

                Section {
                    NavigationLink {
                        advancedSettings
                    } label: {
                        Label("Advanced", systemImage: "gearshape.2")
                    }
                } footer: {
                    Text("Compute backend, reply-suggestion behaviour, and "
                       + "measurement tools.")
                }
            }
            .formStyle(.grouped)
            .navigationTitle("Settings")
            .inlineNavigationTitle()
            // Packs can appear from outside this view — a file copied into
            // Documents, or a download finished while the sheet was closed — so
            // re-derive install state from disk whenever it's shown.
            .onAppear { lora.refresh() }
            .onChange(of: store.selectedModelID) { _ in lora.refresh() }
            #if os(iOS)
            .toolbar {
                ToolbarItem(placement: .trailingBar) {
                    Button("Done") { dismiss() }
                }
            }
            #endif
        }
    }

    // MARK: - On-device SLM

    /// Model as a dropdown, with the selected checkpoint's detail under it.
    ///
    /// Same move as `languageSection`, for the same reason: the list grew with the
    /// catalog, and every entry but one was there only to be read past.
    ///
    /// Models whose `.gguf` isn't in the bundle are excluded rather than listed
    /// and greyed — a `.menu` picker can't disable an individual item, and an
    /// entry you can pick that then refuses is worse than an absent one. The
    /// footer names them, so a missing model is still explained.
    ///
    /// Switching reloads ~300–600 MB of weights, so the picker locks while a
    /// switch is in flight rather than letting a second one queue behind it.
    @ViewBuilder
    private var modelSection: some View {
        Section {
            Picker("Model", selection: modelBinding) {
                ForEach(selectableModels) { model in
                    Text(model.displayName).tag(model.id)
                }
            }
            .pickerStyle(.menu)
            .disabled(store.pendingModelID != nil)

            if store.pendingModelID != nil {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text("Loading model…").foregroundStyle(.secondary)
                }
            } else if let model = store.availableModels
                        .first(where: { $0.id == store.selectedModelID }) {
                Text(model.isAvailable ? model.subtitle : "Not installed")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        } header: {
            Text("On-device SLM")
        } footer: {
            // Safety is deliberately NOT in this list: it runs on its own
            // BERT-Tiny TFLite classifier, not the SLM, and switching the
            // SLM has no effect on it. See `safetySection`.
            Text(modelFooter)
        }
    }

    /// Writes through `switchModel`, which persists the choice and reloads the
    /// engines. Keyed on `id` rather than the option so the picker's selection
    /// survives `availableModels` being rebuilt.
    private var modelBinding: Binding<String> {
        Binding(get: { store.selectedModelID },
                set: { id in
                    guard let model = store.availableModels.first(where: { $0.id == id }),
                          model.id != store.selectedModelID else { return }
                    Task { await store.switchModel(to: model) }
                })
    }

    /// Models that can actually be loaded, plus the current one regardless — if
    /// the selection were ever absent from the menu it is bound to, SwiftUI
    /// renders a blank row.
    private var selectableModels: [SLMModelOption] {
        let usable = store.availableModels.filter(\.isAvailable)
        return usable.contains(where: { $0.id == store.selectedModelID })
            ? usable
            : usable + store.availableModels.filter { $0.id == store.selectedModelID }
    }

    private var modelFooter: String {
        var text = "Powers translate, reply suggestions, summaries and event "
                 + "detection — all on-device via llama.cpp. Switching reloads the "
                 + "model, which takes a moment."
        let missing = store.availableModels.filter { !$0.isAvailable }.map(\.displayName)
        if !missing.isEmpty {
            text += "\n\nNot installed: " + missing.joined(separator: ", ")
                  + ". Add each .gguf to Resources/ to make it selectable."
        }
        return text
    }

    // MARK: - Advanced

    /// Everything that is a tuning knob rather than a preference.
    ///
    /// These four were top-level sections, which put a prompt-prefix cache toggle
    /// and a benchmark harness at the same visual weight as Appearance. They are
    /// not the same kind of thing: two exist for measurement and one reloads every
    /// model when touched. Ordered everyday-first — a reply-suggestion toggle is
    /// something a user might genuinely want; the diagnostics switch is something
    /// you go looking for.
    private var advancedSettings: some View {
        Form {
                Section {
                    Toggle("Auto-suggest replies", isOn: autoSuggestBinding)
                } header: {
                    Text("Reply suggestions")
                } footer: {
                    Text("Generate reply suggestions automatically when it's your "
                       + "turn in an open chat. Turn off to only suggest when you tap "
                       + "the suggest button. Runs on-device.")
                }

                Section {
                    Picker("Compute", selection: backendBinding) {
                        ForEach(SLMBackendOption.allCases) { option in
                            Text(option.label).tag(option)
                        }
                    }
                    .pickerStyle(.segmented)
                    .disabled(store.isSwitchingBackend)

                    if store.isSwitchingBackend {
                        HStack(spacing: 8) {
                            ProgressView().controlSize(.small)
                            Text("Reloading models…").foregroundStyle(.secondary)
                        }
                    } else if let active = store.backendActive {
                        LabeledContent("Running on", value: active.uppercased())
                    }
                } header: {
                    Text("Compute backend")
                } footer: {
                    Text(backendFooter)
                }

                Section {
                    Toggle("Reuse cached prompt prefix", isOn: prefixReuseBinding)
                } header: {
                    Text("Diagnostics")
                } footer: {
                    Text("On (default), a generation skips the leading prompt tokens "
                       + "an earlier call already computed — usually most of them. "
                       + "Turn OFF to measure raw prefill: every call then pays a full "
                       + "prompt pass, so the prefill tok/s in the logs becomes true "
                       + "throughput instead of a per-evaluated-token rate. Slower, and "
                       + "for measurement only.")
                }

                Section {
                    NavigationLink("Benchmark") { BenchmarkView(service: store.service) }
                } header: {
                    Text("Developer")
                } footer: {
                    Text("Measures TTFT and decode throughput of each bundled SLM on this "
                       + "device, and writes the runs to a CSV in Documents/.")
                }
        }
        .formStyle(.grouped)
        .navigationTitle("Advanced")
        .inlineNavigationTitle()
    }

    // MARK: - Language + packs

    /// Language as a dropdown, with the pack state for the CHOSEN language under it.
    ///
    /// This was a list of every language, each row carrying its own pack state and
    /// a Get/Remove button. That made the section as tall as the catalog, and the
    /// catalog is the part that grows.
    ///
    /// What the list was for is kept: a language whose pack isn't installed still
    /// works — it just runs on the base model's own multilingual ability — and
    /// nothing else in the app would tell you a better one is a tap away. So the
    /// pack row stays, scoped to the selection. What is lost is the ability to
    /// pre-fetch a pack for a language you haven't switched to yet, which was a
    /// niche affordance: switching to it later downloads it in place.
    ///
    /// Languages the selected checkpoint has no pack for are excluded from the
    /// menu rather than listed and disabled — a `.menu` picker gives no way to
    /// disable one item, and a pickable entry that silently refuses is worse than
    /// an absent one. The footer names them instead.
    @ViewBuilder
    private var languageSection: some View {
        Section {
            Picker("Language", selection: languageBinding) {
                ForEach(selectableLanguages) { language in
                    Text(language.name).tag(language)
                }
            }
            .pickerStyle(.menu)
            .disabled(store.isSwitchingLanguagePack)

            packRow

            if store.isSwitchingLanguagePack {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text("Reloading model…").foregroundStyle(.secondary)
                }
            }
        } header: {
            Text("Native language")
        } footer: {
            Text(languageFooter)
        }
    }

    /// Languages the selected checkpoint can actually run.
    ///
    /// Always includes the current selection, even if the chosen model doesn't
    /// support it — otherwise switching to such a model empties the picker of the
    /// value it is bound to, and SwiftUI renders a blank selection.
    private var selectableLanguages: [TranslateLanguage] {
        let supported = TranslateLanguage.allCases.filter {
            $0.isSupported(forBaseModel: store.selectedModelID)
        }
        return supported.contains(store.nativeLanguage)
            ? supported : supported + [store.nativeLanguage]
    }

    /// Pack state for the selected language, with its action.
    @ViewBuilder
    private var packRow: some View {
        let language = store.nativeLanguage
        if !language.isSupported(forBaseModel: store.selectedModelID) {
            LabeledContent("Pack") {
                Text("None for \(store.selectedModelName)").foregroundStyle(.secondary)
            }
        } else {
            let pack = LoRACatalog.adapter(baseModelID: store.selectedModelID,
                                           language: language)
            let state = pack.map { lora.state(of: $0) } ?? .notInstalled
            HStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Pack")
                    Text(packStatus(pack, state: state))
                        .font(.caption)
                        .foregroundStyle(state.isFailure ? .red : .secondary)
                }
                Spacer(minLength: 8)
                if let pack { packButton(pack, state: state) }
            }
        }
    }

    /// Whether message screening is actually running, and why not when it isn't.
    ///
    /// There is nothing to configure here — the guardrail has no user-facing
    /// setting — but there IS something to report. The classifier fails open, so a
    /// broken load produces a chat with no flags at all, which reads as "clean".
    /// This section is the one place that distinguishes the two.
    private var guardRegionBinding: Binding<GuardRegion> {
        Binding(get: { store.guardRegion },
                set: { region in Task { await store.setGuardRegion(region) } })
    }

    @ViewBuilder
    private var safetySection: some View {
        Section {
            Picker("Region", selection: guardRegionBinding) {
                ForEach(GuardRegion.allCases) { region in
                    Text(region.displayName).tag(region)
                }
            }

            if let reason = store.guardrailError {
                LabeledContent("Status") {
                    Text("Unavailable").foregroundStyle(.orange)
                }
                Text(reason)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else if store.isReady {
                LabeledContent("Status", value: "Active")
                // The "Rules" row is deliberately gone, along with the caption
                // that qualified it. It reported `guardRuleCount` as "199
                // patterns" — an implementation detail of one of the two
                // screening layers, stated in a unit nobody outside this codebase
                // can act on. A larger number is not more safety (the UK set is
                // 127 English patterns against Vietnam's 200 mostly-political
                // ones), so the row invited a comparison that means nothing, and
                // "not loaded" read as a fault when it is the normal state for a
                // region running on the baseline. Status covers the only thing a
                // user can act on: whether screening works.
                //
                // `guardRuleCount`/`guardRulesAuthored` stay on the store and are
                // now UI-less — the per-set counts still print at load
                // (`[GuardRules] <set>: N patterns (authored: yes/no)`), which is
                // where that detail is actually useful. Kept rather than deleted
                // because Diagnostics is the right home if it's ever wanted back.
                //
                // The "Languages" row is deliberately gone. It listed
                // `GuardPackCatalog.supportedLanguages` — Vietnamese, English —
                // which claimed per-language classifier coverage the app does not
                // actually have: the two `.tflite` packs are byte-identical (same
                // SHA-256), so there is ONE model listed under two names. The pack
                // machinery is the right shape for when a real English pack
                // arrives, but until then the row was advertising a distinction
                // that does not exist. Rules genuinely are per-language; the
                // classifier is not, and `Rules` already reports that.
            } else {
                LabeledContent("Status") {
                    Text("Loading…").foregroundStyle(.secondary)
                }
            }
        } header: {
            Text("Safety")
        } footer: {
            // Says WHERE screening happens, not how it is built. "On-device" is
            // the part that changes what a user should expect — nothing leaves
            // the phone — where "regex rules first, then a classifier" only
            // named the layers the row above no longer counts.
            Text(store.guardrailError == nil
                 ? "Messages are screened on this device before you see them. "
                 + "Nothing is sent anywhere."
                 : "")
        }
    }

    /// The pack action for a row: get it, cancel a download, or remove it.
    @ViewBuilder
    private func packButton(_ pack: LoRAAdapter, state: LoRAInstallState) -> some View {
        switch state {
        case .downloading(let progress):
            HStack(spacing: 8) {
                if let progress {
                    ProgressView(value: progress)
                        .progressViewStyle(.circular)
                        .controlSize(.small)
                } else {
                    ProgressView().controlSize(.small)
                }
                Button("Cancel") { lora.cancel(pack) }.buttonStyle(.borderless)
            }
        case .installed:
            // Only offered for packs we actually downloaded — a bundled pack has
            // no copy of its own to delete, so "Remove" would be a lie.
            if FileManager.default.fileExists(atPath: LoRACatalog.installURL(for: pack).path) {
                Button("Remove", role: .destructive) {
                    lora.remove(pack)
                    Task { await store.applyAdapterChange() }
                }
                .buttonStyle(.borderless)
            }
        case .notInstalled, .failed:
            Button("Get") { download(pack) }
                .buttonStyle(.borderless)
                .disabled(pack.remoteURL == nil && !pack.simulatedDownload)
        }
    }

    /// Kick off a download and apply the pack once it lands — but only if it's the
    /// language currently in use. Fetching a pack for a language you haven't
    /// switched to yet shouldn't cost a model reload; switching to it later will
    /// pick it up.
    private func download(_ pack: LoRAAdapter) {
        Task {
            await lora.downloadAndWait(pack)
            if case .installed = lora.state(of: pack),
               pack.languages.contains(store.nativeLanguage) {
                await store.applyAdapterChange()
            }
        }
    }

    /// Per-language pack status, written so that "no pack" reads as a normal
    /// configuration rather than something missing — for most languages the base
    /// model on its own is the right answer, and only a measured quality gap
    /// justifies a pack existing at all.
    private func packStatus(_ pack: LoRAAdapter?, state: LoRAInstallState) -> String {
        guard let pack else { return "No language pack — uses the base model" }
        switch state {
        case .failed(let message):
            return message
        case .downloading(let progress):
            guard let progress else { return "Downloading language pack…" }
            return "Downloading language pack… \(Int(progress * 100))%"
        case .installed:
            // The install itself is quick; applying it means reloading the model,
            // which on a cold cache is several seconds. Say that in the row —
            // otherwise the download hits 100%, the progress bar vanishes, and the
            // wait that follows looks like nothing is happening.
            if store.isSwitchingLanguagePack, pack.languages.contains(store.nativeLanguage) {
                return "Installed · applying to the model…"
            }
            return pack.languages.count > 1
                ? "Language pack installed · covers \(pack.languageList)"
                : "Language pack installed"
        case .notInstalled:
            if pack.remoteURL == nil && !pack.simulatedDownload {
                return "Pack not published — copy \(pack.fileName) into Documents/"
            }
            return "Language pack available · \(pack.displaySize) — tap Get for better replies"
        }
    }

    /// Explains the direction the features run in, and — when the language in use
    /// has an uninstalled pack — says so directly, since that's the case the user
    /// most needs to act on.
    private var languageFooter: String {
        var text = "A language pack is a small add-on that improves how the model "
                 + "writes in one language."

        if let pack = LoRACatalog.adapter(baseModelID: store.selectedModelID,
                                          language: store.nativeLanguage),
           !LoRACatalog.isInstalled(pack) {
            text += "\n\n\(store.nativeLanguage.name) has a language pack available "
                  + "that isn't installed. \(store.nativeLanguage.name) still works "
                  + "without it — the pack just makes the model noticeably better at it."
        }

        // Languages the picker cannot offer. Named here because they are absent
        // from the menu rather than greyed out inside it, so without this there is
        // nothing to explain why a language you expected is missing.
        let unavailable = TranslateLanguage.allCases
            .filter { !$0.isSupported(forBaseModel: store.selectedModelID) }
            .map(\.name)
        if !unavailable.isEmpty {
            text += "\n\nNot available with \(store.selectedModelName): "
                  + unavailable.joined(separator: ", ")
                  + ". Choose another model above to use them."
        }
        return text
    }
}
