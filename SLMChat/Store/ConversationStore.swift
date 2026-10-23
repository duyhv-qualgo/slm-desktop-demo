import Foundation

/// App-root object that owns the shared `ChatService` and every `Conversation`.
/// Long-lived for the whole app session, which is what keeps background replies
/// alive after the user leaves a chat (see `Conversation`).
@MainActor
final class ConversationStore: ObservableObject {

    @Published var conversations: [Conversation] = []

    /// Global model-load state, surfaced on the home screen.
    @Published var isReady: Bool = false
    @Published var statusText: String? = "Loading models…"
    @Published var errorMessage: String?

    /// Why the safety guardrail isn't screening messages, or nil when it is.
    /// Non-nil is not an app failure — the chat works fine — but it MUST be
    /// visible: with the classifier down every message reads Clean, so an absent
    /// flag would otherwise be read as "checked and safe" (see
    /// `ChatService.guardrailError`).
    @Published private(set) var guardrailError: String?

    /// Live regex pattern count; 0 means the guardrail is running model-only.
    @Published private(set) var guardRuleCount = 0

    /// False when the active region's rule set is only the universal baseline.
    @Published private(set) var guardRulesAuthored = true

    /// Jurisdiction whose rules screen messages, chosen in Settings. Defaults to
    /// the device's region on first launch, then persisted.
    @Published private(set) var guardRegion: GuardRegion

    /// Currently selected on-device feature SLM, and the row being loaded while a
    /// switch is in flight (drives the Settings sheet's checkmark / spinner).
    @Published private(set) var selectedModelID: String
    @Published private(set) var pendingModelID: String?

    /// The user's native language, chosen in Settings. Default Vietnamese, so the
    /// NLP features run native→English by default. Persisted across launches.
    @Published private(set) var nativeLanguage: TranslateLanguage

    /// App appearance (light / dark / follow system), chosen in Settings. Applied
    /// at the app root via `.preferredColorScheme`. Persisted across launches.
    @Published private(set) var appearance: AppearanceMode

    /// Whether reply suggestions auto-generate when it's your turn (vs. only on the
    /// suggest button). Chosen in Settings, default on. Persisted across launches.
    @Published private(set) var autoSuggestEnabled: Bool

    /// Compute backend for the on-device SLMs, chosen in Settings. Persisted.
    /// `backendActive` is what the engines actually ended up on — a GPU request
    /// is served on CPU where Metal isn't available (simulator), so the sheet
    /// reports the real answer rather than the wish.
    @Published private(set) var backendOption: SLMBackendOption
    @Published private(set) var backendActive: String?
    /// True while the engines are reloading for a backend change.
    @Published private(set) var isSwitchingBackend = false

    /// True while the task engine is reloading to pick up a different LoRA
    /// language pack — after a language change or a pack install/delete. Only a
    /// pack that actually differs triggers this; with none installed, switching
    /// language never reloads anything.
    @Published private(set) var isSwitchingLanguagePack = false

    /// Download/install state of the LoRA language packs, for the Settings rows.
    let loraStore = LoRAStore()

    /// The SLMs the user can switch between in Settings.
    let availableModels = DesktopModels.taskModels

    /// Language packs published for the currently selected checkpoint. Packs are
    /// per (model, language), so this list changes when the model does.
    var languagePacks: [LoRAAdapter] {
        LoRACatalog.adapters(forBaseModel: selectedModelID)
    }

    /// Display name of the currently selected on-device feature SLM — the model
    /// that runs translation / summarization / suggestions. Used to attribute
    /// on-device output (e.g. the summary sheet's footnote) to the chosen model.
    var selectedModelName: String { DesktopModels.taskModel(for: selectedModelID).displayName }

    private static let modelDefaultsKey = "selectedTaskModelID"
    private static let nativeLanguageKey = "nativeLanguageID"
    private static let appearanceKey = "appearanceMode"
    private static let autoSuggestKey = "autoSuggestEnabled"
    private static let backendKey = "slmBackendOption"
    private static let guardRegionKey = "guardRegion"

    let service = ChatService()

    /// The pinned group chat (user + 3 bots). Long-lived so its autonomous loop
    /// survives navigation. In-memory only — not persisted.
    lazy var group = GroupChat(service: service)

    /// Exact + semantic search over every message (on-device embeddings).
    lazy var search = MessageSearch(store: self)

    /// Predefined people you can start a new chat with. Persona and avatar derive
    /// from the name (`Prompts.persona(for:)`, `Conversation.initials`/`tint`).
    let roster = ["Alex", "Mai", "Sam", "Linh", "Minh", "Huy",
                  "Trang", "Nam", "Lan", "Quang"]

    init() {
        // Restore the persisted model choice, falling back to the default if the
        // saved model's file is no longer bundled, then hand it to the service so
        // the first warm-up loads the right checkpoint.
        let saved = UserDefaults.standard.string(forKey: Self.modelDefaultsKey)
        let resolved = DesktopModels.resolve(saved)
        selectedModelID = resolved.id
        service.taskModelResourceName = resolved.resourceName
        // The LoRA pack is keyed by (model, language), so the service needs the
        // catalog id too — not just the file basename.
        service.taskModelID = resolved.id

        // Restore the persisted native language (default Vietnamese) and hand it to
        // the service so the features run in the right direction from the first call.
        // A saved language the RESOLVED model can't serve falls back to the default
        // rather than launching into a language nothing can handle. That is a real
        // case now, not a defensive one: the extra languages come from the Qwen3 v6
        // packs, so quitting on Russian and relaunching after that checkpoint went
        // missing has to land somewhere sane.
        let savedLang = UserDefaults.standard.string(forKey: Self.nativeLanguageKey)
        let restored = savedLang.flatMap(TranslateLanguage.init(rawValue:)) ?? .vietnamese
        let language = restored.isSupported(forBaseModel: resolved.id) ? restored : .vietnamese
        nativeLanguage = language
        service.nativeLanguage = language

        // Restore the guardrail region (default: the device's own).
        let savedRegion = UserDefaults.standard.string(forKey: Self.guardRegionKey)
        // `restore` (not `init(rawValue:)`) so a value written under an older
        // spelling still resolves — see `GuardRegion.restore`.
        let region = savedRegion.flatMap(GuardRegion.restore(rawValue:)) ?? .deviceDefault
        guardRegion = region
        service.guardRegion = region

        // Restore the persisted appearance (default: follow the system).
        let savedAppearance = UserDefaults.standard.string(forKey: Self.appearanceKey)
        appearance = savedAppearance.flatMap(AppearanceMode.init(rawValue:)) ?? .system

        // Restore auto-suggest (default on if never set), and hand it to the service.
        let autoSuggest = UserDefaults.standard.object(forKey: Self.autoSuggestKey) as? Bool ?? true
        autoSuggestEnabled = autoSuggest
        service.autoSuggestEnabled = autoSuggest

        // Restore the compute backend (default: auto) BEFORE the first warm-up, so
        // the models load on the chosen backend instead of loading twice.
        let savedBackend = UserDefaults.standard.string(forKey: Self.backendKey)
        let backend = savedBackend.flatMap(SLMBackendOption.init(rawValue:)) ?? .auto
        backendOption = backend
        service.backendOption = backend

        // Restore the saved inbox if present; otherwise seed a fresh demo inbox.
        if let stored = ConversationPersistence.load() {
            conversations = stored.map(restore)
        } else {
            seed()
        }
        // Every conversation writes the whole inbox back to disk on any durable change.
        for c in conversations {
            c.onPersist = { [weak self] in self?.save() }
        }
    }

    // MARK: - Persistence

    /// Write the current inbox to disk. Called by each `Conversation` via its
    /// `onPersist` hook after a durable change.
    func save() {
        ConversationPersistence.save(conversations.map(\.snapshot))
        search.scheduleRefresh()
    }

    /// Remove a conversation from the inbox and persist the change.
    func delete(_ conversation: Conversation) {
        conversations.removeAll { $0.id == conversation.id }
        save()
    }

    /// Open the conversation with `name`, creating a new blank one if none exists
    /// (so picking an existing contact never duplicates the thread).
    func startConversation(with name: String) -> Conversation {
        if let existing = conversations.first(where: { $0.contactName == name }) {
            return existing
        }
        let convo = Conversation(contactName: name, messages: [], service: service)
        convo.onPersist = { [weak self] in self?.save() }
        conversations.append(convo)
        save()
        return convo
    }

    private func restore(_ stored: StoredConversation) -> Conversation {
        Conversation(contactName: stored.contactName,
                     messages: stored.messages.map(ChatMessage.init),
                     service: service)
    }

    /// Conversations ordered by most recent activity (like a real inbox).
    var sortedConversations: [Conversation] {
        conversations.sorted { $0.lastActivityDate > $1.lastActivityDate }
    }

    // MARK: - Lifecycle

    func warmUpIfNeeded() async {
        guard !isReady else { return }
        let res = await service.warmUpIfNeeded()
        // Record what the engines actually loaded on, so Settings can show it from
        // first launch rather than only after a manual switch.
        backendActive = service.activeBackend
        guardrailError = service.guardrailError
        guardRuleCount = service.guardRuleCount
        guardRulesAuthored = service.guardRulesAuthored
        if res.ok {
            isReady = true
            statusText = res.status
            // After the chat model, so search indexing never delays it.
            Task { await search.start() }
        } else {
            errorMessage = res.error
            statusText = nil
        }
    }

    /// Switch the on-device feature SLM. Reloads the task engine with the selected
    /// checkpoint; on success it becomes the persisted default. A no-op if the model
    /// is already active, unavailable, or another switch is in flight.
    func switchModel(to option: SLMModelOption) async {
        guard option.id != selectedModelID, pendingModelID == nil, option.isAvailable
        else { return }

        pendingModelID = option.id
        errorMessage = nil
        statusText = "Loading \(option.displayName)…"

        let res = await service.switchTaskModel(to: option)
        if res.ok {
            selectedModelID = option.id
            UserDefaults.standard.set(option.id, forKey: Self.modelDefaultsKey)
            isReady = true
            statusText = "\(option.displayName) ready."
            await demoteLanguageIfUnsupported(by: option)
        } else {
            errorMessage = "Couldn't load \(option.displayName): \(res.error ?? "load failed")"
            statusText = nil
        }
        pendingModelID = nil
    }

    /// Fall back to Vietnamese when the newly loaded checkpoint doesn't serve the
    /// language in use — e.g. leaving Qwen3 v6 (seven packs) for Gemma-3 (two)
    /// while set to Russian.
    ///
    /// Runs AFTER the model switch has succeeded, not before, at the cost of a
    /// second load in this one path. Demoting first would be cheaper but leaves
    /// the app lying whenever the switch then fails: the store would read
    /// "Vietnamese" while the still-loaded previous model kept its Russian adapter
    /// applied, and every feature would run under an adapter for a language the UI
    /// says isn't selected. Going through `setNativeLanguage` also means the
    /// reload, the persisted default and the status text are the same ones a
    /// manual language change produces.
    private func demoteLanguageIfUnsupported(by option: SLMModelOption) async {
        guard !nativeLanguage.isSupported(forBaseModel: option.id) else { return }
        let dropped = nativeLanguage
        await setNativeLanguage(.vietnamese)
        statusText = "\(option.displayName) has no \(dropped.name) pack — "
                   + "switched to Vietnamese."
    }

    /// Switch the compute backend and reload both SLMs on it. Unlike the other
    /// settings this is not a cheap flag flip — the models are released and
    /// reloaded, so the SLM features are unavailable for the duration and a
    /// failure has to reach the user (they're left with no model loaded).
    func setBackend(_ option: SLMBackendOption) async {
        guard option != backendOption, !isSwitchingBackend else { return }

        isSwitchingBackend = true
        errorMessage = nil
        statusText = "Reloading models on \(option.label)…"

        let res = await service.switchBackend(to: option)
        backendOption = option
        UserDefaults.standard.set(option.rawValue, forKey: Self.backendKey)
        backendActive = res.active
        if res.ok {
            isReady = true
            // Name the ACTIVE backend: asking for GPU on the simulator lands on CPU,
            // and a status line claiming otherwise would send you chasing ghosts.
            statusText = service.backendDemotedToCPU && option == .gpu
                ? "GPU unavailable on this device — running on CPU."
                : "Models ready on \(res.active.uppercased())."
        } else {
            isReady = false
            errorMessage = "Couldn't reload the models on \(option.label): "
                         + "\(res.error ?? "load failed"). Pick another backend."
            statusText = nil
        }
        isSwitchingBackend = false
    }

    /// Change the user's native language. Updates the shared service so the next
    /// translate / suggest / summarize call runs in the new direction, and persists
    /// the choice. Cached translations on existing messages are left as-is.
    ///
    /// When the two languages have DIFFERENT LoRA packs installed this also
    /// reloads the task engine, because an adapter can only be changed by
    /// reloading (see `ChatService.switchNativeLanguage`). That path costs a model
    /// load, so it is entered only when the pack actually differs — with no packs
    /// installed, which is the default, this stays the cheap flag flip it was.
    func setNativeLanguage(_ language: TranslateLanguage) async {
        // Settings won't offer an unsupported language, but the guard belongs here
        // too: selecting one would run every feature against a language the
        // finetune has never seen, and fail as bad output rather than an error.
        guard language.isSupported(forBaseModel: selectedModelID),
              language != nativeLanguage else { return }
        let willReload = service.languageChangeNeedsReload(to: language)
        nativeLanguage = language
        UserDefaults.standard.set(language.rawValue, forKey: Self.nativeLanguageKey)

        if willReload {
            isSwitchingLanguagePack = true
            statusText = "Loading \(language.name) language pack…"
        }
        let res = await service.switchNativeLanguage(to: language)
        if willReload {
            isSwitchingLanguagePack = false
            if res.ok {
                statusText = "\(language.name) language pack ready."
            } else {
                // The engine keeps the previously loaded model on a failed load,
                // so the features still work — just without the new pack.
                errorMessage = "Couldn't load the \(language.name) language pack: "
                             + "\(res.error ?? "load failed"). Using the base model."
                statusText = nil
            }
        }
    }

    /// Reload the task engine so a pack installed or deleted from Settings takes
    /// effect. Safe to call when nothing changed — the service checks first.
    func applyAdapterChange() async {
        guard service.needsReloadForAdapterChange, !isSwitchingLanguagePack else { return }
        isSwitchingLanguagePack = true
        statusText = "Applying language pack…"
        let res = await service.reloadTaskEngineForAdapterChange()
        isSwitchingLanguagePack = false
        if res.ok {
            statusText = "Language pack applied."
        } else {
            errorMessage = "Couldn't apply the language pack: \(res.error ?? "load failed")."
            statusText = nil
        }
    }

    /// Change the jurisdiction whose rules screen messages, and persist it.
    ///
    /// Only the rule set reloads — the classifier packs are per-language and say
    /// the same thing in every region. Already-classified messages keep their
    /// stored verdicts: classification is once-per-message by design, so a region
    /// change applies going forward rather than re-screening history.
    func setGuardRegion(_ region: GuardRegion) async {
        guard region != guardRegion else { return }
        guardRegion = region
        UserDefaults.standard.set(region.rawValue, forKey: Self.guardRegionKey)
        guardRuleCount = await service.switchGuardRegion(to: region)
        guardRulesAuthored = service.guardRulesAuthored
    }

    /// Change the app appearance. Republishes so the app root re-applies the
    /// preferred color scheme immediately, and persists the choice.
    func setAppearance(_ mode: AppearanceMode) {
        guard mode != appearance else { return }
        appearance = mode
        UserDefaults.standard.set(mode.rawValue, forKey: Self.appearanceKey)
    }

    /// Toggle auto-suggest, push it to the shared service so open chats pick it up
    /// immediately, and persist the choice.
    func setAutoSuggest(_ enabled: Bool) {
        guard enabled != autoSuggestEnabled else { return }
        autoSuggestEnabled = enabled
        service.autoSuggestEnabled = enabled
        UserDefaults.standard.set(enabled, forKey: Self.autoSuggestKey)
    }

    // MARK: - Image search

    /// One matched image in an image search — the message, its thread, and which of
    /// its tags matched (for display).
    struct ImageSearchHit: Identifiable {
        let id: UUID
        let conversation: Conversation
        let message: ChatMessage
        let matchedTags: [String]
        let score: Int
    }

    /// True if any image in the inbox has been tagged — drives the search screen's
    /// empty state ("attach a photo first" vs "no matches").
    var hasTaggedImages: Bool {
        conversations.contains { convo in
            convo.messages.contains { ($0.imageTags?.isEmpty == false) }
        }
    }

    /// Search chat-attached images by their on-device tags. A Vietnamese query is
    /// first translated to English (the tag vocabulary is English-only) via the task
    /// SLM, then matched lexically against each image's cached tags. Results are
    /// ranked so images whose highest-confidence tags matched come first. Pass a
    /// `scope` conversation to search only that thread's media (the contact-info
    /// screen); omit it to search the whole inbox.
    func searchImages(_ query: String, in scope: Conversation? = nil) async -> [ImageSearchHit] {
        let english = await englishQuery(for: query)
        let terms = english.lowercased()
            .split { !$0.isLetter && !$0.isNumber }
            .map(String.init)
            .filter { $0.count >= 2 }
        guard !terms.isEmpty else { return [] }

        var hits: [ImageSearchHit] = []
        for convo in (scope.map { [$0] } ?? conversations) {
            for m in convo.messages where m.hasImage {
                guard let tags = m.imageTags, !tags.isEmpty else { continue }
                var matched: [String] = []
                var score = 0
                for (rank, tag) in tags.enumerated() {
                    let words = tag.lowercased().split(separator: " ").map(String.init)
                    let isMatch = terms.contains { term in
                        words.contains { Self.wordMatches(term: term, word: $0) }
                    }
                    if isMatch {
                        matched.append(tag)
                        score += max(1, tags.count - rank)   // top tags weigh more
                    }
                }
                if !matched.isEmpty {
                    hits.append(ImageSearchHit(id: m.id, conversation: convo,
                                               message: m, matchedTags: matched, score: score))
                }
            }
        }
        return hits.sorted { $0.score > $1.score }
    }

    /// Loose lexical match between a query term and one tag word: exact, or a
    /// length-3+ prefix in either direction (tolerates plurals / partial typing).
    private static func wordMatches(term: String, word: String) -> Bool {
        if term == word { return true }
        guard term.count >= 3, word.count >= 3 else { return false }
        return word.hasPrefix(term) || term.hasPrefix(word)
    }

    /// Translate a non-English query to English (image tags are English-only).
    /// Returns the query unchanged when it already looks English, can't be
    /// identified, or translation fails.
    ///
    /// Was Vietnamese-only, keyed on `vietnameseRatio >= 0.3`. That heuristic can
    /// recognise exactly one language to translate FROM, so a Chinese, Russian or
    /// Spanish query scored 0, was passed through untouched, and matched nothing —
    /// offline photo search silently didn't work for six of the seven languages.
    private func englishQuery(for query: String) async -> String {
        guard let source = Prompts.sourceLanguage(of: query), source != .english else {
            return query
        }
        let prompt = Prompts.translate(query, from: source, to: .english)
        let raw = await service.taskEngine.complete(system: nil, history: [], user: prompt,
                                                    maxTokens: service.translateTokens,
                                                    temperature: service.translateTemp,
                                                    label: .translateQuery)
        return raw.map(Prompts.parseTranslation) ?? query
    }

    // MARK: - Seed data

    /// A few contacts with distinct personas. The trip-planning thread starts with
    /// unread friend messages so the unread badge + Summarize demo work from a cold
    /// start; the others open already read.
    private func seed() {
        let now = Date()

        let alex = Conversation(
            contactName: "Alex",
            messages: [
                ChatMessage(sender: .friend,
                            text: "Hey! Are we still on for the trip this weekend?",
                            date: now.addingTimeInterval(-600),
                            isRead: false),
                ChatMessage(sender: .friend,
                            text: "I found a cheap hostel near the beach, and the bus leaves Saturday at 8am.",
                            date: now.addingTimeInterval(-580),
                            isRead: false),
            ],
            service: service)

        let mai = Conversation(
            contactName: "Mai",
            messages: [
                ChatMessage(sender: .you,
                            text: "That phở place was amazing, thanks for the rec!",
                            date: now.addingTimeInterval(-3600 * 5)),
                ChatMessage(sender: .friend,
                            text: "Right?? Told you it was the best in town.",
                            date: now.addingTimeInterval(-3600 * 5 + 40)),
            ],
            service: service)

        let sam = Conversation(
            contactName: "Sam",
            messages: [
                ChatMessage(sender: .friend,
                            text: "Pushed the fix, can you review the PR when you get a sec?",
                            date: now.addingTimeInterval(-3600 * 26)),
                ChatMessage(sender: .you,
                            text: "On it, will take a look shortly.",
                            date: now.addingTimeInterval(-3600 * 26 + 120)),
            ],
            service: service)

        conversations = [alex, mai, sam]
    }
}
