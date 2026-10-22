import Foundation

/// Which language pack a reply suggestion should be generated under — see
/// `ChatService.suggestAdapter(for:)`.
enum SuggestAdapter: Equatable {
    /// Generate under this pack: it serves the conversation's language and is
    /// installed.
    case pack(String)
    /// Generate under whatever the engine already loaded. Either the conversation
    /// is in the native language (so the loaded pack IS the right one), or no pack
    /// for this pair was ever published and bare-ish output is the best available.
    case loaded
    /// A pack for this language exists but the user hasn't installed it. NOT a
    /// generation input: the caller must stop and surface this instead, because
    /// the alternative — generating under the native pack — is the wrong-language
    /// mush the per-conversation adapter exists to prevent.
    case missing(LoRAAdapter)
}

/// Shared model plumbing for the whole app: the two on-device engines, the cloud
/// chat backend, and the one-time warm-up. It owns NO conversation state — every
/// `Conversation` calls into the same `ChatService`, so the models load once and
/// are shared across all chats (RAM stays flat regardless of chat count).
///
/// This is the non-UI half of what used to be `ChatViewModel`; per-chat state and
/// orchestration now live on `Conversation`.
final class ChatService {

    /// Task finetune for the three features (translate / suggest / summarize).
    let taskEngine = SLMEngine(namespace: "task")

    /// On-device text-guardrail classifier (BERT-Tiny via TFLite). Replaces the
    /// old LLM-prompt safety path — see `Conversation.classifySafety`.
    let guardEngine = GuardEngine()

    /// On-device image-tagging classifier (EdgeNeXt via TFLite). Tags photos the
    /// user attaches to a chat so they're searchable offline — see
    /// `Conversation.tagImage` and `ConversationStore.searchImages`.
    let imageTagEngine = ImageTagEngine()

    // The classifier and tokenizer are per-language now — see `GuardPackCatalog`.
    // The regex rules are per-REGION — see `GuardRegion`.

    /// Jurisdiction whose rules screen this user. Defaults to the device's own
    /// region; the store overwrites it from the persisted choice before warm-up.
    var guardRegion: GuardRegion = .deviceDefault

    /// How many rule patterns are live (0 = model-only). Surfaced in Settings.
    private(set) var guardRuleCount = 0

    /// False when the active region ships only the universal baseline.
    private(set) var guardRulesAuthored = true

    /// Why the guardrail isn't running, or nil when it loaded fine.
    ///
    /// A failed guardrail load is non-fatal — `classify` fails open to `.clean` —
    /// which is exactly what makes it dangerous to leave silent: every message
    /// then reads Clean, and that is indistinguishable from content genuinely
    /// being safe. Recording the reason here lets the UI say "safety checks are
    /// off" instead of implying "nothing was flagged".
    private(set) var guardrailError: String?

    /// Bundle basenames for the image-tagging assets (dropped into Resources/).
    static let imageTagModelResource = "image-tagging"
    static let imageTagListResource = "image-tag-list"

    /// Cloud chat backend for the friend's replies — the ONLY way the simulated
    /// chat partner speaks. Non-nil when `Resources/APIConfig.plist` is present.
    ///
    /// There used to be an on-device fallback (`chatEngine`, a second ~500 MB
    /// `chat-qwen.gguf`) for when the plist was absent. It's gone: this project
    /// always runs the API, the file was never in the repo, and every branch that
    /// reached that engine was therefore dead code that no one could exercise.
    /// Restoring it means restoring one `SLMEngine` and the else-branches in
    /// `Conversation.replyFromFriend` / `seedBacklog` / `GroupChat.botReply` —
    /// see git history.
    ///
    /// Nil is NOT fatal. Everything the demo is actually about — translate,
    /// suggest, summarize, event, guardrail, image tagging — is on-device and
    /// unaffected; only the friend's replies stop working, and they say so.
    let chatAPI: ChatAPI? = ChatAPIConfig.load().map(ChatAPI.init)

    var usesAPI: Bool { chatAPI != nil }

    /// What to show the user when a friend reply is attempted with no API
    /// configured. Names the file, because that is the entire fix.
    static let missingAPIMessage =
        "No chat backend configured — copy APIConfig.example.plist to "
      + "Resources/APIConfig.plist and add your key. On-device features "
      + "(translate, suggest, summarize) work without it."

    /// Bundle basename of the on-device SLM the feature engine loads. Swappable at
    /// runtime from the Settings sheet; defaults to the gemma-3 270m finetune.
    /// The store sets this from the persisted selection before the first warm-up.
    var taskModelResourceName = ModelCatalog.defaultTaskModel.resourceName

    /// Catalog id of that same checkpoint. Kept alongside the resource name
    /// because a LoRA pack is identified by (model, language) — an adapter is
    /// bound to the exact weights it was trained on, so the model half of the key
    /// has to travel with the load. See `LoRACatalog`.
    var taskModelID = ModelCatalog.defaultTaskModel.id {
        didSet { taskEngine.setPrefixReuse(prefixReuseActive) }
    }

    /// The user's native language, chosen in Settings (default Vietnamese). Drives
    /// the default direction of the NLP features — translation goes native→English,
    /// and suggestions/summaries default to this language. The store sets it from
    /// the persisted selection; conversations read it when building prompts.
    var nativeLanguage: TranslateLanguage = .vietnamese

    /// Compute backend for the two SLM engines, chosen in Settings. Changing it
    /// requires reloading the models (it's a load-time property), which is what
    /// `switchBackend` does. The store sets it from the persisted selection before
    /// the first warm-up.
    var backendOption: SLMBackendOption = .auto

    /// Set once a GPU load has failed on this device, pinning every later load to
    /// CPU for the rest of the session. Cleared when the user picks a backend in
    /// Settings, so an explicit GPU choice always gets a fresh attempt.
    private(set) var backendDemotedToCPU = false

    /// The backend string handed to `SLMEngine.warmUp` for app loads.
    private var backend: String {
        backendDemotedToCPU ? SLMBackend.cpu : SLMBackend.preferred(backendOption)
    }

    /// Load one engine, falling back to CPU when a GPU load fails.
    ///
    /// A Metal load can fail for reasons the app can't detect in advance — e.g. a
    /// device whose Metal implementation lacks residency sets, where ggml can't
    /// allocate its buffers and the load simply returns nil. Without this fallback
    /// the app would sit there with NO model loaded and every feature dead, which
    /// is a far worse outcome than running the same model a bit slower on CPU.
    ///
    /// The demotion is sticky for the session so the remaining engines don't each
    /// repeat the doomed GPU attempt. On failure the ORIGINAL (GPU) error is
    /// reported — it's the one that explains what went wrong.
    private func load(_ engine: SLMEngine, model: String,
                      adapterPath: String? = nil) async -> (ok: Bool, error: String?) {
        let requested = backend
        let res = await engine.warmUp(modelName: model, backend: requested,
                                      adapterPath: adapterPath)
        guard !res.ok, requested == SLMBackend.gpu else { return res }

        NSLog("[SLMPerf] GPU load failed for %@ (%@) — falling back to CPU",
              model, res.error ?? "unknown")
        backendDemotedToCPU = true
        let fallback = await engine.warmUp(modelName: model, backend: SLMBackend.cpu,
                                           adapterPath: adapterPath)
        return fallback.ok ? fallback : res
    }

    /// LoRA pack for the current (task model, native language) pair, or nil to run
    /// the task engine on bare weights.
    ///
    /// nil is a normal, shipping outcome — not a failure. Only languages the base
    /// handles poorly need a pack, so most pairs resolve to nil and simply use the
    /// model as trained.
    /// Logs which of the two nil cases applies, because they mean different things
    /// and the load line can only report `lora=none` for both: no pack exists for
    /// this pair (expected — bare weights are correct), versus a pack exists but
    /// isn't installed yet (the user has a download to do).
    private var taskAdapterPath: String? {
        guard let pack = LoRACatalog.adapter(baseModelID: taskModelID,
                                             language: nativeLanguage) else {
            NSLog("[LoRA] no pack published for (%@, %@) — using bare weights",
                  taskModelID, nativeLanguage.rawValue)
            return nil
        }
        guard let path = LoRACatalog.installedPath(for: pack) else {
            NSLog("[LoRA] pack %@ not installed — using bare weights", pack.fileName)
            return nil
        }
        return path
    }

    /// LoRA pack to generate auto-reply suggestions with: the one for the language
    /// the CONVERSATION is being held in, not the one for the user's own language.
    ///
    /// This is the one feature where the two differ on purpose. A suggestion is
    /// text the user is about to send into that conversation, so it has to be
    /// written in its language and by a model adapted to it — a Vietnamese-adapted
    /// model asked for Spanish produces grammatical mush ("Tú bajar la pelota y
    /// hacer tiempo…"), which is a chip nobody can tap. Translation, summaries and
    /// event extraction go the other way: their output is for the user to READ, so
    /// they stay on `nativeLanguage` and on the pack that serves it.
    ///
    /// The three outcomes are NOT interchangeable, which is why this isn't a
    /// `String?`. `.loaded` and `.missing` would both be nil, and nil means "use
    /// whatever the engine loaded" — so an uninstalled pack would silently generate
    /// under the NATIVE one, i.e. exactly the mush described above, with the perf
    /// line reporting `lora=…vietnamese` on a Spanish chat. A pack the user simply
    /// hasn't installed is a fixable state, so it is raised rather than papered
    /// over; a pack that was never published isn't, so that case runs.
    ///
    /// Every outcome is logged: downstream they are indistinguishable, and they
    /// mean different things (a detection bug / a catalog gap / a download the user
    /// hasn't done).
    func suggestAdapter(for language: TranslateLanguage) -> SuggestAdapter {
        guard language != nativeLanguage else {
            NSLog("[LoRA] suggest lang=%@ → loaded pack (conversation is in the native language)",
                  language.rawValue)
            return .loaded
        }
        guard let pack = LoRACatalog.adapter(baseModelID: taskModelID, language: language) else {
            // Nothing to install — this pair has no pack and never will until one
            // is trained, so running under the loaded pack is the only option that
            // produces anything at all.
            NSLog("[LoRA] suggest lang=%@ → loaded pack (no %@ pack published for %@)",
                  language.rawValue, language.rawValue, taskModelID)
            return .loaded
        }
        guard let path = LoRACatalog.installedPath(for: pack) else {
            NSLog("[LoRA] suggest lang=%@ → BLOCKED (%@ not installed)",
                  language.rawValue, pack.fileName)
            return .missing(pack)
        }
        NSLog("[LoRA] suggest lang=%@ → %@", language.rawValue, pack.fileName)
        return .pack(path)
    }

    /// What to tell the user when suggestions are blocked on an uninstalled pack.
    /// Names the language and points at the one place that fixes it.
    static func missingPackNotice(_ pack: LoRAAdapter,
                                  language: TranslateLanguage) -> String {
        "\(language.name) reply suggestions need the \(pack.displayName) — "
      + "install it in Settings › Language packs."
    }

    /// Path of the language pack actually applied to the task engine, or nil for
    /// bare weights. Differs from `taskAdapterPath` when a pack was installed or
    /// removed since the last load — Settings uses that gap to offer a reload.
    var activeTaskAdapterPath: String? { taskEngine.activeLoRAPath }

    /// True when the pack on disk no longer matches the one in the loaded engine,
    /// i.e. the user has just downloaded or deleted one. Nothing breaks in this
    /// state; the change simply hasn't taken effect yet.
    var needsReloadForAdapterChange: Bool {
        taskAdapterPath != taskEngine.activeLoRAPath
    }

    /// Whether switching to `language` would force a model reload — i.e. the two
    /// languages resolve to different LoRA packs. Lets the UI show a spinner
    /// before the reload starts, and stay silent when there's nothing to reload.
    func languageChangeNeedsReload(to language: TranslateLanguage) -> Bool {
        guard language != nativeLanguage else { return false }
        return LoRACatalog.installedPath(baseModelID: taskModelID,
                                         language: language) != taskAdapterPath
    }

    /// What the task engine is ACTUALLY running on after its load — a `"gpu"`
    /// request lands on CPU where Metal isn't available. nil before any load.
    var activeBackend: String? { taskEngine.activeBackend }

    /// Whether the task engine may reuse cached prompt prefixes. Off is a
    /// measurement mode — see `SLMEngine.setPrefixReuse`. The store sets it from
    /// the persisted selection before the first warm-up.
    var prefixReuseEnabled: Bool = true {
        didSet { taskEngine.setPrefixReuse(prefixReuseActive) }
    }

    /// Whether prefix reuse actually runs: off for Qwen3.5 whatever the setting.
    /// It is hybrid attention/SSM (`qwen35`), and llama.cpp can't roll its SSM
    /// state back to a token position, so a partial cache removal
    /// is always refused and the slot gets cleared anyway — trying is wasted work.
    /// Gate reuse-only work (the suggest pre-warm) on this, not on the setting.
    var prefixReuseActive: Bool {
        if taskModelID == "qwen3.5-0.8b" {
            return false
        } else {
            return prefixReuseEnabled
        }
    }

    /// When true, reply suggestions are generated automatically (gated + debounced)
    /// once it's the user's turn in the open chat, instead of only on a button tap.
    /// The store sets it from the persisted selection.
    var autoSuggestEnabled: Bool = true

    // Per-task generation budgets / sampling for the ON-DEVICE features.
    //
    // Friend replies aren't here: they run on the API, whose sampling lives in
    // `ChatAPI` (temperature is set in its request body). `replyTokens` /
    // `replyTemp` used to sit at the top of this list and were the natural thing
    // to reach for when tuning Alex — they governed nothing.
    let suggestTokens: Int32 = 160   // one JSON object now carries all candidates
    let suggestTemp: Float = 0.8
    let summaryTokens: Int32 = 96
    // Budget for ONE map chunk or one intermediate fold of a long message.
    //
    // Deliberately larger than `summaryTokens`, which is sized for a conversation
    // summary (measured output 55-88 tokens). A map chunk summarises up to
    // `Prompts.longChunkChars` (600) characters of arbitrary prose, and at ~2.9
    // chars/token for Vietnamese that does not fit in 96 — measured on device, a
    // chunk wanted 358 characters of output and stopped at `max_tokens`, leaving a
    // partial summary ending mid-sentence.
    //
    // That failure is uniquely nasty in a map-reduce: the truncated partial is
    // INPUT to the fold, so the final summary faithfully summarises damaged text
    // and reports `stop=eos` — the visible symptom appears one stage away from the
    // cause. Intermediate folds get the same budget for the same reason.
    let summaryMapTokens: Int32 = 192
    // The FINAL long-message summary gets a larger output budget than a
    // conversation summary — a per-message summary that's cut off at 96 tokens
    // reads as truncated. Prompt (bounded to a reduce chunk) + this stays well
    // under the model's ~1024-token context. See `summarizeLong`.
    let summaryLongTokens: Int32 = 256

    /// Prompt scaffolding around a chunk: the chat template plus the instruction
    /// line, plus margin. Measured — the benchmark prompt goes 31 -> 43 tokens
    /// through the template (~12), and "Summarize the following conversation in
    /// Vietnamese:" is ~12 more. The rest is headroom, because overrunning the
    /// context fails a generation outright.
    private static let promptOverheadTokens = 48

    /// Tokens of SOURCE TEXT one map chunk may carry.
    ///
    /// Derived, never hardcoded: the slot holds `contextTokensPerSlot`, and the
    /// prompt and everything generated from it must both fit. Writing a literal
    /// here would silently overrun the moment someone raised `summaryMapTokens` or
    /// lowered the context — the two numbers are not independent.
    ///
    ///     1024 context - 192 output - 48 overhead = 784 tokens of text
    ///
    /// (The raw arithmetic allows ~800; the overhead figure is what buys the
    /// margin that keeps a long generation from running into the ceiling.)
    var summaryChunkTokens: Int {
        max(64, Int(ChatLM.contextTokensPerSlot()) - Int(summaryMapTokens)
                - Self.promptOverheadTokens)
    }

    /// Same budget for a REDUCE fold, which folds partial summaries rather than
    /// source text but shares the context and the same output budget.
    var summaryFoldTokens: Int { summaryChunkTokens }

    /// Chunk `text` with the loaded model's tokenizer, falling back to the
    /// character heuristic only when nothing is loaded (tests, pre-warm-up).
    func chunkForSummary(_ text: String, maxTokens: Int) -> [String] {
        guard taskEngine.tokenCount(of: "probe") != nil else {
            NSLog("[Summarize] no model loaded — chunking by characters (estimate)")
            return Prompts.chunk(text)
        }
        return Prompts.chunk(text, maxTokens: maxTokens) { [taskEngine] piece in
            taskEngine.tokenCount(of: piece) ?? Prompts.estimatedTokens(piece)
        }
    }
    let summaryTemp: Float = 0.3
    let translateTokens: Int32 = 160
    // Translation budget (see `translateLongStream`). One call is one 1024-token
    // slot, shared by the prompt and the translation it generates — overrunning it
    // fails the generation outright. A translation can be LONGER than its source:
    // English → Vietnamese measured 1.27x (Qwen3.5) and 1.42x (Qwen2.5) the
    // tokens, plus ~12 for the JSON the finetune wraps it in. So output gets the
    // bigger share:
    //
    //     48 prompt overhead + 390 source + 580 output (≥ 1.42 × 390 + 24) = 1018
    //
    // The old split (720 source, 256 output) cut off any English text over ~180
    // tokens before its last sentence.
    static let translatePromptOverheadTokens = 48
    let translateChunkInputTokens = 390
    let translateChunkTokens: Int32 = 580
    // Hard maximum for one message: two chunks. Chunks run one after another on
    // the shared task model (~6.5 s each on Metal), so a longer text would hold it
    // — and every other feature — for minutes; it is refused instead.
    var translateMaxInputTokens: Int { 2 * translateChunkInputTokens }

    /// Source tokens of `text` by the loaded model's tokenizer, or the character
    /// estimate when nothing is loaded yet.
    private func translationTokens(_ text: String) -> Int {
        taskEngine.tokenCount(of: text) ?? Prompts.estimatedTokens(text)
    }

    /// Shown in place of a translation for a message over `translateMaxInputTokens`
    /// (~560 English words). Persisted like any translation; Retranslate re-checks.
    static let tooLongToTranslateNote = "(too long to translate on device)"

    /// Over `translateMaxInputTokens`: the UI shows a "too long" note instead of
    /// translating, and `translateLongStream` yields nothing.
    func isTooLongToTranslate(_ text: String) -> Bool {
        translationTokens(text.trimmingCharacters(in: .whitespacesAndNewlines)) > translateMaxInputTokens
    }

    /// Split at sentence (then clause, then word) boundaries into pieces of at most
    /// `translateChunkInputTokens` — a short message is one piece.
    func chunkForTranslation(_ text: String) -> [String] {
        chunkForSummary(text, maxTokens: translateChunkInputTokens)
    }
    let translateTemp: Float = 0.2
    let eventTokens: Int32 = 96
    let eventTemp: Float = 0.1   // near-deterministic structured extraction

    // Magic Compose (rewrite the draft / generate a reply from a chip).
    //
    // Sized against `suggestTokens`, not below it: a suggest call returns a JSON
    // object of three SHORT chips inside 160, while this returns ONE message that
    // the user is about to send — and the Elaborate chip's entire job is to make
    // that message longer than the draft. Cutting it off mid-sentence is worse
    // here than anywhere else in the app, because the truncated text lands in the
    // input box looking like something ready to send.
    let composeTokens: Int32 = 200
    // Lower than `suggestTemp` (0.8) and for the opposite reason. Suggest wants
    // three visibly DIFFERENT chips from one generation, so it pays for variety.
    // Compose returns a single rewrite of the user's own words, where variety
    // reads as the model wandering off the draft — but it is not extraction
    // either, so it stays above the translate/event floor. Retry re-rolls, which
    // is what needs some randomness left in it.
    let composeTemp: Float = 0.5

    private var didWarmUp = false

    /// Load the models once. The task finetune always loads on-device; the chat
    /// model loads only when there's no API configured (saves ~500 MB RAM).
    /// Idempotent — safe to call from the store on every appearance.
    func warmUpIfNeeded() async -> (ok: Bool, status: String?, error: String?) {
        guard !didWarmUp else { return (true, nil, nil) }

        let taskRes = await load(taskEngine, model: taskModelResourceName,
                                 adapterPath: taskAdapterPath)

        // Guardrail is non-fatal: if it fails to load, `classify` fails open to
        // .clean, so the app still runs (just without safety flags). The reason is
        // kept on `guardrailError` so the UI can say so — failing open silently
        // would make "safety is broken" look identical to "everything is clean".
        // Preload the pack for the user's own language; others load on first use.
        MemoryTracker.mark("before guard warm-up")
        let guardRes = await guardEngine.warmUp(language: nativeLanguage,
                                                region: guardRegion)
        MemoryTracker.mark("after guard warm-up")
        guardrailError = guardRes.ok ? nil : (guardRes.error ?? "unknown")
        guardRuleCount = guardEngine.ruleCount
        guardRulesAuthored = guardEngine.rulesAuthored
        if !guardRes.ok {
            NSLog("[GuardEngine] warm-up failed: %@", guardRes.error ?? "unknown")
        }

        // Image tagging is likewise non-fatal: if it fails to load, `tagImage`
        // returns [] and attached photos simply aren't searchable.
        let imageRes = await imageTagEngine.warmUp(modelName: Self.imageTagModelResource,
                                                   tagListName: Self.imageTagListResource)
        if !imageRes.ok {
            NSLog("[ImageTagEngine] warm-up failed: %@", imageRes.error ?? "unknown")
        }

        guard taskRes.ok else {
            return (false, nil, "Failed to load task model: \(taskRes.error ?? "load failed")")
        }
        didWarmUp = true
        // A missing API config is reported in the status line, not as a failure:
        // the task model loaded, so every on-device feature works. Only the
        // simulated friend is mute, and `replyFromFriend` says so when tried.
        return (true, usesAPI ? "Features on-device · friends via API."
                              : "Features on-device · no chat backend configured.", nil)
    }

    // MARK: - Benchmark isolation

    /// Release the SLM engine so a benchmark can measure a model's memory cost
    /// against a clean baseline. Without this the app's `taskEngine` already has
    /// one benchmark checkpoint mapped, and a second mapping of the same file
    /// shares physical pages — so that one model looks nearly free while the
    /// others pay full price. The TFLite classifiers stay loaded (tens of MB, no
    /// unload path). **The app's SLM features are dead until `restoreSLMs()` runs
    /// — callers must pair the two.**
    func releaseSLMs() async {
        await taskEngine.unload()
    }

    /// Reload what `releaseSLMs()` unloaded. Returns false if anything failed to
    /// come back — the caller must surface that rather than swallow it.
    func restoreSLMs() async -> Bool {
        await load(taskEngine, model: taskModelResourceName,
                   adapterPath: taskAdapterPath).ok
    }

    /// Reload the on-device feature engine with a different SLM checkpoint. The load
    /// is serialized on the task engine's own queue, so any in-flight generation
    /// finishes first. On failure the previously loaded model stays active (the
    /// engine only swaps its bridge on a successful load), so a bad/ missing file
    /// can't leave the features dead — the caller just surfaces the error.
    /// Reloads with the language pack belonging to the NEW model, not the old one:
    /// packs are per (model, language), so switching checkpoint switches adapter
    /// too — and a pack trained for the previous model must never be carried over,
    /// since llama.cpp would apply it without complaint and quietly corrupt every
    /// generation.
    func switchTaskModel(to option: SLMModelOption) async -> (ok: Bool, error: String?) {
        let adapter = LoRACatalog.installedPath(baseModelID: option.id,
                                                language: nativeLanguage)
        let res = await load(taskEngine, model: option.resourceName, adapterPath: adapter)
        if res.ok {
            taskModelResourceName = option.resourceName
            taskModelID = option.id
        }
        return (res.ok, res.error)
    }

    /// Point the features at a new native language, reloading the task engine when
    /// that changes which LoRA pack applies.
    ///
    /// The reload is skipped when the pack is unchanged — including the common
    /// case where neither language has one, so both resolve to bare weights. That
    /// keeps switching language free for most users and costs a model load only
    /// for those who actually have a pack installed.
    ///
    /// A reload is the only correct way to change adapter: `llama_set_adapters_lora`
    /// acts on the context, so swapping in place would leave every cached prompt
    /// prefix computed under the previous adapter — reused silently, and wrong.
    @discardableResult
    func switchNativeLanguage(to language: TranslateLanguage) async -> (ok: Bool, reloaded: Bool, error: String?) {
        guard language != nativeLanguage else { return (true, false, nil) }
        let previous = taskAdapterPath
        nativeLanguage = language
        let next = taskAdapterPath
        guard next != previous else { return (true, false, nil) }

        let res = await load(taskEngine, model: taskModelResourceName, adapterPath: next)
        if !res.ok {
            NSLog("[LoRA] reload for language %@ failed: %@",
                  language.rawValue, res.error ?? "unknown")
        }
        return (res.ok, true, res.error)
    }

    /// Reload the task engine so a newly downloaded (or deleted) pack takes
    /// effect, without changing model or language. Settings calls this after an
    /// install finishes.
    @discardableResult
    func reloadTaskEngineForAdapterChange() async -> (ok: Bool, error: String?) {
        await load(taskEngine, model: taskModelResourceName, adapterPath: taskAdapterPath)
    }

    /// Switch the compute backend and reload both SLM engines on it — the backend
    /// is fixed when a model is loaded, so there's no cheaper way to change it.
    ///
    /// The old models are released FIRST: loading the new ones while the old are
    /// still mapped would hold two full copies at once, and on a GPU load that
    /// second copy is what pushes a 4 GB device over. That means the SLM features
    /// are briefly dead mid-switch, and if the reload fails they stay dead until
    /// the caller retries — so failures must be surfaced, not swallowed.
    ///
    /// Returns the backend actually in use afterwards, which may not be the one
    /// requested (the simulator has no Metal compute device).
    func switchBackend(to option: SLMBackendOption) async -> (ok: Bool, active: String, error: String?) {
        backendOption = option
        backendDemotedToCPU = false   // an explicit choice deserves a fresh attempt
        await releaseSLMs()
        let ok = await restoreSLMs()
        return (ok, backend, ok ? nil : "reload failed on \(backend)")
    }

    // MARK: - Safety classification

    /// Safety classification: the on-device BERT-Tiny classifier, and nothing else.
    ///
    /// This used to be a two-stage cascade — BERT-Tiny first, then the task SLM
    /// re-classifying anything it flagged, with the SLM's verdict allowed to clear
    /// the flag. That second stage has been removed. What remains is one ~10 MB
    /// TFLite model returning a 10-class + 4-severity verdict in milliseconds, with
    /// no SLM call and no generation on the flagged path.
    ///
    /// Consequence worth knowing: BERT-Tiny's verdict is now FINAL. Nothing
    /// downgrades a false positive, so the flag rate is whatever the classifier
    /// says — see `GuardEngine.decide` for its thresholds.
    func classify(_ text: String, language: TranslateLanguage) async -> SafetyResult {
        let started = DispatchTime.now().uptimeNanoseconds
        let verdict = await guardEngine.classify(text, language: language)
        let ms = Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000

        // One line, carrying everything needed to act on it — because the two
        // questions a reader has are "who flagged this?" and, when it is slow,
        // "slow at what?", and both used to require a second log statement that
        // may not even belong to this message (the guard queue interleaves).
        //
        //   [Guard] [English] "…" → Scam / fraud/Critical (FLAGGED by rule) 3.1ms [48ch]   ← 0 windows: the regex answered
        //   [Guard] [Vietnamese] "…" → Self-harm p_unsafe=0.999 (FLAGGED by model) 16.2ms [105ch 3win 16.2work/16.0cpu]
        //   [Guard] [Vietnamese] "…" → Self-harm p_unsafe=0.999 (FLAGGED by model) 1172.7ms [105ch 3win 16.2work/16.0cpu +1156.5queued 8behind]
        //                                                                                          ^ the model was never the cost
        //   [Guard] [English] "…" → Clean/Safe p_unsafe=0.008 (clean by model) 7.7ms [44ch 1win 7.6work/6.9cpu]
        //                                                ^ 0.8% unsafe mass, i.e. 99.2% sure it is clean — NOT "0.8% confident"
        //   [Guard] [Thai] "…" → Clean/Safe (clean not screened) 0.2ms   ← a gap, not a pass
        //
        // Only the model line carries a number, because only the model computes
        // one. A missing `p_unsafe=` is itself information.
        //
        // No severity after the class on the model lines: v16 has no severity
        // head, so only a rule hit prints the second half (see `Severity`).
        //
        // `ms` is still the caller's wall clock, which is what the user feels;
        // the bracket decomposes it. They differ slightly — the bracket misses
        // the hop back off the guard queue — so trust `ms` for "how slow" and the
        // bracket for "why".
        let t = verdict.timing
        // Length, because the text in this line is truncated to 60 characters and
        // a verdict that won't reproduce is usually a verdict on a DIFFERENT
        // string than the one you read here. Logging the whole message instead
        // would leak content the guardrail just decided to hide.
        var detail = String(format: " %dch", text.count)
        // `win` counts passes that RAN, not windows planned — the loop stops at
        // the first Critical because nothing outranks it, so a 3-window message
        // can legitimately report 1.
        //
        // CPU alongside wall, because they diverge exactly when it matters and
        // the gap is the whole diagnosis: `.utility` work gets descheduled rather
        // than slowed, so starvation inflates wall time while CPU stays flat.
        if t.windows > 0 {
            detail += String(format: " %dwin %.1fwork/%.1fcpu", t.windows, t.workMS, t.cpuMS)
        }
        // Queue wait is the interesting number precisely when it is large, and
        // noise when it is not. Sub-millisecond waits are every idle message.
        if t.waitMS >= 1 { detail += String(format: " +%.1fqueued %dbehind", t.waitMS, t.behind) }

        // "Class/Severity" when severity is known, bare "Class" when it isn't.
        // Since v16 only rule hits carry one, so the presence of the second half
        // is itself the tell for which layer spoke — and printing a placeholder
        // ("Self-harm/—") would put a column in the log that never says anything.
        let outcome = verdict.severity
            .map { "\(verdict.label.displayName)/\($0.displayName)" }
            ?? verdict.label.displayName

        // `p_unsafe=`, and ONLY for model verdicts.
        //
        // The number is the total probability mass on unsafe classes. That is
        // the right thing to print next to FLAGGED, but it read backwards next
        // to Clean while it was called `conf=`: "Clean/Safe conf=0.008" looks
        // like a model that is 0.8% sure, when it is 99.2% sure — 0.008 is how
        // little unsafe mass it found.
        //
        // Naming it fixed that and exposed a worse one. `SafetyResult.clean`
        // carries `confidence: 1`, so the two non-model outcomes both printed
        // 1.000 meaning opposite things:
        //
        //   Scam/Critical p_unsafe=1.000 (FLAGGED by rule)   ← a convention; no
        //                                                      probability exists
        //   Clean/Safe    p_unsafe=1.000 (clean not screened) ← NOTHING RAN, yet
        //                                                      it reads 100% unsafe
        //
        // Only `decide` computes this quantity, so only `.model` prints it. The
        // source tag carries the rest: `by rule` means a pattern matched, `not
        // screened` means no verdict was formed. The `confidence: 1` convention
        // stays on the struct — `GuardrailDemoTests` separates rule hits from
        // model hits with `confidence < 1` — it just no longer reaches the log
        // wearing a name that makes a claim about it.
        var score = ""
        if case .model = verdict.source {
            score = String(format: " p_unsafe=%.3f", verdict.confidence)
        }

        NSLog("[Guard] [%@] \"%@\" → %@%@ (%@ %@) %.1fms%@",
              language.name, String(text.prefix(60)),
              outcome, score,
              verdict.isUnsafe ? "FLAGGED" : "clean", verdict.source.logDescription, ms,
              detail.isEmpty ? "" : " [\(detail.dropFirst())]")
        return verdict
    }

    /// Switch the jurisdiction whose rules apply. Reloads only the rule set — the
    /// classifier packs are per-language and unaffected. Returns the new count.
    @discardableResult
    func switchGuardRegion(to region: GuardRegion) async -> Int {
        guardRegion = region
        guardRuleCount = await guardEngine.loadRules(for: region)
        guardRulesAuthored = guardEngine.rulesAuthored
        return guardRuleCount
    }

    // MARK: - Long-message summarization (chunked map-reduce)

    /// Summarize one long free-text message with a chunked map-reduce, so text far
    /// larger than the small model's ~1024-token context still summarizes without
    /// anything being cut off:
    ///
    ///  - **Map:** split the message into `Prompts.longChunkChars`-sized chunks and
    ///    summarize each into a compact partial.
    ///  - **Reduce:** fold the partials into one, but re-chunk them at the larger
    ///    `Prompts.longReduceChars` budget first, so no single fold prompt exceeds
    ///    the context. If they don't all fit one fold, fold hierarchically (layer
    ///    by layer) until a single summary remains. The FINAL fold — the one that
    ///    yields that single summary — gets the larger `summaryLongTokens` budget so
    ///    the result the user reads isn't truncated.
    ///
    /// `language` is the output language name. Returns nil only if every model call
    /// failed. Backs `Conversation.summarizeMessage` and `GroupChat.summarizeMessage`.
    /// Stream a single-call summarization. Yields the best-effort decoded `summary`
    /// text after each token (so the sheet renders it live), then the authoritative
    /// `parseSummary` result as the final element. Backs the single-call paths — the
    /// whole-conversation summary and a single-chunk long message. Multi-chunk long
    /// messages still go through the map-reduce `summarizeLong`.
    ///
    /// `.bufferingNewest(1)` coalesces tokens: if the main-actor consumer lags, it
    /// only ever sees the latest partial (never a backlog), and the final value —
    /// yielded last — is always delivered.
    func summarizeStream(prompt: String, maxTokens: Int32) -> AsyncStream<String> {
        let engine = taskEngine
        let temp = summaryTemp
        return AsyncStream(bufferingPolicy: .bufferingNewest(1)) { continuation in
            Task {
                NSLog("[Summarize] STREAM — raw SLM input:\n%@", prompt)
                let box = SummaryBuffer()
                await engine.stream(system: nil, history: [], user: prompt,
                                    maxTokens: maxTokens, temperature: temp,
                                    label: .summarize) { chunk, _ in
                    guard !chunk.isEmpty else { return }
                    box.text += chunk
                    if let partial = Prompts.partialSummary(from: box.text) {
                        continuation.yield(partial)
                    }
                }
                let final = Prompts.parseSummary(box.text)
                NSLog("[Summarize] STREAM done: %d chars: %@", final.count, final)
                continuation.yield(final)
                continuation.finish()
            }
        }
    }

    func summarizeLong(_ text: String, language: String, speaker: String) async -> String? {
        NSLog("[Summarize] input: %d chars, language=%@, speaker=%@", text.count, language, speaker)
        // Map over the original message's chunks.
        let mapChunks = chunkForSummary(text, maxTokens: summaryChunkTokens)
        NSLog("[Summarize] MAP: %d chunk(s) (chunkChars=%d)", mapChunks.count, Prompts.longChunkChars)
        var summaries = await summarizeChunks(mapChunks, language: language, speaker: speaker,
                                              maxTokens: summaryMapTokens, phase: "MAP")
        guard !summaries.isEmpty else {
            NSLog("[Summarize] MAP produced 0 summaries — every chunk failed, returning nil")
            return nil
        }

        // Reduce: fold until one summary remains. Each pass re-chunks the joined
        // summaries at the reduce budget so every fold prompt fits the context; the
        // pass that collapses to a single chunk is the final one (larger budget).
        var pass = 0
        while summaries.count > 1 {
            pass += 1
            let chunks = chunkForSummary(summaries.joined(separator: "\n"),
                                        maxTokens: summaryFoldTokens)
            let isFinal = chunks.count == 1
            NSLog("[Summarize] REDUCE pass %d: %d summaries → %d fold-chunk(s)%@",
                  pass, summaries.count, chunks.count, isFinal ? " (FINAL)" : "")
            let folded = await summarizeChunks(chunks, language: language, speaker: speaker,
                                               maxTokens: isFinal ? summaryLongTokens : summaryMapTokens,
                                               phase: "REDUCE#\(pass)")
            // Model failed, or a pathological input that won't shrink → hand back
            // what we have rather than looping forever.
            guard !folded.isEmpty else {
                NSLog("[Summarize] REDUCE pass %d failed — joining current %d summaries", pass, summaries.count)
                return summaries.joined(separator: " ")
            }
            guard folded.count < summaries.count else {
                NSLog("[Summarize] REDUCE pass %d did not shrink (%d→%d) — joining to avoid loop",
                      pass, summaries.count, folded.count)
                return folded.joined(separator: " ")
            }
            summaries = folded
        }
        let final = summaries.first
        NSLog("[Summarize] DONE: final summary %d chars: %@",
              final?.count ?? 0, final ?? "<nil>")
        return final
    }

    /// Most SOURCE tokens a conversation summary will read, newest first.
    ///
    /// A conversation has no length limit — a long backlog renders a prompt of any
    /// size — so something has to bound it. Bounding by tokens rather than message
    /// count keeps the limit honest across languages, and taking the NEWEST is the
    /// right end to keep: "what did I miss" is about recent messages, and the older
    /// ones are the ones the user most likely already read.
    ///
    /// 2048 is deliberately larger than one slot (1024): the transcript is chunked
    /// and folded, so the ceiling is what the summarizer can READ, not what fits a
    /// single call.
    let summaryMaxInputTokens = 2048

    /// Summarize a conversation, chunking when the transcript is too long for one
    /// call.
    ///
    /// Replaces a single unbounded `summarizeStream` on the rendered prompt, which
    /// simply failed once the transcript passed the slot — observed at 1385 tokens,
    /// where the engine (correctly) refused rather than overrunning its batch.
    ///
    /// Chunks are groups of whole MESSAGES, never split mid-message: each group is
    /// rendered by `Prompts.summarize`, so every model call sees the same
    /// `[Name]: …` transcript format the finetune was trained on, rather than a
    /// fragment of one.
    func summarizeConversationStream(_ messages: [ChatMessage],
                                     friendName: String,
                                     native: TranslateLanguage) -> AsyncStream<SummaryEvent> {
        AsyncStream(bufferingPolicy: .bufferingNewest(1)) { continuation in
            Task {
                let groups = self.groupMessagesForSummary(messages, friendName: friendName,
                                                          native: native)
                guard !groups.isEmpty else { continuation.finish(); return }

                // Fits one call — no map-reduce, no progress line.
                if groups.count == 1 {
                    let prompt = Prompts.summarize(groups[0], friendName: friendName, native: native)
                    for await partial in self.summarizeStream(prompt: prompt,
                                                              maxTokens: self.summaryTokens) {
                        continuation.yield(.text(partial))
                    }
                    continuation.finish()
                    return
                }

                NSLog("[Summarize] CONVERSATION: %d message(s) → %d chunk(s)",
                      messages.count, groups.count)
                var parts: [String] = []
                for (i, group) in groups.enumerated() {
                    let prompt = Prompts.summarize(group, friendName: friendName, native: native)
                    if let raw = await self.taskEngine.complete(
                        system: nil, history: [], user: prompt,
                        maxTokens: self.summaryMapTokens, temperature: self.summaryTemp,
                        label: .summarizeChunk(phase: "CONV-MAP")) {
                        let parsed = Prompts.parseSummary(raw)
                        if !parsed.isEmpty {
                            Self.warnIfTruncated(parsed, phase: "CONV-MAP", index: i + 1)
                            parts.append(parsed)
                        }
                    }
                    continuation.yield(.mapping(done: i + 1, total: groups.count))
                }
                guard !parts.isEmpty else { continuation.finish(); return }
                guard parts.count > 1 else {
                    continuation.yield(.text(parts[0])); continuation.finish(); return
                }

                let fold = Prompts.combineSummaries(parts, language: native.name)
                for await partial in self.summarizeStream(prompt: fold,
                                                          maxTokens: self.summaryLongTokens) {
                    continuation.yield(.text(partial))
                }
                continuation.finish()
            }
        }
    }

    /// Take the newest messages up to `summaryMaxInputTokens`, then pack them into
    /// groups whose rendered prompt fits one call.
    ///
    /// Both limits are measured with the model's tokenizer. A single message larger
    /// than a whole chunk still gets its own group — better an over-long prompt the
    /// engine refuses (loudly, with the message named) than silently dropping the
    /// message the user asked about.
    private func groupMessagesForSummary(_ messages: [ChatMessage],
                                         friendName: String,
                                         native: TranslateLanguage) -> [[ChatMessage]] {
        let usable = messages.filter { !$0.text.isEmpty }
        guard !usable.isEmpty else { return [] }
        let count: (String) -> Int = { [taskEngine] in
            taskEngine.tokenCount(of: $0) ?? Prompts.estimatedTokens($0)
        }

        // Newest-first walk, so what gets dropped is the oldest.
        var kept: [ChatMessage] = []
        var total = 0
        for m in usable.reversed() {
            let n = count(m.text) + 8   // "[Name]: " and the newline
            if total + n > summaryMaxInputTokens, !kept.isEmpty { break }
            kept.append(m); total += n
        }
        kept.reverse()
        if kept.count < usable.count {
            NSLog("[Summarize] CONVERSATION: reading the newest %d of %d messages (%d tok cap)",
                  kept.count, usable.count, summaryMaxInputTokens)
        }

        // Pack into per-call groups against the real prompt, instruction included.
        var groups: [[ChatMessage]] = []
        var current: [ChatMessage] = []
        for m in kept {
            let candidate = current + [m]
            if !current.isEmpty,
               count(Prompts.summarize(candidate, friendName: friendName, native: native))
                   > summaryChunkTokens {
                groups.append(current); current = [m]
            } else {
                current = candidate
            }
        }
        if !current.isEmpty { groups.append(current) }
        return groups
    }

    /// What a long-message summarization is doing right now.
    ///
    /// Two distinct signals, because a map-reduce has two distinct kinds of work.
    /// Only the FINAL fold produces text the user reads: every MAP output and every
    /// intermediate fold is scaffolding that a later pass replaces. Rendering those
    /// would show text and then take it away, which is worse than showing nothing —
    /// hence progress for them and streamed text only at the end.
    enum SummaryEvent {
        /// One MAP chunk finished. `total` is known up front (it is the chunk
        /// count); this is the bulk of the work on a long message.
        case mapping(done: Int, total: Int)
        /// A reduce pass is folding partial summaries together. Deliberately
        /// untotalled — how many passes are needed depends on how fast the
        /// summaries shrink, which isn't knowable in advance.
        case combining
        /// Summary text, streaming from the final fold. May arrive many times.
        case text(String)
    }

    /// Streaming map-reduce summary of one long message.
    ///
    /// Same algorithm as the blocking form: MAP over the message's chunks, then
    /// fold until a single summary remains, re-chunking each pass so every fold
    /// prompt fits the context. The difference is that it reports progress
    /// throughout and streams the last pass, so a 5-chunk message stops being ~30 s
    /// of unexplained spinner.
    ///
    /// Yields no `.text` at all if the model failed everywhere, which is how the
    /// caller distinguishes failure from an empty summary.
    func summarizeLongStream(_ text: String,
                             language: String,
                             speaker: String) -> AsyncStream<SummaryEvent> {
        AsyncStream(bufferingPolicy: .bufferingNewest(1)) { continuation in
            Task {
                let mapChunks = self.chunkForSummary(text, maxTokens: self.summaryChunkTokens)
                guard !mapChunks.isEmpty else { continuation.finish(); return }

                // A message that fits in one chunk needs no map-reduce at all: it is
                // a single call, so stream it directly. Going through the general
                // path would emit .mapping(1, 1) and flash a progress line for the
                // whole generation.
                if mapChunks.count == 1 {
                    await self.streamFinalFold(mapChunks[0], language: language,
                                               speaker: speaker,
                                               maxTokens: self.summaryTokens,
                                               into: continuation)
                    continuation.finish()
                    return
                }

                NSLog("[Summarize] STREAM MAP: %d chunk(s)", mapChunks.count)
                var summaries: [String] = []
                for (i, chunk) in mapChunks.enumerated() {
                    let prompt = Prompts.summarizeText(chunk, language: language, speaker: speaker)
                    if let raw = await self.taskEngine.complete(
                        system: nil, history: [], user: prompt,
                        maxTokens: self.summaryMapTokens, temperature: self.summaryTemp,
                        label: .summarizeChunk(phase: "MAP")) {
                        let parsed = Prompts.parseSummary(raw)
                        if !parsed.isEmpty {
                            Self.warnIfTruncated(parsed, phase: "MAP", index: i + 1)
                            summaries.append(parsed)
                        }
                    }
                    continuation.yield(.mapping(done: i + 1, total: mapChunks.count))
                }
                guard !summaries.isEmpty else {
                    NSLog("[Summarize] STREAM MAP produced nothing — every chunk failed")
                    continuation.finish()
                    return
                }

                // Fold until one remains. The pass that collapses to a single chunk
                // is the one the user actually reads, so it streams; the rest only
                // report progress.
                var pass = 0
                while summaries.count > 1 {
                    pass += 1
                    let folds = self.chunkForSummary(summaries.joined(separator: "\n"),
                                                     maxTokens: self.summaryFoldTokens)
                    NSLog("[Summarize] STREAM REDUCE pass %d: %d → %d fold(s)%@",
                          pass, summaries.count, folds.count,
                          folds.count == 1 ? " (FINAL)" : "")

                    if folds.count == 1 {
                        await self.streamFinalFold(folds[0], language: language,
                                                   speaker: speaker,
                                                   maxTokens: self.summaryLongTokens,
                                                   into: continuation)
                        continuation.finish()
                        return
                    }

                    continuation.yield(.combining)
                    let folded = await self.summarizeChunks(folds, language: language,
                                                            speaker: speaker,
                                                            maxTokens: self.summaryMapTokens,
                                                            phase: "REDUCE#\(pass)")
                    // Failed, or a pathological input that refuses to shrink — hand
                    // back what exists rather than looping forever. Both are the
                    // blocking version's guards, kept identical.
                    guard !folded.isEmpty else {
                        continuation.yield(.text(summaries.joined(separator: " ")))
                        continuation.finish()
                        return
                    }
                    guard folded.count < summaries.count else {
                        continuation.yield(.text(folded.joined(separator: " ")))
                        continuation.finish()
                        return
                    }
                    summaries = folded
                }

                // Exactly one summary and no fold was needed.
                continuation.yield(.text(summaries[0]))
                continuation.finish()
            }
        }
    }

    /// Flag a partial summary that looks cut off mid-sentence.
    ///
    /// Worth its own warning because of where the damage surfaces: a truncated MAP
    /// output is INPUT to the fold, so the final summary reads as cropped while
    /// reporting `stop=eos`, and the `[SLMPerf]` line that actually says
    /// `stop=max_tokens` belongs to a call several seconds earlier. This puts the
    /// complaint next to its cause.
    ///
    /// A heuristic, not a proof — a summary can legitimately end without
    /// punctuation — so it warns and continues rather than failing anything.
    private static func warnIfTruncated(_ text: String, phase: String, index: Int) {
        guard let last = text.trimmingCharacters(in: .whitespacesAndNewlines).last,
              !".!?…:;\"')」』".contains(last) else { return }
        NSLog("[Summarize] WARNING %@ chunk %d ends mid-sentence (%d chars) — likely hit "
              + "the token cap. Every later fold now summarises damaged text: %@",
              phase, index, text.count, String(text.suffix(60)))
    }

    /// Run the one pass whose output the user reads, streaming it out as `.text`.
    private func streamFinalFold(_ chunk: String,
                                 language: String,
                                 speaker: String,
                                 maxTokens: Int32,
                                 into continuation: AsyncStream<SummaryEvent>.Continuation) async {
        let prompt = Prompts.summarizeText(chunk, language: language, speaker: speaker)
        for await partial in summarizeStream(prompt: prompt, maxTokens: maxTokens) {
            continuation.yield(.text(partial))
        }
    }

    /// Summarize each chunk independently, dropping any the model failed on. Shared
    /// by both the map and reduce passes of `summarizeLong`. `phase` is a log label
    /// identifying the pass (e.g. "MAP", "REDUCE#1").
    private func summarizeChunks(_ chunks: [String],
                                 language: String,
                                 speaker: String,
                                 maxTokens: Int32,
                                 phase: String) async -> [String] {
        var out: [String] = []
        for (i, chunk) in chunks.enumerated() {
            let prompt = Prompts.summarizeText(chunk, language: language, speaker: speaker)
            NSLog("[Summarize] %@ chunk %d/%d — raw SLM input:\n%@", phase, i + 1, chunks.count, prompt)
            if let raw = await taskEngine.complete(system: nil, history: [],
                                                   user: prompt,
                                                   maxTokens: maxTokens, temperature: summaryTemp,
                                                   label: .summarizeChunk(phase: phase)) {
                let parsed = Prompts.parseSummary(raw)
                NSLog("[Summarize] %@ chunk %d/%d (%d chars) → %d chars: %@",
                      phase, i + 1, chunks.count, chunk.count, parsed.count, parsed)
                out.append(parsed)
            } else {
                NSLog("[Summarize] %@ chunk %d/%d (%d chars) → SLM returned nil",
                      phase, i + 1, chunks.count, chunk.count)
            }
        }
        return out
    }

    /// Translate a message of any length, streamed for the UI to render token by
    /// token.
    ///
    /// Split into pieces of at most `translateChunkInputTokens` (a short message is
    /// just one), each translated with the `translateChunkTokens` output budget and
    /// joined back in order — see the budget arithmetic above. Unlike a summary
    /// there is no reduce: every chunk's full translation is kept. A message over
    /// `translateMaxInputTokens` is not translated at all (callers check
    /// `isTooLongToTranslate` first to say why).
    ///
    /// Each element is the WHOLE translation so far — completed chunks joined with
    /// the chunk currently arriving — so a consumer just assigns it and never has
    /// to accumulate. That matters for a chunked job: the caller has no idea where
    /// one chunk ends and the next begins, and shouldn't need to.
    ///
    /// Yields nothing at all if every chunk fails, which is how the caller
    /// distinguishes failure from an empty translation.
    func translateLongStream(_ text: String,
                             to target: TranslateLanguage) -> AsyncStream<String> {
        let engine = taskEngine
        let temp = translateTemp
        let cap = translateChunkTokens
        let chunks = isTooLongToTranslate(text) ? [] : chunkForTranslation(text)
        // Detected ONCE over the whole message, not per chunk: a single chunk can
        // be too short to identify, and a message whose chunks disagreed about
        // their own language would translate inconsistently.
        let source = Prompts.sourceLanguage(of: text)

        return AsyncStream(bufferingPolicy: .bufferingNewest(1)) { continuation in
            Task {
                guard !chunks.isEmpty else { continuation.finish(); return }
                var completed: [String] = []
                for (i, chunk) in chunks.enumerated() {
                    let prompt = Prompts.translate(chunk, from: source, to: target)
                    NSLog("[Translate] STREAM chunk %d/%d — raw SLM input:\n%@",
                          i + 1, chunks.count, prompt)
                    let box = SummaryBuffer()
                    await engine.stream(system: nil, history: [], user: prompt,
                                        maxTokens: cap, temperature: temp,
                                        label: .translate) { piece, _ in
                        guard !piece.isEmpty else { return }
                        box.text += piece
                        guard let partial = Prompts.partialTranslation(from: box.text) else { return }
                        continuation.yield((completed + [partial]).joined(separator: " "))
                    }
                    // Re-parse authoritatively: the partial view is best-effort, and
                    // the final one strips anything the stream showed provisionally.
                    let final = Prompts.parseTranslation(box.text)
                    NSLog("[Translate] STREAM chunk %d/%d done: %d chars", i + 1, chunks.count, final.count)
                    guard !final.isEmpty else { continue }   // drop a failed chunk, keep going
                    completed.append(final)
                    continuation.yield(completed.joined(separator: " "))
                }
                continuation.finish()
            }
        }
    }

    // MARK: - Long-message translation (chunked)


    // MARK: - Image tagging

    /// Tag one attached image with the on-device EdgeNeXt classifier, returning the
    /// tag names sorted by descending score. Empty if the engine is unavailable or
    /// the image can't be processed (attachment stays un-searchable, never blocks).
    func tagImage(_ data: Data) async -> [String] {
        let tags = await imageTagEngine.tag(imageData: data)
        NSLog("[ImageTag] %d tags: %@", tags.count,
              tags.prefix(8).map { "\($0.name):\(String(format: "%.2f", $0.score))" }
                  .joined(separator: ", "))
        return tags.map(\.name)
    }
}

/// Mutable box for accumulating streamed summary tokens off the main thread. Only
/// touched inside the engine's serialized stream callback, so no locking is needed.
private final class SummaryBuffer {
    var text = ""
}
