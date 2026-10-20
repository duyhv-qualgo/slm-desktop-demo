import Foundation
import SwiftUI

/// One chat thread with a single contact. Long-lived: the `ConversationStore`
/// retains every `Conversation`, so a reply `Task` started in `send()` keeps
/// running even after the user navigates back to the home list — which is exactly
/// what lets a reply land as *unread* while the user is away.
///
/// Owns its messages, its per-chat UI state, and the orchestration for the three
/// features. All model work goes through the shared `ChatService`.
@MainActor
final class Conversation: ObservableObject, Identifiable {

    let id = UUID()
    /// When this thread was created — used as the inbox sort key until the first
    /// message arrives, so a brand-new blank chat sorts to the top instead of the bottom.
    let createdAt = Date()
    /// Display name of the contact and the persona name used in prompts.
    let contactName: String
    /// This contact's chat persona (folded into the chat prompt / API system role).
    let persona: String

    /// Up-to-two-letter monogram for the avatar (renders reliably on every device
    /// and simulator, unlike emoji).
    var initials: String { Avatar.initials(contactName) }

    /// Deterministic avatar tint derived from the name, so each contact keeps a
    /// stable color across launches.
    var tint: Color { Avatar.tint(contactName) }

    @Published var messages: [ChatMessage]
    @Published var input: String = ""
    /// Ticks on every programmatic write to `input` via `setInput`. See there.
    @Published private(set) var inputRevision = 0

    @Published var isBotTyping: Bool = false
    @Published var isSuggesting: Bool = false
    @Published var isSummarizing: Bool = false

    @Published var suggestions: [String] = []
    @Published var summary: String?
    @Published var showSummary: Bool = false
    /// What the summarizer is doing while there is no text yet. A long backlog is
    /// chunked into several model calls, so an unlabelled spinner reads as a hang.
    @Published var summaryProgress: String?

    @Published var errorMessage: String?
    /// Neutral (non-error) note, e.g. the smart-reply model declining to suggest.
    @Published var notice: String?

    /// Set when suggestions are blocked because this conversation's language pack
    /// isn't installed. Carries the pack itself, not just a message, so the notice
    /// can offer the install right there — Settings only exposes a Get button for
    /// the NATIVE language's pack, so telling the user to go there would be telling
    /// them to do something the app gives them no way to do.
    @Published var missingPack: LoRAAdapter?

    /// An event the task model spotted in a recent message, surfaced as an in-chat
    /// reminder banner. Transient (not persisted); survives navigation because the
    /// store keeps this object alive.
    @Published var pendingEvent: DetectedEvent?
    /// Transient "✓ Reminder set" confirmation shown after the user taps Remind me.
    @Published var eventReminderNote: String?

    /// True while the chat screen for this conversation is on-screen. Set by
    /// `ChatView.onAppear/onDisappear`. Drives whether an arriving reply is stored
    /// read (user is looking) or unread (user has navigated away).
    var isActive: Bool = false

    /// Pending debounce for auto-suggest, cancelled whenever the relevant state
    /// changes (new message, user starts typing, chat closed).
    private var autoSuggestTask: Task<Void, Never>?

    /// Id of the friend message we've already auto-suggested for, so we generate at
    /// most once per incoming message. Nothing resets it, so this is a hard
    /// once-per-message cap — see `scheduleAutoSuggest`.
    private var lastAutoSuggestedID: UUID?

    /// What the last send pre-warmed for the next suggest (see
    /// `prewarmNextSuggest`), kept so the suggest can say whether it was served
    /// from it and, if not, why. `tokens` is nil until the prefill returns.
    private struct SuggestPrewarm {
        let id = UUID()
        let windowStart: Int
        let language: TranslateLanguage
        let adapter: String?
        var tokens: Int?
    }
    private var suggestPrewarm: SuggestPrewarm?

    /// Window start and language of the previous suggest in this chat — a change
    /// in either means the prompt's prefix moved (a re-anchor), which is the
    /// only kind of turn allowed to evaluate the whole window.
    private var lastSuggestWindow: (start: Int, language: TranslateLanguage)?

    /// Shared model plumbing. `unowned` because the store owns both the service and
    /// every conversation for the app's lifetime.
    private unowned let service: ChatService

    /// Called after any change that should be written to disk (a message added,
    /// a reply finalized, read state or a cached translation updated). Set by the
    /// store; invoked via `persist()`.
    var onPersist: (() -> Void)?

    /// Flat, Codable snapshot of this thread for on-disk storage.
    var snapshot: StoredConversation {
        StoredConversation(contactName: contactName, messages: messages.map(\.snapshot))
    }

    private func persist() { onPersist?() }

    init(contactName: String,
         messages: [ChatMessage],
         service: ChatService) {
        self.contactName = contactName
        self.persona = Prompts.persona(for: contactName)
        self.messages = messages
        self.service = service
    }

    // MARK: - Row helpers (conversation list)

    var unreadCount: Int {
        messages.filter { $0.sender == .friend && !$0.isRead }.count
    }

    /// Number of friend messages since your last reply. Unlike `unreadCount`, this
    /// does NOT depend on the read flag, so it survives summarizing (which marks
    /// messages read) — the catch-up banner stays visible until you actually reply.
    var unrepliedCount: Int {
        guard let lastYou = messages.lastIndex(where: { $0.sender == .you }) else {
            // You've never replied → every friend message is unreplied.
            return messages.filter { $0.sender == .friend && !$0.text.isEmpty }.count
        }
        return messages[(lastYou + 1)...].filter { $0.sender == .friend && !$0.text.isEmpty }.count
    }

    /// One-line preview for the inbox row.
    ///
    /// Substitutes the class and severity for any message the thread would blur.
    /// Without this the guardrail leaks around itself: the conversation list
    /// renders raw text, so a message hidden behind a reveal overlay inside the
    /// chat was printed in full on the screen before it — and on a screen the
    /// user never had to open. Anything else that shows message text outside the
    /// bubble (a notification, a search result, a widget) needs the same check.
    ///
    /// Deliberately ignores `revealUnsafe`: revealing a message inside the thread
    /// is a choice made for that moment and that screen, and shouldn't quietly
    /// republish it to the inbox.
    ///
    /// An UNSCREENED message still shows its text, consistent with failing open
    /// everywhere else. The window is small because the newest message of any
    /// batch is screened eagerly for exactly this reason — see `seedUnreadBurst`.
    var lastMessagePreview: String {
        guard let last = messages.last(where: { !$0.text.isEmpty }) else { return "" }
        let who = last.sender == .you ? "You: " : ""
        guard !last.isSensitive else {
            let cls = last.safety?.displayName ?? "Flagged"
            let sev = last.severity?.displayName ?? ""
            return who + "⚠️ " + (sev.isEmpty ? cls : "\(cls) · \(sev)")
        }
        return who + last.text
    }

    var lastActivityDate: Date {
        messages.last?.date ?? createdAt
    }

    /// True while any generation for THIS conversation is in flight.
    var isBusy: Bool { isBotTyping || isSuggesting || isSummarizing || isComposeGenerating }

    // MARK: - Sending / friend reply

    func send(_ rawText: String) {
        let text = rawText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !isBotTyping else { return }
        // The draft this bar was rewriting is now a sent message, so the chips
        // have nothing left to act on — and an `Undo` still sitting there would
        // offer to restore a draft into a box the user has already emptied.
        // Placed after the guard so a rejected send leaves the bar alone.
        endCompose()
        input = ""
        suggestions = []
        notice = nil
        markFriendMessagesRead()
        let mine = ChatMessage(sender: .you, text: text)
        messages.append(mine)
        persist()
        prewarmNextSuggest()
        Task { await detectEvent(in: mine) }
        Task { await classifySafety(in: mine) }
        Task { await replyFromFriend() }
    }

    /// Send an image attachment. The bytes are stored on disk (`ImageStore`), the
    /// bubble references them by id, and the on-device tagger runs in the background
    /// so the photo becomes searchable. No friend reply is generated for an image.
    func sendImage(_ data: Data, caption: String = "") {
        guard let imageID = ImageStore.save(data) else {
            errorMessage = "Couldn't save the image."
            return
        }
        // The caption comes out of the same input box, so this consumes the draft
        // exactly as `send` does. `sendFile` deliberately doesn't — it attaches
        // without touching the draft, so a rewrite in progress survives it.
        endCompose()
        markFriendMessagesRead()
        let mine = ChatMessage(sender: .you,
                               text: caption.trimmingCharacters(in: .whitespacesAndNewlines),
                               imageID: imageID)
        messages.append(mine)
        persist()
        Task { await tagImage(in: mine) }
    }

    /// Send a document (non-image file) attachment. Bytes are stored on disk
    /// (`FileStore`); the bubble keeps the id, original name, and size. No tagging or
    /// friend reply is generated for a file.
    func sendFile(_ data: Data, name: String) {
        guard let fileID = FileStore.save(data) else {
            errorMessage = "Couldn't save the file."
            return
        }
        markFriendMessagesRead()
        let mine = ChatMessage(sender: .you, text: "",
                               fileID: fileID, fileName: name, fileSize: data.count)
        messages.append(mine)
        persist()
    }

    /// Run the image-tagging model over an attached image once, caching the tags on
    /// the message so it stays searchable offline and is never re-tagged.
    func tagImage(in message: ChatMessage) async {
        guard let imageID = message.imageID, message.imageTags == nil,
              let data = ImageStore.data(for: imageID) else { return }
        let tags = await service.tagImage(data)
        message.imageTags = tags
        persist()
    }

    /// Generate the friend's reply. The finished bubble is stored UNREAD regardless
    /// of whether the user is looking: a message you haven't answered stays unread —
    /// and therefore summarizable — until you reply (`markFriendMessagesRead()` in
    /// `send()`) or open a summary. Runs to completion even if the user leaves.
    private func replyFromFriend() async {
        isBotTyping = true
        errorMessage = nil
        let placeholder = ChatMessage(sender: .friend, text: "", isRead: true)
        messages.append(placeholder)

        // The friend speaks only through the API. Drop the placeholder rather than
        // leaving an empty bubble typing forever, and name the missing file.
        guard let chatAPI = service.chatAPI else {
            messages.removeAll { $0.id == placeholder.id }
            errorMessage = ChatService.missingAPIMessage
            isBotTyping = false
            return
        }

        // Stream the completion (SSE), displaying the CLEANED text each chunk so
        // the reply types in live.
        let history = Array(messages.dropLast())   // exclude the placeholder
        let raw = RawBuffer()
        let reply = await chatAPI.streamReply(persona: persona, history: history) { piece in
            DispatchQueue.main.async {
                raw.text += piece
                placeholder.text = Prompts.clean(raw.text)
            }
        }
        if let reply, !reply.isEmpty {
            placeholder.text = Prompts.clean(reply)
            placeholder.isRead = false   // un-replied → unread, so it stays summarizable
            Task { await detectEvent(in: placeholder) }
            Task { await classifySafety(in: placeholder) }
        } else {
            messages.removeAll { $0.id == placeholder.id }
            errorMessage = chatAPI.lastError ?? "\(contactName) (API) didn't reply — try again."
        }
        isBotTyping = false
        persist()
        scheduleAutoSuggest()
    }

    // MARK: - Demo: simulated unread backlog

    /// True while a simulated unread burst is being generated (demo helper), so the
    /// toolbar can show a spinner and stay disabled.
    @Published var isSeedingBurst = false

    /// DEMO HELPER: generate a burst of consecutive friend texts in ONE on-device
    /// call and fan it out into several UNREAD bubbles — simulating messages that
    /// piled up while the user was away. A turn-by-turn reply only ever produces a
    /// single bubble, which doesn't exercise the Facebook-style "Summarize N unread
    /// messages" flow; this seeds a realistic list for it to work on. The bubbles
    /// are forced unread (regardless of `isActive`) so the summary banner appears,
    /// and their timestamps are spread over the recent past — ending ~now — so the
    /// burst reads as having accumulated over the last several minutes.
    func seedUnreadBurst() async {
        guard !isSeedingBurst, !isBotTyping else { return }
        isSeedingBurst = true
        isBotTyping = true          // reuse the "… is typing" indicator while it runs
        errorMessage = nil

        // Continue the ACTUAL thread so the burst is context-aware, not random.
        let recent = Array(messages.suffix(Prompts.historyWindow))
        let instruction = Prompts.friendBacklog(from: contactName)

        guard let chatAPI = service.chatAPI else {
            errorMessage = ChatService.missingAPIMessage
            isSeedingBurst = false
            return
        }

        // Put the burst instruction as the FINAL user turn, after the real
        // conversation. Two reasons: (1) Gemini continues a USER turn reliably —
        // continuing after a friend/model turn returns empty or stunted output
        // (the "only 2 messages" bug); (2) keeping the instruction right after the
        // history anchors the burst to the ACTUAL thread + latest message, instead
        // of the generic nudge that drifted onto random new topics. A light
        // identity is the system prompt. (An empty thread → contents is just the
        // instruction turn, which is fine.)
        var apiHistory = recent
        apiHistory.append(ChatMessage(sender: .you, text: instruction))
        let identity = "You are \(contactName), \(Prompts.userName)'s close friend, in a casual text chat with \(Prompts.userName)."
        let raw = await chatAPI.reply(persona: identity, history: apiHistory)

        let lines = raw.map(Prompts.splitBacklog) ?? []
        if lines.isEmpty {
            errorMessage = "\(contactName) didn't reply — try again."
        } else {
            // Newest ends at ~now; each earlier text is ~2.5 min further back, so the
            // pile reads as a backlog that arrived while the user was away.
            let gap: TimeInterval = 150
            let now = Date()
            for (i, text) in lines.enumerated() {
                let ago = Double(lines.count - 1 - i) * gap
                let msg = ChatMessage(sender: .friend, text: text,
                                      date: now.addingTimeInterval(-ago), isRead: false)
                messages.append(msg)
            }
            // A backlog is the BULK path: these are appended unscreened and wait
            // for `screenFromViewport`, so a pile nobody scrolls through costs
            // nothing. The newest is the exception — the inbox row renders it via
            // `lastMessagePreview`, which shows raw text, so deferring that one
            // would put unscreened content on a screen the user never opened.
            if let newest = messages.last {
                Task { await classifySafety(in: newest) }
            }
            persist()
        }
        isBotTyping = false
        isSeedingBurst = false
        scheduleAutoSuggest()
    }

    // MARK: - Feature: Translation

    /// Translate a message. When it's already translated, tapping just toggles the
    /// cached text — unless `force` is set (Retranslate), which drops the cache and
    /// regenerates with the current prompt/model.
    func translate(_ message: ChatMessage, force: Bool = false) async {
        if !force, message.translation != nil {
            message.showTranslation.toggle()
            return
        }
        guard !message.isTranslating else { return }
        if force { message.translation = nil }
        message.isTranslating = true
        let target = Prompts.autoTarget(for: message.text,
                                        native: service.nativeLanguage)
        // Reveal the (empty) translation row up front so streamed text lands in a
        // container the user is already looking at, rather than appearing all at
        // once when the row unhides at the end.
        message.translation = ""
        message.showTranslation = true

        // Chunked so a long message's translation isn't cut off at the token cap;
        // streamed so a long one isn't a silent wait. Each element is the whole
        // translation so far, so this just assigns.
        var received = false
        for await partial in service.translateLongStream(message.text, to: target) {
            received = true
            message.translation = partial
        }
        if !received { message.translation = "(translation failed)" }
        message.isTranslating = false
        persist()   // cache the translation across launches
    }

    // MARK: - Feature: Auto-reply suggestions

    /// The `auto_suggestion` finetune returns all candidates in ONE call as a JSON
    /// object — `{"suggestions":[{intent,text},…],"refused":…,"reason":…}` — with a
    /// refusal signal for sensitive / bad-actor context (surfaced as a neutral
    /// notice, not an error). Language follows the latest message (see
    /// `Prompts.suggestReplies`).
    /// `waitedBefore` is latency the user already spent before this call — the
    /// auto-suggest debounce — so the turn log can report the true end-to-end wait
    /// against the 3s budget, not just the model's share. 0 when called directly
    /// (the missing-pack notice's Suggest button, or group chat).
    func suggest(waitedBefore: Double = 0) async {
        guard !isSuggesting, !isBotTyping else { return }

        // One resolution, two uses: the prompt asks for this language and the
        // generation runs under its LoRA pack. Deriving them separately is how they
        // drift apart — see ChatService.suggestAdapter.
        //
        // Resolved BEFORE the busy flag is claimed, because a missing pack ends the
        // turn without generating and there is nothing to unwind.
        let language = Prompts.conversationLanguage(messages, native: service.nativeLanguage)
        let adapter: String?
        switch service.suggestAdapter(for: language) {
        case .pack(let path): adapter = path
        case .loaded:         adapter = nil
        case .missing(let pack):
            raiseMissingPack(pack, language: language)
            return
        }

        isSuggesting = true
        errorMessage = nil
        notice = nil
        missingPack = nil
        suggestions = []

        let turn = SLMTurnTimer("suggest[1:1]", preDelay: waitedBefore)
        let prompt = Prompts.suggestReplies(history: messages, friendName: contactName,
                                            language: language)
        // print (not NSLog): os_log truncates long strings, cropping the prompt.
        print("[Suggest] SLM input prompt (1:1 with \(contactName)):\n\(prompt)\n[Suggest] --- end prompt ---")
        var result = Prompts.SuggestionResult(suggestions: [], refused: false, reason: nil)
        var sawEngineError = false
        var attempts = 0
        // The FIRST attempt's metrics are the ones that say whether the cache
        // worked; a retry re-sends the same prompt and is served almost whole.
        var firstMetrics: ChatLMTurnMetrics?

        // Sample up to 3× (temperature 0.8): a single generation can come back empty
        // or malformed, so retry until we get chips or an explicit refusal. Each
        // retry is a FULL extra generation — the usual reason a turn blows the
        // latency budget, so the turn log reports how many ran.
        for attempt in 0..<3 {
            attempts = attempt + 1
            let (out, metrics) = await service.taskEngine.completeMeasured(
                system: nil, history: [], user: prompt,
                maxTokens: service.suggestTokens, temperature: service.suggestTemp,
                label: .suggest, adapterPath: adapter)
            if attempt == 0 { firstMetrics = metrics }
            guard let raw = out else {
                sawEngineError = true
                NSLog("[Suggest] attempt %d → SLM returned nil (engine error)", attempt + 1)
                continue
            }
            result = Prompts.parseSuggestions(raw)
            // Single-line (escaped newlines) so it survives a "[Suggest]" console filter.
            let escaped = raw.replacingOccurrences(of: "\n", with: "\\n")
            NSLog("[Suggest] attempt %d — raw SLM output: %@", attempt + 1, escaped)
            NSLog("[Suggest] attempt %d — parsed: refused=%@ reason=%@ suggestions=%@",
                  attempt + 1, "\(result.refused)", result.reason ?? "<nil>", "\(result.suggestions)")
            if result.refused || !result.suggestions.isEmpty { break }
        }

        if result.refused {
            notice = "No auto-reply suggested for this message."
        } else if !result.suggestions.isEmpty {
            suggestions = Array(result.suggestions.prefix(3))
        } else if sawEngineError {
            errorMessage = "Couldn't generate suggestions — try again."
        } else {
            notice = "No suggestion right now — try again."
        }
        // `window` is the block-quantized history slice (see Prompts.historyWindowStride).
        // A stable start across consecutive turns is what lets the engine's KV cache
        // skip re-prefilling — if `cached` on the [SLMPerf] line stays tiny while this
        // start holds steady, the prefix is being broken somewhere else in the prompt.
        let start = Prompts.windowStart(messageCount: messages.count,
                                        window: Prompts.suggestWindow,
                                        stride: Prompts.suggestStride)
        logSuggestKV(firstMetrics, windowStart: start, language: language, adapter: adapter)
        turn.finish("attempts=\(attempts) chips=\(suggestions.count) "
                  + "prompt=\(prompt.count) chars "
                  + "window=\(start)..\(messages.count) (\(messages.count - start) msgs, "
                  + "stride=\(Prompts.suggestStride))")
        isSuggesting = false
    }

    /// One `[SuggestKV]` line per suggest — `eval` (prompt tokens actually run
    /// through the model) is the number the 3 s budget depends on; target ≤60 on
    /// steady turns — plus a `re-anchor` line whenever the window moved, saying
    /// whether the pre-warmed slot absorbed it or the suggest paid a full prefill.
    /// Consumes the pending pre-warm either way.
    private func logSuggestKV(_ m: ChatLMTurnMetrics?, windowStart start: Int,
                              language: TranslateLanguage, adapter: String?) {
        defer {
            suggestPrewarm = nil
            lastSuggestWindow = (start, language)
        }
        guard let m else {
            NSLog("[SuggestKV] no metrics (engine error) window=%d..%d", start, messages.count)
            return
        }
        let cached = Int(m.cachedPrefixTokenCount)
        let prewarm: String
        let prewarmHit: Bool
        if let pw = suggestPrewarm {
            if let n = pw.tokens, cached >= n {
                prewarm = "hit(\(n))"; prewarmHit = true
            } else {
                // Reuse switched off (Settings → Diagnostics) discards every
                // slot on purpose; say so rather than blaming eviction.
                let why = !service.prefixReuseEnabled ? "reuse-off"
                    : !service.prefixReuseActive ? "model-off"
                    : pw.windowStart != start ? "window-moved"
                    : pw.language != language ? "language-changed"
                    : pw.adapter != adapter ? "pack-changed"
                    : pw.tokens == nil ? "not-ready"
                    : "evicted"
                prewarm = "miss(\(why)\(pw.tokens.map { ",\($0)" } ?? ""))"; prewarmHit = false
            }
        } else {
            prewarm = "none"; prewarmHit = false
        }
        let reanchor: String
        switch lastSuggestWindow {
        case nil:                                   reanchor = "first"
        case let prev? where prev.language != language: reanchor = "language"
        case let prev? where prev.start != start:   reanchor = "window"
        default:                                    reanchor = "no"
        }
        NSLog("[SuggestKV] eval=%d in=%d cached=%d slot=%d reanchor=%@ prewarm=%@ window=%d..%d stride=%d",
              m.prefillTokenCount, m.promptTokenCount, cached, m.cacheSlotIndex,
              reanchor, prewarm, start, messages.count, Prompts.suggestStride)
        if reanchor != "no" {
            NSLog("[SuggestKV] re-anchor (%@) window=%d..%d served=%@ eval=%d",
                  reanchor, start, messages.count,
                  prewarmHit ? "prewarm" : "full-prefill", m.prefillTokenCount)
        }
    }

    /// Prefill the next suggest's prompt NOW, while the reply is in flight.
    ///
    /// Everything the next suggest will read except the reply itself is already
    /// known when the user sends: the window it will use (computed for one more
    /// message), the language, the pack. `ChatLM.prefill` evaluates that into a
    /// cacheable slot, so the suggest only pays for the incoming message and the
    /// template tail. On a re-anchor the prefix is new and lands in the other
    /// cacheable slot (LRU), leaving the current one intact — that is the
    /// ping-pong, and it falls out of the existing slot rules.
    ///
    /// Speculative by design. A second send supersedes the first; a friend burst,
    /// language switch or pack change turns it into a logged miss. No race with
    /// the suggest is possible: both run on the engine's serial queue, so a
    /// suggest that arrives early waits for the prefill and then reuses it.
    private func prewarmNextSuggest() {
        // With prefix reuse off — by setting, or for a model it's off for
        // (ChatService.prefixReuseActive) — the suggest throws the slot away
        // anyway, so a pre-warm would be pure wasted prefill.
        guard service.autoSuggestEnabled, service.prefixReuseActive, isActive else { return }
        let language = Prompts.conversationLanguage(messages, native: service.nativeLanguage)
        let adapter: String?
        switch service.suggestAdapter(for: language) {
        case .pack(let path): adapter = path
        case .loaded:         adapter = nil
        case .missing:        return   // suggest will raise the notice; nothing to warm
        }
        let start = Prompts.windowStart(messageCount: messages.count + 1,
                                        window: Prompts.suggestWindow,
                                        stride: Prompts.suggestStride)
        let prefix = Prompts.suggestPrewarmPrefix(history: messages, friendName: contactName,
                                                  language: language)
        let pending = SuggestPrewarm(windowStart: start, language: language,
                                     adapter: adapter, tokens: nil)
        suggestPrewarm = pending
        Task { [weak self] in
            guard let self else { return }
            let m = await self.service.taskEngine.prefill(user: prefix, adapterPath: adapter)
            // Only record against the pre-warm this task started — a later send
            // may already have replaced it.
            guard self.suggestPrewarm?.id == pending.id else { return }
            self.suggestPrewarm?.tokens = m.map { Int($0.promptTokenCount) }
            NSLog("[SuggestKV] prewarm window=%d..%d tokens=%d eval=%d slot=%d",
                  start, self.messages.count + 1,
                  m?.promptTokenCount ?? -1, m?.prefillTokenCount ?? -1, m?.cacheSlotIndex ?? -1)
        }
    }

    func useSuggestion(_ text: String) {
        setInput(text)
        suggestions = []
        notice = nil
        missingPack = nil
    }

    /// Write the draft PROGRAMMATICALLY — a tapped suggestion, an applied
    /// rewrite, an undo — and guarantee the input field picks it up.
    ///
    /// The field keeps its own local copy of the text (see `InputBar`), so it has
    /// to mirror this back. Observing `input` alone isn't enough: the field must
    /// ignore the `input = ""` that `send` performs, and an undo back to an EMPTY
    /// draft is indistinguishable from it by value. `inputRevision` is what
    /// separates them — it ticks only here, so the field can mirror on the tick
    /// and let a bare value change pass.
    func setInput(_ text: String) {
        input = text
        inputRevision &+= 1
    }

    /// Stop, and tell the user what to install — the response to `.missing`.
    ///
    /// Generating anyway is the one option not on the table: it would run this
    /// conversation's language through the NATIVE pack, which is the failure the
    /// per-conversation adapter exists to prevent, and it would do so invisibly.
    private func raiseMissingPack(_ pack: LoRAAdapter, language: TranslateLanguage) {
        errorMessage = nil
        suggestions = []
        notice = ChatService.missingPackNotice(pack, language: language)
        missingPack = pack
    }

    /// Auto-generate reply suggestions when it's the user's turn — gated and
    /// debounced so the on-device model isn't spent on every incoming token or on a
    /// message the user is already replying to. Safe to call often (e.g. on every
    /// keystroke): it cancels any pending run and re-evaluates the guards.
    ///
    /// **This is now the ONLY route to a reply suggestion in a 1:1 chat.** The
    /// input bar's manual suggest button was removed (it sat next to the Magic
    /// Compose wand as a second, near-identical sparkle button), which leaves two
    /// gaps a reader should know about rather than rediscover:
    ///
    ///  - **No regeneration.** `lastAutoSuggestedID` allows one generation per
    ///    incoming message. A poor set of three chips is the only set that message
    ///    will produce.
    ///  - **The Settings toggle is now an off switch for the whole feature.** With
    ///    `autoSuggestEnabled` false there is no other entry point, so suggestions
    ///    simply never appear.
    ///
    /// Both were accepted deliberately. Magic Compose's smart-reply chips cover
    /// the "give me something else" case from the wand, and the `Suggest` button
    /// on the missing-pack notice still calls `suggest()` directly. Group chat is
    /// unaffected — it has no Magic Compose, so `GroupChatView` keeps its button.
    func scheduleAutoSuggest() {
        autoSuggestTask?.cancel()

        // Fire only when: enabled, this chat is on-screen, the user isn't composing,
        // nothing else is generating, the friend spoke last (your turn), and we
        // haven't already suggested for that exact message.
        // `!isComposeOpen`: the compose bar is already proposing text for this
        // input box, and chips popping up underneath it are a second, competing
        // answer to the question the user is mid-way through answering.
        guard service.autoSuggestEnabled, isActive, input.isEmpty, !isBusy,
              !isComposeOpen, suggestions.isEmpty, unrepliedCount > 0,
              let last = messages.last(where: { !$0.text.isEmpty }),
              last.sender == .friend, last.id != lastAutoSuggestedID else { return }

        let targetID = last.id
        let debounce: Double = 1.2

        let language = Prompts.conversationLanguage(messages, native: service.nativeLanguage)
        switch service.suggestAdapter(for: language) {
        case .missing(let pack):
            // Nothing is generated for this message — the pack that would make the
            // output usable isn't installed. Raise it once and stop: marking the
            // message as handled keeps the notice from being re-raised on every
            // keystroke, and installing the pack makes the suggest button work
            // immediately, so the user is not stuck waiting for a new message.
            raiseMissingPack(pack, language: language)
            lastAutoSuggestedID = targetID
            return
        case .pack(let path):
            // Get the language pack in memory during the debounce, not during the
            // generation. Switching adapters is otherwise free, but the FIRST use of
            // a pack in a session parses ~39 MB off disk, and that lands inside the
            // suggest call — the one path with a 3 s budget. The debounce is dead
            // time anyway, and the engine queue is idle in it.
            //
            // Speculative on purpose: it prewarms even if the guards below later
            // reject the suggest. The cost of a wasted prewarm is memory that the
            // next suggest in this thread would have spent regardless.
            service.taskEngine.prewarmAdapter(path: path)
        case .loaded:
            break   // already resident — it's the pack the engine loaded with
        }

        autoSuggestTask = Task { [weak self] in
            // Let an incoming burst settle before spending a generation. This wait
            // counts against the user-visible budget, so it's handed to `suggest`.
            try? await Task.sleep(nanoseconds: UInt64(debounce * 1_000_000_000))
            guard !Task.isCancelled, let self else { return }
            // Re-check the guards — state may have moved during the debounce.
            guard self.service.autoSuggestEnabled, self.isActive, self.input.isEmpty,
                  !self.isBusy, !self.isComposeOpen, self.suggestions.isEmpty,
                  self.messages.last(where: { !$0.text.isEmpty })?.id == targetID else { return }
            self.lastAutoSuggestedID = targetID
            await self.suggest(waitedBefore: debounce)
        }
    }

    /// Cancel any pending auto-suggest (e.g. when the chat is closed).
    func cancelAutoSuggest() { autoSuggestTask?.cancel() }

    // MARK: - Feature: Magic Compose

    /// Where the compose bar is in its cycle. One enum rather than a fistful of
    /// booleans because most combinations are impossible — there is no such thing
    /// as generating-and-refused — and every one of those combinations would be a
    /// state the view has to render something for.
    enum ComposeState: Equatable {
        /// Bar hidden. The only state in which the feature is not running.
        case idle
        /// Chips shown, nothing generated yet.
        case menu
        /// A generation is in flight.
        case generating
        /// The rewrite has landed in the input box; Undo is still offered.
        case applied
        /// The model declined the instruction. Payload is the refusal reason.
        case refused(reason: String?)
        /// The generation failed or came back blank — worth retrying, unlike
        /// `.refused`.
        case failed(String)
    }

    @Published private(set) var composeState: ComposeState = .idle

    /// The draft as it stood when the bar opened. Two jobs: it is the `{DRAFT}`
    /// in the prompt, and — being empty or not — it is the ONLY thing that picks
    /// the chip set (rewrite vs smart reply).
    @Published private(set) var composeDraft = ""

    /// The chip the current preview came from, so the bar can mark it active and
    /// Retry can re-run the same instruction.
    @Published private(set) var composeInstruction: Prompts.ComposeInstruction?

    /// The draft to restore on Undo. Distinct from `composeDraft` only in intent:
    /// both hold the original, and both deliberately survive an applied rewrite,
    /// so Retry and every other chip keep acting on what the USER wrote rather
    /// than compounding rewrites of rewrites.
    private var composeUndoText = ""

    var isComposeOpen: Bool { composeState != .idle }
    var isComposeGenerating: Bool { composeState == .generating }

    /// Chips for the current draft state — rewrite when there is a draft, smart
    /// reply when there isn't.
    var composeChips: [Prompts.ComposeInstruction] {
        Prompts.composeChips(hasDraft: !composeDraft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
    }

    /// Open the compose bar over `draft`.
    ///
    /// The draft is passed IN rather than read from `input`, because `input` is
    /// not what the user is typing into — the field keeps its text locally for
    /// the latency reason documented on `InputBar`, and only pushes it up on
    /// send. Reading `input` here would compose against the previous message.
    func beginCompose(draft: String) {
        composeDraft = draft
        composeUndoText = draft
        composeInstruction = nil
        composeState = .menu
        // The compose bar and the suggestion chips both propose text for the same
        // input box; showing them together is two competing answers to one question.
        suggestions = []
        notice = nil
        cancelAutoSuggest()
    }

    /// Close the bar and leave the input box exactly as it is.
    func endCompose() {
        composeTask?.cancel()
        composeTask = nil
        composeState = .idle
        composeDraft = ""
        composeUndoText = ""
        composeInstruction = nil
        missingPack = nil
    }

    /// In-flight generation, so closing the bar (or tapping another chip) doesn't
    /// leave a stale result to land in the input box later.
    private var composeTask: Task<Void, Never>?

    /// Run one chip. Cancels any generation already in flight — tapping a second
    /// chip means the user changed their mind, and the older result is now noise.
    func compose(_ instruction: Prompts.ComposeInstruction) {
        composeTask?.cancel()
        composeTask = Task { [weak self] in
            await self?.runCompose(instruction)
        }
    }

    /// Re-run the active chip. Same instruction, same draft — the temperature is
    /// what makes the second answer different.
    func retryCompose() {
        guard let instruction = composeInstruction else { return }
        compose(instruction)
    }

    /// Put the original draft back and close.
    ///
    /// The only way back, now that a chip writes straight into the input box with
    /// no confirmation step — which is why the bar stays open on `.applied`
    /// rather than dismissing itself the moment the text lands.
    func undoCompose() {
        if case .applied = composeState { setInput(composeUndoText) }
        endCompose()
    }

    /// One generation: resolve the language, run it under that language's pack,
    /// and land the result in `composeState`.
    ///
    /// Language resolution mirrors `suggest` — one resolution feeding both the
    /// prompt's `{LANG}` and the LoRA pack the generation runs under, because
    /// deriving them separately is how they drift apart (see
    /// `ChatService.suggestAdapter`) — with one addition: when there IS a draft,
    /// the draft's own language outranks the thread's. `Prompts.composeLanguage`
    /// carries the reasoning; the short version is that rewriting an English
    /// draft under the Vietnamese pack translates it instead. With no draft this
    /// resolves identically to auto-suggest.
    private func runCompose(_ instruction: Prompts.ComposeInstruction) async {
        // Cleared up front, not just on the missing-pack branch: a pack installed
        // since the last attempt would otherwise leave a stale Get button beside
        // a result that no longer needs it.
        missingPack = nil
        let language = Prompts.composeLanguage(history: messages,
                                               draft: composeDraft,
                                               native: service.nativeLanguage)
        let adapter: String?
        switch service.suggestAdapter(for: language) {
        case .pack(let path): adapter = path
        case .loaded:         adapter = nil
        case .missing(let pack):
            // Same rule as suggest: generating under the native pack would
            // produce confident, wrong-language text in the box the user is about
            // to send from. Stop and name the install instead.
            composeInstruction = instruction
            composeState = .failed(ChatService.missingPackNotice(pack, language: language))
            missingPack = pack
            return
        }

        composeInstruction = instruction
        composeState = .generating
        NSLog("[Compose] chip=%@ draftLang=%@ convLang=%@ -> lang=%@",
              instruction.id,
              Prompts.sourceLanguage(of: composeDraft)?.rawValue ?? "<undetermined>",
              Prompts.conversationLanguage(messages, native: service.nativeLanguage).rawValue,
              language.rawValue)

        let turn = SLMTurnTimer("compose[\(instruction.id)]")
        let prompt = Prompts.magicCompose(history: messages,
                                          instruction: instruction,
                                          language: language,
                                          draft: composeDraft,
                                          friendName: contactName)
        // print (not NSLog): os_log truncates long strings, cropping the prompt.
        print("[Compose] SLM input prompt (\(instruction.id) with \(contactName)):\n\(prompt)\n[Compose] --- end prompt ---")

        let raw = await service.taskEngine.complete(system: nil, history: [], user: prompt,
                                                    maxTokens: service.composeTokens,
                                                    temperature: service.composeTemp,
                                                    label: .compose,
                                                    adapterPath: adapter)
        guard !Task.isCancelled else { return }

        let outcome: Prompts.ComposeOutcome
        if let raw {
            let escaped = raw.replacingOccurrences(of: "\n", with: "\\n")
            NSLog("[Compose] raw SLM output: %@", escaped)
            outcome = Prompts.parseCompose(raw)
        } else {
            NSLog("[Compose] SLM returned nil (engine error)")
            outcome = .empty
        }

        switch outcome {
        case .text(let text):
            // Straight into the input box — no preview, no Replace tap. The
            // ORIGINAL draft is what `composeUndoText` holds and what every chip
            // rewrites (see `composeDraft`), so tapping a second chip re-rewrites
            // the user's own words rather than compounding on the first result.
            setInput(text)
            composeState = .applied
        case .refused(let reason):
            NSLog("[Compose] refused reason=%@", reason ?? "<nil>")
            composeState = .refused(reason: reason)
        case .empty:
            composeState = .failed("Couldn\u{2019}t rewrite that — try again.")
        }

        let start = Prompts.windowStart(messageCount: messages.count,
                                        window: Prompts.suggestWindow)
        // `lang` is the resolution BOTH the prompt and the adapter came from, so a
        // wrong-language rewrite can be traced to the detection in one line rather
        // than inferred from the output. Pair it with the `[LoRA] suggest lang=…`
        // line `suggestAdapter` logs to see which pack it actually resolved to.
        turn.finish("chip=\(instruction.id) mode=\(instruction.mode) "
                  + "lang=\(language.rawValue) "
                  + "draft=\(composeDraft.count) chars prompt=\(prompt.count) chars "
                  + "window=\(start)..\(messages.count)")
    }

    // MARK: - Feature: Summarization

    func summarizeUnread() async {
        guard !isSummarizing else { return }
        // Summarize ONLY the un-replied messages, but hand the model a short window of
        // preceding messages (BOTH sides) as background so it can interpret them —
        // e.g. the friend's burst references a trip you asked about. The context is
        // NOT summarized, so the result reflects what you actually missed.
        // The unreplied region starts right after your last reply — NOT from the
        // read flag, which summarizing clears. This keeps every summarize call
        // scoped to ONLY the messages you haven't replied to, even on a repeat tap.
        let boundary = (messages.lastIndex(where: { $0.sender == .you }) ?? -1) + 1
        let unread = messages[boundary...].filter { $0.sender == .friend && !$0.text.isEmpty }
        guard !unread.isEmpty else { return }

        let ctxStart = max(0, boundary - Prompts.unreadContextWindow)
        let context = messages[ctxStart..<boundary].filter { !$0.text.isEmpty }
        // Context + unread, chunked when the transcript is too long for one call.
        // The context messages lead so the model reads them as background before the
        // messages actually being summarized.
        isSummarizing = true
        errorMessage = nil
        summary = ""
        summaryProgress = nil
        showSummary = true      // present the sheet now; it streams in live
        for await event in service.summarizeConversationStream(Array(context) + Array(unread),
                                                               friendName: contactName,
                                                               native: service.nativeLanguage) {
            switch event {
            case .mapping(let done, let total): summaryProgress = "Reading part \(done) of \(total)…"
            case .combining:                    summaryProgress = "Combining…"
            case .text(let partial):            summaryProgress = nil; summary = partial
            }
        }
        summaryProgress = nil
        if (summary ?? "").isEmpty { summary = "(summary failed)" }
        markFriendMessagesRead()
        isSummarizing = false
    }

    // MARK: - Feature: Per-message summary (long messages)

    /// The long message whose summary sheet is currently presented (`nil` = none).
    /// Set by `summarizeMessage`; the chat screen presents a sheet bound to it,
    /// exactly like the whole-conversation summary — not an inline blurb.
    @Published var summaryMessage: ChatMessage?

    /// Summarize ONE long message on demand (from its long-press menu) and present
    /// it in a sheet. The heavy lifting is a chunked map-reduce in
    /// `ChatService.summarizeLongStream`, so a message far longer than the small model's
    /// context still summarizes. Not cached — every tap clears the previous summary
    /// and recomputes so the `[Summarize]` debug logs run each time.
    func summarizeMessage(_ message: ChatMessage) async {
        guard !message.text.isEmpty else { return }
        summaryMessage = message   // present the sheet (shows a spinner while it runs)
        if message.isSummarizingLong { return }
        message.longSummary = nil
        message.isSummarizingLong = true
        let speaker = message.sender == .you ? Prompts.userName : (message.authorName ?? contactName)
        let lang = service.nativeLanguage.name
        message.summaryProgress = nil

        // One path for every length. A short message is simply the zero-fold case;
        // a long one maps, folds, and streams the final pass. Progress covers the
        // steps whose output the user never sees.
        var gotText = false
        for await event in service.summarizeLongStream(message.text,
                                                       language: lang, speaker: speaker) {
            switch event {
            case .mapping(let done, let total):
                message.summaryProgress = "Reading part \(done) of \(total)…"
            case .combining:
                message.summaryProgress = "Combining…"
            case .text(let partial):
                gotText = true
                message.summaryProgress = nil
                message.longSummary = partial
            }
        }
        if !gotText || (message.longSummary ?? "").isEmpty {
            message.longSummary = "(summary failed)"
        }
        message.summaryProgress = nil
        message.isSummarizingLong = false
        persist()
    }

    // MARK: - Feature: Event detection

    /// Run the on-device task model over one message to see if it describes an
    /// event; if so, surface it as a reminder banner. Gated by a cheap heuristic so
    /// ordinary chatter never touches the model. Fire-and-forget from the callers.
    func detectEvent(in message: ChatMessage) async {
        let text = message.text
        guard Prompts.mightMentionEvent(text) else { return }
        let prompt = Prompts.extractEvent(from: text, language: service.nativeLanguage)
        NSLog("[Event] input prompt:\n%@", prompt)
        let raw = await service.taskEngine.complete(system: nil, history: [],
                                                    user: prompt,
                                                    maxTokens: service.eventTokens,
                                                    temperature: service.eventTemp,
                                                    label: .event)
        NSLog("[Event] raw SLM output: %@", raw ?? "<nil>")
        guard let raw,
              let event = DetectedEvent.parse(raw),
              event.hasEvent else { return }
        NSLog("[Event] parsed: intent=%@ name=%@ date=%@ time=%@ location=%@",
              event.intent, event.eventName, event.date, event.time, event.location)
        // Concreteness gate: farewell pleasantries ("gặp lại" + "sáng mai", no time/
        // place) are false alarms — don't raise the banner for them.
        guard event.isActionable else {
            NSLog("[Event] suppressed — low-confidence / false alarm")
            return
        }
        pendingEvent = event
    }

    func dismissEvent() {
        pendingEvent = nil
    }

    // MARK: - Feature: Safety guardrail

    /// Distinct unsafe classes currently present in the thread (drives the banner).
    var flaggedLabels: [SafetyLabel] {
        var seen = Set<Int>()
        var out: [SafetyLabel] = []
        for m in messages {
            if let l = m.safety, l.isUnsafe, seen.insert(l.rawValue).inserted {
                out.append(l)
            }
        }
        return out
    }

    /// Distinct unsafe classes present, each paired with the highest severity seen
    /// for it — so the banner can show e.g. "Child safety · Critical".
    ///
    /// Severity is optional and stays optional: since v16 only rule hits have one,
    /// so a class detected solely by the model contributes `nil` and the banner
    /// names the class alone. This used to fold nil to `.safe`, which read as an
    /// assessment ("we looked, it's fine") on a message that had just been
    /// flagged — and would have tinted the banner as though nothing serious were
    /// present.
    var flaggedItems: [(label: SafetyLabel, severity: Severity?)] {
        var maxSeverity: [SafetyLabel: Severity] = [:]
        var order: [SafetyLabel] = []
        for m in messages {
            guard let l = m.safety, l.isUnsafe else { continue }
            if !order.contains(l) { order.append(l) }
            // Only a real severity can raise the maximum. A nil leaves whatever
            // an earlier rule hit on the same class established.
            if let sev = m.severity, sev.rawValue > (maxSeverity[l]?.rawValue ?? -1) {
                maxSeverity[l] = sev
            }
        }
        return order.map { ($0, maxSeverity[$0]) }
    }

    /// Classify ONE message (incoming or outgoing) with the on-device guardrail
    /// model (BERT-Tiny via `GuardEngine`). Event-driven: called per new message.
    /// The label + severity are cached on the message and persisted, so a message
    /// is only ever classified once. Fire-and-forget.
    /// Which guardrail pack screens this message.
    ///
    /// The message's own language when it can be told, otherwise the thread's.
    /// That fallback is the point: chat turns are short ("ok", "haha", "?"), and
    /// `NLLanguageRecognizer` abstains on most of them. Detection is what's
    /// unreliable on a short message — not the screening, which still runs. The
    /// shortest violations in the rule set are six characters, so skipping short
    /// text would skip exactly what the guardrail is for.
    private func guardLanguage(for message: ChatMessage) -> TranslateLanguage {
        message.detectedLanguage
            ?? Prompts.conversationLanguage(messages, native: service.nativeLanguage)
    }

    /// Screen a message that just arrived while the user could be looking.
    ///
    /// The verdict lands ~7–25 ms after the bubble renders, which is a frame or
    /// two — imperceptible. That only holds for messages arriving one at a time
    /// at human speed; a bulk backlog goes through `screenFromViewport` instead.
    func classifySafety(in message: ChatMessage) async {
        // Through the screener, not around it. The two paths overlap constantly:
        // a message arriving while the chat is open is classified here AND asked
        // for by its row's `onAppear` moments later, and both would pass
        // `runScreening`'s own `safetyLabel == nil` check because that runs before
        // the await. The screener holds the only dedupe gate.
        await screener.screenNow(message)
    }

    // MARK: - Guardrail demo

    /// True while a scripted guardrail scenario is playing.
    @Published private(set) var isPlayingDemo = false

    /// Play a scripted scenario one message at a time, screening each.
    ///
    /// Runs through the LIVE path (`classifySafety`), not the viewport one, so
    /// each verdict lands as its bubble appears — that's what makes the log
    /// readable as a sequence rather than a burst.
    ///
    /// The pause between lines is the point: the guardrail itself takes ~7–25 ms,
    /// far too fast to watch. This paces it to conversation speed so the verdict
    /// for each line can be read before the next arrives.
    ///
    /// Paced and typed like a real reply, not dealt out on a metronome. A friend
    /// line waits a beat, raises the same "…is typing" indicator `replyFromFriend`
    /// does, holds it for as long as the text would take to thumb-type, and only
    /// then appears — because a fixed gap with nothing in it is what made this
    /// read as a log dump. Your own lines skip the indicator: nobody else is
    /// typing them.
    ///
    /// `pace` is retained as an override for anything that wants the old flat
    /// rhythm; the default of nil means length-derived.
    func playGuardrailDemo(_ lines: [GuardrailDemo.Line],
                           named name: String,
                           pace: UInt64? = nil) async {
        guard !isPlayingDemo else { return }
        isPlayingDemo = true
        // The indicator is the demo's for the duration. Clear it on every exit —
        // a cancelled Task would otherwise strand "Alex is typing…" on screen and
        // leave the composer disabled with no way back.
        defer { isPlayingDemo = false; isBotTyping = false }

        NSLog("[GuardDemo] ===== %@ — %d lines =====", name, lines.count)

        var timings: [Double] = []

        for (i, line) in lines.enumerated() {
            // The first line lands immediately; the wait belongs BETWEEN messages.
            if i > 0 {
                let step = GuardrailDemo.pacing(for: line)
                try? await Task.sleep(nanoseconds: pace ?? step.read)
                if pace == nil && line.sender == .friend {
                    isBotTyping = true
                    try? await Task.sleep(nanoseconds: step.typing)
                    isBotTyping = false
                }
            }

            let msg = ChatMessage(sender: line.sender, text: line.text,
                                  isRead: line.sender == .you)
            messages.append(msg)
            persist()

            // Await the verdict rather than firing and forgetting, so the log
            // reads in order and the expectation sits next to its result.
            // WALL CLOCK, and deliberately so: it includes the wait to get back
            // onto the main actor after the verdict is ready. That wait is often
            // the larger half — appending a message re-lays out the list, and a
            // Violation/Critical verdict makes SwiftUI rasterize a blur. Compare
            // with the `[Guard] … Nms` line for the same message to separate
            // them: that one times only the classifier. A large gap between the
            // two is UI cost, not guardrail cost.
            let started = DispatchTime.now().uptimeNanoseconds
            await classifySafety(in: msg)
            let ms = Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000
            timings.append(ms)

            let got = msg.safety.map {
                "\($0.displayName)/\(msg.severity?.displayName ?? "—")"
            } ?? "not classified"
            NSLog("[GuardDemo] %2d/%d  %6.1fms  %@\n            expected: %@\n            got:      %@",
                  i + 1, lines.count, ms,
                  line.text.count > 60 ? String(line.text.prefix(60)) + "…" : line.text,
                  line.expect, got)
        }

        // The first line of a run carries one-time cost the rest don't — the
        // language pack loads on first use (~50 ms warm, ~320–500 ms cold) and
        // TFLite's first invoke on a fresh interpreter is ~100 ms. Reporting the
        // median alongside the mean keeps that one outlier from being read as
        // the per-message price.
        if !timings.isEmpty {
            let sorted = timings.sorted()
            let median = sorted[sorted.count / 2]
            let mean = timings.reduce(0, +) / Double(timings.count)
            NSLog("[GuardDemo] wall clock (incl. main-actor wait): first %.1fms · "
                + "median %.1fms · mean %.1fms · slowest %.1fms — compare with [Guard] "
                + "for classifier-only time", timings[0], median, mean, sorted.last ?? 0)
        }
        NSLog("[GuardDemo] ===== %@ complete =====", name)
    }

    // MARK: - Viewport screening (bulk history)

    /// Bidirectional look-ahead around the visible row. See `MessageScreener`.
    static let screenLookAhead = MessageScreener.lookAhead

    /// Ordering, look-ahead and shedding all live in the screener — see that type
    /// for why a task-per-message drained FIFO was wrong under a fast scroll.
    private lazy var screener = MessageScreener { [weak self] message, priority in
        await self?.runScreening(message, persist: priority == .immediate ? .immediately
                                                                          : .coalesced)
    }

    /// Classifications actually run. Test seam.
    var screeningRuns: Int { screener.runs }

    /// Messages queued for screening but not finished — the real backlog under a
    /// flood, one layer above `GuardEngine.backlogDepth`.
    var pendingScreenings: Int { screener.pendingCount }

    /// Messages dropped to keep the queue bounded. Re-requested if scrolled back to.
    var shedScreenings: Int { screener.shed }

    /// The recipient reported a Critical message.
    ///
    /// Persisted immediately rather than coalesced: unlike a screening verdict,
    /// which the app can always recompute, a report is a decision the user made
    /// once. Losing it to a crash inside the 500 ms debounce window would put the
    /// button back with no trace that they had already acted.
    func report(_ message: ChatMessage, reason: ReportReason, note: String = "") {
        GuardReport.submit(message, reason: reason, note: note,
                           author: contactName, context: "DM with \(contactName)")
        persist()
    }

    /// A row scrolled into view.
    func screenFromViewport(_ message: ChatMessage) {
        guard let idx = messages.firstIndex(where: { $0.id == message.id }) else { return }
        screener.viewportDidShow(index: idx, in: messages)
    }

    // MARK: - Screening internals

    private enum ScreenPersist {
        /// Write straight away — a single message the user is watching.
        case immediately
        /// Batch it. A look-ahead sweep lands several verdicts in a few hundred
        /// ms, and `persist` re-encodes and rewrites EVERY conversation.
        case coalesced
    }

    private func runScreening(_ message: ChatMessage, persist mode: ScreenPersist) async {
        guard message.safetyLabel == nil, !message.text.isEmpty else { return }
        let result = await service.classify(message.text,
                                            language: guardLanguage(for: message))

        // Everything from here runs on the MAIN actor. Measured at ~1.5 ms even on
        // a 1000-message thread (`ScreeningTailTests`), so it is NOT the cost it
        // looks like it should be — `persist` re-encodes every conversation, which
        // sounds expensive and isn't at these sizes.
        //
        // Kept instrumented anyway, because it is the obvious suspect whenever a
        // verdict appears slow and ruling it out quickly is worth one comparison.
        // When wall-clock time for a classification far exceeds the `[Guard]`
        // figure and this line does NOT fire, the gap is main-actor contention —
        // the continuation waiting for a main thread that SwiftUI is busy with.
        let tailStart = DispatchTime.now().uptimeNanoseconds
        message.apply(result)
        let applied = DispatchTime.now().uptimeNanoseconds
        switch mode {
        case .immediately: persist()
        case .coalesced:   persistCoalesced()
        }
        let done = DispatchTime.now().uptimeNanoseconds

        let applyMs = Double(applied - tailStart) / 1_000_000
        let persistMs = Double(done - applied) / 1_000_000
        if applyMs + persistMs > 16 {       // one 60 Hz frame
            NSLog("[GuardPerf] main-actor tail %.1fms (apply %.1fms, persist %.1fms) — "
                + "%d messages in thread", applyMs + persistMs, applyMs, persistMs,
                  messages.count)
        }
    }

    private var coalescedPersist: Task<Void, Never>?

    private func persistCoalesced() {
        coalescedPersist?.cancel()
        coalescedPersist = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 500_000_000)
            guard !Task.isCancelled else { return }
            self?.persist()
        }
    }

    /// Mock (in-app only) reminder: clear the banner and flash a confirmation that
    /// auto-dismisses after a few seconds.
    func confirmReminder() {
        guard let event = pendingEvent else { return }
        let name = event.eventName.isEmpty ? "this event" : event.eventName
        pendingEvent = nil
        if let start = event.resolvedStart {
            let df = DateFormatter()
            df.dateFormat = "EEE, MMM d · h:mm a"
            eventReminderNote = "Reminder set for \(name) — \(df.string(from: start))"
        } else {
            eventReminderNote = "Reminder set for \(name)"
        }
        Task { [weak self] in
            try? await Task.sleep(nanoseconds: 4_000_000_000)
            self?.eventReminderNote = nil
        }
    }

    // MARK: - Read state

    func markFriendMessagesRead() {
        var changed = false
        for m in messages where m.sender == .friend && !m.isRead {
            m.isRead = true
            changed = true
        }
        if changed { persist() }
    }

    /// Mark the conversation unread from the inbox. Flips the friend messages since
    /// your last reply back to unread so the badge reflects what you'd catch up on;
    /// if you've already replied to everything, just the last friend message.
    func markUnread() {
        let boundary = (messages.lastIndex(where: { $0.sender == .you }) ?? -1) + 1
        var burst = messages[boundary...].filter { $0.sender == .friend }
        if burst.isEmpty, let last = messages.last(where: { $0.sender == .friend }) {
            burst = [last]
        }
        var changed = false
        for m in burst where m.isRead {
            m.isRead = false
            changed = true
        }
        if changed {
            // Mutating a child message's @Published in place doesn't fire our own
            // objectWillChange, so nudge it to refresh the inbox row's badge.
            objectWillChange.send()
            persist()
        }
    }
}

/// Identity-based `Hashable` so a `Conversation` can be a `NavigationStack` path
/// value. Keyed on the immutable `id`, so the methods are safely `nonisolated`
/// despite the type being `@MainActor`.
extension Conversation: Hashable {
    nonisolated static func == (lhs: Conversation, rhs: Conversation) -> Bool {
        lhs.id == rhs.id
    }
    nonisolated func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
}

/// Mutable reference box for accumulating streamed tokens, so the streaming
/// callback can append without capturing an inout var.
private final class RawBuffer {
    var text = ""
}
