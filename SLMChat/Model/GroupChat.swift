import Foundation

/// A group chat: the user plus three chatbots who talk among themselves. An
/// autonomous loop (toggled by `/start` / `/stop`) has a randomly-chosen bot post a
/// message every `tickSeconds`, so it feels like a live group without spamming.
///
/// Long-lived (owned by the store), so the loop keeps running even when the user
/// navigates away — same pattern as `Conversation`.
@MainActor
final class GroupChat: ObservableObject {

    let title = "Group"
    /// The three bots. Personas come from `Prompts.persona(for:)`.
    let botNames = ["Mai", "Sam", "Linh"]

    @Published var messages: [ChatMessage] = []
    @Published var input: String = ""
    @Published var isRunning = false      // autonomous loop active
    @Published var typingBot: String?     // name of the bot currently "typing"
    @Published var errorMessage: String?

    // Summarization
    @Published var summary: String?
    @Published var showSummary = false
    /// See `Conversation.summaryProgress`.
    @Published var summaryProgress: String?
    @Published var isSummarizing = false
    // Auto-reply suggestions
    @Published var suggestions: [String] = []
    @Published var isSuggesting = false
    @Published var notice: String?
    /// See `Conversation.missingPack` — the pack blocking suggestions here, if any.
    @Published var missingPack: LoRAAdapter?

    var isActive = false

    /// Seconds between autonomous bot messages (spam guard).
    static let tickSeconds: UInt64 = 5
    /// Delay before a bot replies to one of the user's own messages.
    static let directReplyDelay: UInt64 = 3

    private unowned let service: ChatService
    private var isGenerating = false
    private var lastAuthor: String?
    private var loopTask: Task<Void, Never>?

    init(service: ChatService) {
        self.service = service
    }

    var isBusy: Bool { typingBot != nil || isSummarizing || isSuggesting }

    /// Bot messages that arrived while the user was away from the group screen.
    var unreadCount: Int { messages.filter { $0.sender == .friend && !$0.isRead }.count }

    // MARK: - Input / commands

    /// Handle the input field: `/start` and `/stop` toggle the loop; anything else is
    /// sent as a message.
    func handleInput(_ rawText: String) {
        let text = rawText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        input = ""
        switch text.lowercased() {
        case "/start": start()
        case "/stop":  stop()
        default:       send(text)
        }
    }

    private func send(_ text: String) {
        suggestions = []
        notice = nil
        markGroupRead()
        let mine = ChatMessage(sender: .you, text: text)
        messages.append(mine)
        Task { await classifySafety(in: mine) }
        // A random bot replies to me shortly after (independent of the tick loop).
        Task {
            try? await Task.sleep(nanoseconds: Self.directReplyDelay * 1_000_000_000)
            await postRandomBotMessage()
        }
    }

    // MARK: - Autonomous loop

    func start() {
        guard !isRunning else { return }
        isRunning = true
        // Task created in a @MainActor context is main-actor-isolated, so it touches
        // `isRunning` / `postRandomBotMessage()` directly. Strong self is fine — the
        // store owns the group for the app's lifetime and `stop()` cancels the task.
        loopTask = Task {
            while isRunning && !Task.isCancelled {
                try? await Task.sleep(nanoseconds: Self.tickSeconds * 1_000_000_000)
                if !isRunning || Task.isCancelled { break }
                await postRandomBotMessage()
            }
        }
    }

    func stop() {
        isRunning = false
        loopTask?.cancel()
        loopTask = nil
    }

    // MARK: - Safety screening

    /// The group's messages were NOT screened at all until this existed — neither
    /// what the user sent nor what the bots posted. Worse than an obvious gap:
    /// `GroupChatView` renders through the same `MessageBubble` as a 1:1 chat, so
    /// unscreened messages looked exactly like messages that had been screened and
    /// came back clean. Nothing in the UI could tell them apart.
    ///
    /// This mirrors `Conversation`'s screening, with two differences the group
    /// forces:
    ///  - No `persist()`. The group is in-memory only, so a verdict lives as long
    ///    as the app run and there is nothing to coalesce writes for.
    ///  - Verdicts are attributable to an AUTHOR (`ChatMessage.authorName`), not
    ///    just a side. That is the group's whole point for a guardrail: in a 1:1
    ///    chat "the other person" is unambiguous, while a group needs to know
    ///    WHICH member sent the flagged message.
    /// Bidirectional look-ahead around the visible row. See `MessageScreener`.
    static let screenLookAhead = MessageScreener.lookAhead

    /// Same scheduler as `Conversation` — nearest-to-viewport first, shed the
    /// far work. Identical logic, and too easy to get subtly wrong twice.
    private lazy var screener = MessageScreener { [weak self] message, _ in
        await self?.runScreening(message)
    }

    /// Classifications actually run. Test seam.
    var screeningRuns: Int { screener.runs }

    /// Messages queued for screening but not finished — the real backlog under a
    /// flood, one layer above `GuardEngine.backlogDepth`.
    var pendingScreenings: Int { screener.pendingCount }

    /// Messages dropped to keep the queue bounded. Re-requested if scrolled back to.
    var shedScreenings: Int { screener.shed }

    /// Screen a message that just arrived while the user could be looking.
    func classifySafety(in message: ChatMessage) async {
        await screener.screenNow(message)
    }

    /// The recipient reported a Critical message.
    ///
    /// The author name is the whole difference from a DM: in a group "the other
    /// person" is ambiguous, so a report that cannot name WHO is not actionable.
    /// This is the field a per-user violation signal would be built from.
    func report(_ message: ChatMessage, reason: ReportReason, note: String = "") {
        GuardReport.submit(message, reason: reason, note: note,
                           author: message.authorName ?? "Unknown member",
                           context: "Group: \(title)")
    }

    /// A row scrolled into view.
    func screenFromViewport(_ message: ChatMessage) {
        guard let idx = messages.firstIndex(where: { $0.id == message.id }) else { return }
        screener.viewportDidShow(index: idx, in: messages)
    }

    private func runScreening(_ message: ChatMessage) async {
        guard message.safetyLabel == nil, !message.text.isEmpty else { return }
        // The group's language fallback is weaker than a 1:1 chat's: a mixed
        // group falls back to the THREAD's dominant language, so a short message
        // from a member writing in another language is screened by the wrong
        // pack. Falling back to that AUTHOR's recent messages would be better.
        let language = message.detectedLanguage
            ?? Prompts.conversationLanguage(messages, native: service.nativeLanguage)
        let result = await service.classify(message.text, language: language)
        message.apply(result)
    }

    // MARK: - Guardrail demo

    @Published private(set) var isPlayingDemo = false

    /// Play a scripted group scenario, screening each message as it lands.
    ///
    /// Ends with a per-AUTHOR tally, which is the part a 1:1 demo can't show and
    /// the part that matters for what gets built next: a per-user violation
    /// signal is exactly this tally, accumulated. Seeing it printed makes two
    /// things concrete that are easy to wave past in design —
    ///
    ///  - the flagged count lands on one member while the bystanders stay at 0,
    ///    which is the signal working; and
    ///  - in the forwarded-scam scenario it lands on someone who was WARNING the
    ///    group, because the classifier has no notion of intent. That member
    ///    would accrue a strike under a naive count.
    ///
    /// Paced like the 1:1 one, with the indicator naming whoever is about to
    /// speak — `typingBot` already carries a name, so a pile-on shows the members
    /// taking turns rather than messages materialising on a metronome. That
    /// turn-taking IS the scenario in the group cases.
    func playGuardrailDemo(_ lines: [GuardrailDemo.Line],
                           named name: String,
                           pace: UInt64? = nil) async {
        guard !isPlayingDemo else { return }
        isPlayingDemo = true
        stop()                    // the tick loop would interleave random bot posts
        // `stop()` clears `typingBot`, but an early exit mid-line would not —
        // clear it here too so no member is left permanently typing.
        defer { isPlayingDemo = false; typingBot = nil }

        NSLog("[GuardDemo] ===== GROUP: %@ — %d lines =====", name, lines.count)

        var flagged: [String: Int] = [:]
        var totals: [String: Int] = [:]
        var timings: [Double] = []

        for (i, line) in lines.enumerated() {
            if i > 0 {
                let step = GuardrailDemo.pacing(for: line)
                try? await Task.sleep(nanoseconds: pace ?? step.read)
                // Only a named member can be shown typing — `line.author == nil`
                // is the user's own line, and "You are typing…" is nonsense.
                if pace == nil, let author = line.author {
                    typingBot = author
                    try? await Task.sleep(nanoseconds: step.typing)
                    typingBot = nil
                }
            }

            let msg = ChatMessage(sender: line.sender, text: line.text,
                                  isRead: true, authorName: line.author)
            messages.append(msg)

            let started = DispatchTime.now().uptimeNanoseconds
            await classifySafety(in: msg)
            timings.append(Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000)

            let who = line.author ?? "You"
            totals[who, default: 0] += 1
            if msg.safety?.isUnsafe == true { flagged[who, default: 0] += 1 }

            let got = msg.safety.map {
                "\($0.displayName)/\(msg.severity?.displayName ?? "—")"
            } ?? "not classified"
            NSLog("[GuardDemo] %2d/%d %7.1fms  %@: %@\n            expected: %@\n            got:      %@",
                  i + 1, lines.count, timings[i], who,
                  line.text.count > 55 ? String(line.text.prefix(55)) + "…" : line.text,
                  line.expect, got)
        }

        NSLog("[GuardDemo] ---- per-author tally ----")
        for who in totals.keys.sorted() {
            NSLog("[GuardDemo]   %-6@ %d flagged of %d",
                  who as NSString, flagged[who] ?? 0, totals[who] ?? 0)
        }
        if !timings.isEmpty {
            let sorted = timings.sorted()
            NSLog("[GuardDemo] timing: first %.1fms · median %.1fms · slowest %.1fms",
                  timings[0], sorted[sorted.count / 2], sorted.last ?? 0)
        }
        NSLog("[GuardDemo] ===== GROUP: %@ complete =====", name)
    }

    /// Dump `count` messages into the group at once, UNSCREENED, the way a
    /// community backlog actually arrives.
    ///
    /// No pacing and no classification here on purpose. This is the bulk path:
    /// the messages land, and nothing is screened until the user scrolls and
    /// `screenFromViewport` asks for them. That is the whole design under test —
    /// a flood should cost nothing until someone looks at it, and then cost only
    /// what they look at.
    ///
    /// Scroll the group after calling this and watch `[GuardEngine] … behind` in
    /// the log: the depth number is how far the screener is trailing your thumb.
    func floodGroup(count: Int = 120) {
        let lines = GuardrailDemo.flood(count: count)
        for line in lines {
            messages.append(ChatMessage(sender: line.sender, text: line.text,
                                        isRead: false, authorName: line.author))
        }
        NSLog("[GuardDemo] flooded %d messages, %d unscreened — scroll to screen them",
              count, messages.filter { $0.safetyLabel == nil }.count)
    }

    // MARK: - Bot reply

    /// Pick a random bot (never the same one twice in a row) and have it post a reply
    /// based on the current transcript. Serialized by `isGenerating`.
    private func postRandomBotMessage() async {
        guard !isGenerating else { return }
        let candidates = botNames.filter { $0 != lastAuthor }
        guard let bot = (candidates.isEmpty ? botNames : candidates).randomElement() else { return }

        isGenerating = true
        typingBot = bot
        errorMessage = nil

        let others = botNames.filter { $0 != bot }
        let system = Prompts.groupSystem(bot: bot, others: others)
        let transcript = Prompts.groupTranscript(messages)

        // Both flags must be cleared here: `isGenerating` gates every later call,
        // so leaving it set would silence the group permanently rather than for
        // this one turn.
        guard let chatAPI = service.chatAPI else {
            errorMessage = ChatService.missingAPIMessage
            typingBot = nil
            isGenerating = false
            return
        }
        let reply = await chatAPI.reply(persona: system,
                                        history: [ChatMessage(sender: .you, text: transcript)])

        if let reply, !Prompts.clean(reply).isEmpty {
            // Read if the user is looking, unread if they've navigated away.
            let posted = ChatMessage(sender: .friend, text: Prompts.clean(reply),
                                     isRead: isActive, authorName: bot)
            messages.append(posted)
            Task { await classifySafety(in: posted) }
            lastAuthor = bot
        } else if reply == nil {
            errorMessage = service.chatAPI?.lastError ?? "A bot didn't reply — try again."
        }

        typingBot = nil
        isGenerating = false
    }

    // MARK: - Read state / summarization

    func markGroupRead() {
        for m in messages where m.sender == .friend && !m.isRead { m.isRead = true }
    }

    /// Summarize what the user missed (unread bot messages), reusing the on-device
    /// summarization task. Falls back to the recent thread if nothing is unread.
    func summarizeUnread() async {
        guard !isSummarizing else { return }
        // Summarize ONLY the missed (unread) messages, with a short window of preceding
        // messages as background so the model can interpret them — the context itself
        // is not summarized. renderNamed uses each bot's authorName.
        // Unread plus a short lead-in of context, or the recent thread when nothing
        // is unread. The service bounds and chunks this, so an accumulated backlog
        // no longer builds a prompt too large for one call.
        // Nothing unread means nothing to catch up on, so this does nothing. It used
        // to fall back to summarizing the last 20 messages — that is a different
        // feature (summarize the whole conversation) wearing this one's button, and
        // it has been removed.
        guard let firstUnread = messages.firstIndex(where: { $0.sender == .friend && !$0.isRead })
        else { return }
        let unread = messages[firstUnread...].filter { !$0.text.isEmpty }
        guard !unread.isEmpty else { return }
        let ctxStart = max(0, firstUnread - Prompts.unreadContextWindow)
        let context = messages[ctxStart..<firstUnread].filter { !$0.text.isEmpty }
        let sourceMessages = Array(context) + Array(unread)

        isSummarizing = true
        errorMessage = nil
        summary = ""
        summaryProgress = nil
        showSummary = true      // present the sheet now; it streams in live
        for await event in service.summarizeConversationStream(sourceMessages,
                                                               friendName: Prompts.friendName,
                                                               native: service.nativeLanguage) {
            switch event {
            case .mapping(let done, let total): summaryProgress = "Reading part \(done) of \(total)…"
            case .combining:                    summaryProgress = "Combining…"
            case .text(let partial):            summaryProgress = nil; summary = partial
            }
        }
        summaryProgress = nil
        if (summary ?? "").isEmpty { summary = "(summary failed)" }
        markGroupRead()
        isSummarizing = false
    }

    // MARK: - Per-message summary (long messages)

    /// The long message whose summary sheet is currently presented (`nil` = none).
    @Published var summaryMessage: ChatMessage?

    /// Summarize ONE long group message on demand (from its long-press menu) and
    /// present it in a sheet, via the chunked map-reduce in
    /// `ChatService.summarizeLongStream`. Not cached — every tap recomputes so the
    /// `[Summarize]` debug logs run each time.
    func summarizeMessage(_ message: ChatMessage) async {
        guard !message.text.isEmpty else { return }
        summaryMessage = message
        if message.isSummarizingLong { return }
        message.longSummary = nil
        message.isSummarizingLong = true
        let speaker = message.sender == .you ? Prompts.userName : (message.authorName ?? Prompts.friendName)
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
    }

    // MARK: - Translation

    func translate(_ message: ChatMessage, force: Bool = false) async {
        if !force, message.translation != nil {
            message.showTranslation.toggle()
            return
        }
        guard !message.isTranslating else { return }
        if force { message.translation = nil }
        // Over the on-device maximum: say so rather than translate part of it.
        if service.isTooLongToTranslate(message.text) {
            message.translation = ChatService.tooLongToTranslateNote
            message.showTranslation = true
            return
        }
        message.isTranslating = true
        let target = Prompts.autoTarget(for: message.text, native: service.nativeLanguage)
        // See Conversation.translate — same flow: reveal the row first, then stream
        // the whole-translation-so-far into it.
        message.translation = ""
        message.showTranslation = true

        var received = false
        for await partial in service.translateLongStream(message.text, to: target) {
            received = true
            message.translation = partial
        }
        if !received { message.translation = "(translation failed)" }
        message.isTranslating = false
    }

    // MARK: - Auto-reply suggestions

    /// Suggest a few short replies the user could send into the group next.
    func suggest() async {
        guard !isSuggesting else { return }

        // Same pairing as the 1:1 path: one resolved language drives both the
        // prompt's target and the LoRA pack the generation runs under, and an
        // uninstalled pack stops the turn instead of quietly running the native one.
        let language = Prompts.conversationLanguage(messages, native: service.nativeLanguage)
        let adapter: String?
        switch service.suggestAdapter(for: language) {
        case .pack(let path): adapter = path
        case .loaded:         adapter = nil
        case .missing(let pack):
            errorMessage = nil
            suggestions = []
            notice = ChatService.missingPackNotice(pack, language: language)
            missingPack = pack
            return
        }

        isSuggesting = true
        errorMessage = nil
        notice = nil
        missingPack = nil
        suggestions = []

        let turn = SLMTurnTimer("suggest[group]")
        let prompt = Prompts.suggestReplies(history: messages,
                                            language: language)  // renderHandles uses authorName
        // print (not NSLog): os_log truncates long strings, cropping the prompt.
        print("[Suggest] SLM input prompt (group \"\(title)\"):\n\(prompt)\n[Suggest] --- end prompt ---")
        var result = Prompts.SuggestionResult(suggestions: [], refused: false, reason: nil)
        var attempts = 0

        // Sample up to 3× — a single generation can come back empty or malformed.
        // Each retry is a full extra generation; the turn log reports how many ran.
        for attempt in 0..<3 {
            attempts = attempt + 1
            let (out, metrics) = await service.taskEngine.completeMeasured(
                system: nil, history: [], user: prompt,
                maxTokens: service.suggestTokens, temperature: service.suggestTemp,
                label: .suggest, adapterPath: adapter)
            if attempt == 0, let m = metrics {
                // Same `eval` metric as the 1:1 path; group chat has no pre-warm.
                NSLog("[SuggestKV] group eval=%d in=%d cached=%d slot=%d prewarm=none",
                      m.prefillTokenCount, m.promptTokenCount, m.cachedPrefixTokenCount,
                      m.cacheSlotIndex)
            }
            guard let raw = out else { continue }
            result = Prompts.parseSuggestions(raw)
            if result.refused || !result.suggestions.isEmpty { break }
        }

        if result.refused {
            notice = "No auto-reply suggested right now."
        } else {
            suggestions = Array(result.suggestions.prefix(3))
            if suggestions.isEmpty { notice = "No suggestion right now — try again." }
        }
        let start = Prompts.windowStart(messageCount: messages.count,
                                        window: Prompts.suggestWindow,
                                        stride: Prompts.suggestStride)
        turn.finish("attempts=\(attempts) chips=\(suggestions.count) "
                  + "prompt=\(prompt.count) chars "
                  + "window=\(start)..\(messages.count) (\(messages.count - start) msgs, "
                  + "stride=\(Prompts.suggestStride))")
        isSuggesting = false
    }

    func useSuggestion(_ text: String) {
        input = text
        suggestions = []
        notice = nil
    }
}
