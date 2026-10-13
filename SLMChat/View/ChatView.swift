import SwiftUI
import PhotosUI
import UniformTypeIdentifiers

/// The chat screen for one conversation: message list, an on-device summary banner
/// for unread messages, auto-reply suggestion chips, the Magic Compose bar, and
/// the input bar.
///
/// Translation direction is not a UI choice: it is detected per message by
/// `Prompts.autoTarget`. The nav-bar override that used to sit here is gone —
/// it never affected the per-message Translate button's own label, so an
/// overridden direction and the label the user tapped could disagree.
///
/// Auto-reply suggestions have NO manual button here — they are generated
/// automatically when it's the user's turn (`Conversation.scheduleAutoSuggest`)
/// and land in `suggestionRow`. The input bar's one AI control is the Magic
/// Compose wand. Two adjacent sparkle buttons that both meant "help me write"
/// were the reason; see the notes on `scheduleAutoSuggest` for what that costs.
///
/// State lives on the injected `Conversation` (owned by the store), NOT on a local
/// view model — so a reply generated after the user leaves this screen still lands.
struct ChatView: View {
    @ObservedObject var conversation: Conversation
    @ObservedObject var store: ConversationStore
    /// Set by sidebar search: scroll to this message and flash it.
    var jump: JumpRequest? = nil

    /// The message currently flashed after a search jump.
    @State private var flashID: UUID?
    /// The contact-info panel on the right (the design's split-view toggle).
    @State private var showInfo = false
    /// A file is being dragged over the thread.
    @State private var dropTargeted = false

    var body: some View {
        HStack(spacing: 0) {
            VStack(spacing: 0) {
                header
                Divider().opacity(0.5)
                guardrailUnavailableBanner
                guardrailBanner
                summaryBanner
                eventBanner
                messageList
                suggestionRow
                if conversation.isComposeOpen {
                    MagicComposeBar(conversation: conversation, store: store, isReady: isReady)
                }
                InputBar(conversation: conversation, isReady: isReady)
            }
            .overlay { if dropTargeted { dropHighlight } }
            .onDrop(of: [.image, .fileURL], isTargeted: $dropTargeted, perform: handleDrop)

            if showInfo {
                Divider()
                ContactInfoView(conversation: conversation, store: store)
                    .frame(width: 320)
                    .transition(.move(edge: .trailing))
            }
        }
        .animation(.easeInOut(duration: 0.2), value: conversation.isComposeOpen)
        .animation(.easeInOut(duration: 0.2), value: showInfo)
        .onAppear {
            conversation.isActive = true
            // Opening a chat that's already waiting on your reply → offer suggestions.
            conversation.scheduleAutoSuggest()
        }
        .onDisappear {
            conversation.isActive = false
            conversation.cancelAutoSuggest()
            conversation.endCompose()
        }
        // As the draft changes: typing cancels a pending auto-suggest; clearing it
        // (or a tapped chip that later empties) lets one reschedule.
        .onChange(of: conversation.input) { _ in conversation.scheduleAutoSuggest() }
        .sheet(isPresented: $conversation.showSummary) {
            SummarySheet(summary: conversation.summary ?? "",
                         isStreaming: conversation.isSummarizing,
                         progress: conversation.summaryProgress,
                         modelName: store.selectedModelName)
        }
    }

    private var isReady: Bool { store.isReady }

    // MARK: - Header

    private var header: some View {
        ChatHeader(title: conversation.contactName,
                   subtitle: headerSubtitle,
                   onTitleTap: { showInfo.toggle() }) {
            AvatarCircle(name: conversation.contactName, size: 40, tint: conversation.tint)
        } actions: {
            simulateBurstButton
            guardrailDemoButton
            HeaderIconButton(systemImage: "sidebar.right", help: "Contact info, media and files",
                             active: showInfo) { showInfo.toggle() }
        }
    }

    private var headerSubtitle: String {
        if conversation.isBotTyping { return "typing…" }
        return "On-device AI · \(store.selectedModelName)"
    }

    // MARK: - Drag and drop

    private var dropHighlight: some View {
        RoundedRectangle(cornerRadius: 12)
            .stroke(Theme.accent, style: StrokeStyle(lineWidth: 2, dash: [6]))
            .background(Theme.accent.opacity(0.06))
            .overlay(Label("Drop to send", systemImage: "square.and.arrow.down")
                        .font(.headline).foregroundStyle(Theme.accent))
            .padding(8)
            .allowsHitTesting(false)
    }

    /// Images go through `sendImage` (so they get tagged for photo search); any
    /// other file is sent as a document.
    private func handleDrop(_ providers: [NSItemProvider]) -> Bool {
        guard isReady, !conversation.isBotTyping else { return false }
        for provider in providers {
            _ = provider.loadFileRepresentation(for: .item) { url, _, _ in
                guard let url, let data = try? Data(contentsOf: url) else { return }
                let isImage = UTType(filenameExtension: url.pathExtension)?.conforms(to: .image) ?? false
                let name = url.lastPathComponent
                Task { @MainActor in
                    if isImage { conversation.sendImage(data) }
                    else { conversation.sendFile(data, name: name) }
                }
            }
        }
        return true
    }

    // MARK: - Message list

    private var messageList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 6) {
                    OnDeviceNotice()
                    let messages = conversation.messages
                    ForEach(Array(messages.enumerated()), id: \.element.id) { index, message in
                        if ThreadLayout.startsDay(messages, index) {
                            DaySeparator(date: message.date)
                        }
                        MessageBubble(message: message,
                                      nativeLanguage: store.nativeLanguage,
                                      onTranslate: { Task { await conversation.translate(message) } },
                                      onRetranslate: { Task { await conversation.translate(message, force: true) } },
                                      onSummarize: { Task { await conversation.summarizeMessage(message) } },
                                      contactName: conversation.contactName,
                                      onReport: { reason, note in
                                          conversation.report(message, reason: reason, note: note)
                                      })
                        .searchFlash(flashID == message.id)
                        .id(message.id)
                        // Bulk history is appended unscreened, so this is what
                        // gets it classified — the row asks as it materializes,
                        // and screens a little ahead of itself. A no-op for
                        // anything already screened, which is the common case.
                        .onAppear { conversation.screenFromViewport(message) }
                    }
                    if conversation.isBotTyping {
                        typingIndicator
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .id("typing")
                    }
                }
                .padding()
            }
            // Tap anywhere in the message area (including empty space) to dismiss
            // the keyboard, and let a drag dismiss it interactively too.
            .dismissKeyboardOnTap()
            .scrollDismissesKeyboard(.interactively)
            .onAppear {
                // Open at the FIRST UNREAD message, the way a messenger does, and
                // fall back to the latest when everything has been read.
                //
                // This is not cosmetic for the guardrail: it decides which
                // direction the reader travels. Anchored here, the unread
                // messages are BELOW and the reader scrolls DOWN into them —
                // which is why `MessageScreener` looks ahead in both directions
                // rather than only into older history.
                //
                // `.top` so the first unread starts at the top of the screen with
                // its unread neighbours below it, rather than being the last
                // thing visible.
                DispatchQueue.main.async {
                    if SearchJump.perform(jump, in: conversation.messages.map(\.id),
                                          proxy: proxy, flash: $flashID) { return }
                    let firstUnread = conversation.messages
                        .first { $0.sender == .friend && !$0.isRead }
                    if let firstUnread {
                        proxy.scrollTo(firstUnread.id, anchor: .top)
                    } else {
                        proxy.scrollTo(conversation.messages.last?.id, anchor: .bottom)
                    }
                }
            }
            .onChange(of: jump) { request in
                _ = SearchJump.perform(request, in: conversation.messages.map(\.id),
                                       proxy: proxy, flash: $flashID)
            }
            .onChange(of: conversation.messages.count) { _ in
                withAnimation { proxy.scrollTo(conversation.messages.last?.id, anchor: .bottom) }
            }
        }
        // Per-message summary, presented as a sheet (like the conversation summary).
        .sheet(item: $conversation.summaryMessage) {
            MessageSummarySheet(message: $0, modelName: store.selectedModelName)
        }
    }


    private var typingIndicator: some View {
        HStack(spacing: 6) {
            ProgressView().controlSize(.small)
            Text("\(conversation.contactName) is typing…")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    // MARK: - Summary banner (Feature 4)

    @ViewBuilder
    private var summaryBanner: some View {
        let count = conversation.unrepliedCount
        // Always offer the catch-up summary once there's a real backlog of
        // unreplied messages (more than 3). This is driven by the unreplied
        // count, not the read flag, so summarizing doesn't hide the banner —
        // it stays until you actually reply.
        if count > 3 {
            Button {
                Task { await conversation.summarizeUnread() }
            } label: {
                HStack(spacing: 10) {
                    Image(systemName: "text.append")
                        .font(.title3)
                        .foregroundStyle(Color.accentColor)
                    VStack(alignment: .leading, spacing: 1) {
                        Text("Catch up before you reply")
                            .font(.subheadline.weight(.semibold))
                        Text("Summarize \(count) new message\(count == 1 ? "" : "s") from \(conversation.contactName)")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    if conversation.isSummarizing {
                        ProgressView().controlSize(.small)
                    } else {
                        Image(systemName: "chevron.right").font(.caption).foregroundStyle(.secondary)
                    }
                }
                .bannerCard()
            }
            .buttonStyle(.plain)
            .disabled(conversation.isBusy || !isReady)
        }
    }

    // MARK: - Safety guardrail banner (Feature 6)

    /// Shown when the classifier failed to load. Without it the screen is silently
    /// misleading: `classify` fails open to `.clean`, so no bubble is ever flagged
    /// and the chat looks exactly like one that was screened and found safe. This
    /// says the check didn't run — it does not claim the content is unsafe.
    @ViewBuilder
    private var guardrailUnavailableBanner: some View {
        if store.guardrailError != nil {
            HStack(spacing: 8) {
                Image(systemName: "exclamationmark.shield")
                    .foregroundStyle(.secondary)
                Text("Safety checks unavailable — messages aren't being screened.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                Spacer(minLength: 0)
            }
            .bannerCard(.secondary)
        }
    }

    @ViewBuilder
    private var guardrailBanner: some View {
        let items = conversation.flaggedItems
        if !items.isEmpty {
            // No severity (a model-only detection) or `.safe` both render as the
            // bare class — there is nothing informative to append in either case.
            let summary = items.map { item -> String in
                guard let sev = item.severity, sev != .safe else {
                    return item.label.displayName
                }
                return "\(item.label.displayName) · \(sev.displayName)"
            }.joined(separator: ", ")
            // Tint the banner by the most serious severity present.
            let critical = items.contains { $0.severity == .critical }
            HStack(spacing: 8) {
                Image(systemName: "exclamationmark.shield.fill")
                    .foregroundStyle(critical ? .red : .orange)
                Text("Flagged content: \(summary)")
                    .font(.subheadline.weight(.medium))
                    .lineLimit(2)
                Spacer(minLength: 0)
            }
            .bannerCard(critical ? .red : .orange)
        }
    }

    // MARK: - Event reminder banner (Feature 5)

    @ViewBuilder
    private var eventBanner: some View {
        if let event = conversation.pendingEvent {
            HStack(spacing: 12) {
                Image(systemName: "calendar")
                    .font(.title3)
                    .foregroundStyle(Color.accentColor)
                VStack(alignment: .leading, spacing: 2) {
                    Text(event.eventName.isEmpty ? "Event detected" : event.eventName)
                        .font(.subheadline.weight(.semibold))
                        .lineLimit(1)
                    if !event.subtitle.isEmpty {
                        Text(event.subtitle)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(2)
                    }
                }
                Spacer()
                Button("Dismiss") { conversation.dismissEvent() }
                    .font(.caption)
                    .buttonStyle(.borderless)
                Button("Remind me") { conversation.confirmReminder() }
                    .font(.caption.weight(.semibold))
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
            }
            .bannerCard()
        } else if let note = conversation.eventReminderNote {
            HStack(spacing: 6) {
                Image(systemName: "checkmark.circle.fill")
                Text(note)
            }
            .font(.caption.weight(.medium))
            .foregroundStyle(.green)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal).padding(.vertical, 8)
        }
    }

    // MARK: - Suggestion chips (Feature 3)

    @ViewBuilder
    private var suggestionRow: some View {
        if conversation.isSuggesting {
            // Auto-suggest is the only feature that starts without the user
            // asking, so this is the one place it can say it's working. It used
            // to be the spinner inside the input bar's suggest button; that
            // button is gone, and a silent 3 s gap before chips appear reads as
            // nothing happening.
            HStack(spacing: 8) {
                ProgressView().controlSize(.small)
                Text("Suggesting replies\u{2026}")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer(minLength: 0)
            }
            .padding(.horizontal).padding(.vertical, 8)
        } else if !conversation.suggestions.isEmpty {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(conversation.suggestions, id: \.self) { s in
                        SuggestionChip(text: s) { conversation.useSuggestion(s) }
                    }
                }
                .padding(.horizontal, 24).padding(.vertical, 8)
            }
        } else if let notice = conversation.notice {
            // Neutral note (e.g. the smart-reply model declined to suggest).
            HStack(spacing: 6) {
                Image(systemName: conversation.missingPack == nil
                      ? "info.circle" : "arrow.down.circle")
                Text(notice)
                // The install lives HERE, not only in Settings, because Settings
                // offers a Get button for the native language's pack alone — a user
                // whose native language is Vietnamese has no other route to the
                // Spanish pack this notice is asking for.
                if let pack = conversation.missingPack {
                    Spacer(minLength: 8)
                    packInstallButton(pack)
                }
            }
            .font(.caption)
            .foregroundStyle(.secondary)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal).padding(.vertical, 8)
        }
    }

    /// Get / progress / retry for the pack blocking suggestions in this chat.
    /// Mirrors the Settings row, minus the states that can't occur here (a pack
    /// that is already installed doesn't block anything).
    @ViewBuilder
    private func packInstallButton(_ pack: LoRAAdapter) -> some View {
        switch store.loraStore.state(of: pack) {
        case .downloading(let progress):
            if let progress {
                ProgressView(value: progress).progressViewStyle(.linear).frame(width: 60)
            } else {
                ProgressView().controlSize(.small)
            }
        case .installed:
            // The download finished while this notice was on screen. Clearing it
            // turns the stale "install it" line into a live suggest button.
            Button("Suggest") {
                conversation.missingPack = nil
                conversation.notice = nil
                Task { await conversation.suggest() }
            }
            .font(.caption)
        case .failed, .notInstalled:
            Button("Get") { store.loraStore.download(pack) }
                .font(.caption)
        }
    }

    // MARK: - Toolbar

    /// DEMO trigger: one tap fans a single on-device/API generation out into several
    /// UNREAD friend bubbles, so the "Summarize N unread messages" banner has a
    /// realistic backlog to show off (a turn-by-turn reply only yields one bubble).
    /// A dedicated, always-visible toolbar button — not buried in a menu — so it's
    /// easy to hit live.
    /// DEMO trigger: replay a scripted conversation that trips the guardrail, one
    /// message at a time, so the classifier can be watched deciding in the log
    /// (`LOG_FILTER=1 ./scripts/run_sim.sh`, or filter on `[GuardDemo]`).
    ///
    /// Scripted rather than generated, so the verdicts are identical every run —
    /// which is what makes it useful for spotting a regression. The ✨ button
    /// beside it does the opposite: real generation, different every time.
    @ViewBuilder
    private var guardrailDemoButton: some View {
        Menu {
            // Language is a submenu, not a sibling row: one scenario in five
            // languages was five near-identical rows, and with several scenarios
            // the harm being demonstrated — the part that differs — is what
            // scrolls away. A single-language scenario stays one tap.
            ForEach(GuardrailDemo.oneToOne) { scenario in
                if scenario.menuIsFlat, let only = scenario.variants.first {
                    Button(scenario.name) { play(scenario, only) }
                } else {
                    Menu(scenario.name) {
                        ForEach(scenario.variants) { variant in
                            Button(variant.label) { play(scenario, variant) }
                        }
                    }
                }
            }
        } label: {
            if conversation.isPlayingDemo {
                ProgressView().controlSize(.small)
            } else {
                Image(systemName: "shield.lefthalf.filled")
            }
        }
        .headerMenuStyle()
        .help("Guardrail demo: replay a scripted conversation")
        .disabled(conversation.isPlayingDemo || !isReady)
    }

    /// Play one variant. The name carries the language so the `[GuardDemo]`
    /// header says which pack the run exercised — with eight distinct packs,
    /// "Harassment → sextortion" alone no longer identifies a run.
    private func play(_ scenario: GuardrailDemo.Scenario,
                      _ variant: GuardrailDemo.Scenario.Variant) {
        Task {
            await conversation.playGuardrailDemo(variant.lines,
                                                 named: scenario.title(variant))
        }
    }

    @ViewBuilder
    private var simulateBurstButton: some View {
        Button {
            Task { await conversation.seedUnreadBurst() }
        } label: {
            if conversation.isSeedingBurst {
                ProgressView().controlSize(.small)
            } else {
                Image(systemName: "sparkles")
            }
        }
        .headerButtonStyle()
        .help("Simulate a burst of unread messages")
        .disabled(conversation.isBusy || !isReady)
    }

}

/// The message input bar, deliberately split into its own view with **local**
/// text state. Binding the field to `conversation.input` (an `@Published` on the
/// whole thread) made every keystroke — and the focus/keyboard cycle — republish
/// the conversation and re-render all of `ChatView` (the full message list and
/// every banner), causing tap-to-type lag. Keeping the text local means typing
/// only invalidates this small view; the message list is untouched.
private struct InputBar: View {
    @ObservedObject var conversation: Conversation
    let isReady: Bool
    @State private var text = ""
    @State private var photoItem: PhotosPickerItem?
    @State private var showFileImporter = false

    @State private var showPhotoPicker = false

    private var canSend: Bool {
        isReady && !conversation.isBotTyping
            && !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private func send() {
        guard canSend else { return }
        conversation.send(text)
        text = ""
    }

    var body: some View {
        HStack(spacing: 12) {
            // Attach: photo library or any file. Tagging runs on-device after send.
            Menu {
                Button { showPhotoPicker = true } label: {
                    Label("Photo Library…", systemImage: "photo.on.rectangle")
                }
                Button { showFileImporter = true } label: {
                    Label("File…", systemImage: "paperclip")
                }
            } label: {
                Image(systemName: "plus").font(.system(size: 20, weight: .regular))
            }
            .menuStyle(.borderlessButton)
            .menuIndicator(.hidden)
            .tint(.primary)
            .fixedSize()
            .help("Attach a photo or file — or drop one on the chat")
            .disabled(!isReady || conversation.isBotTyping)
            .photosPicker(isPresented: $showPhotoPicker, selection: $photoItem, matching: .images)
            .onChange(of: photoItem) { newItem in
                guard let newItem else { return }
                let caption = text
                Task {
                    if let data = try? await newItem.loadTransferable(type: Data.self) {
                        conversation.sendImage(data, caption: caption)
                    }
                    await MainActor.run { photoItem = nil; text = "" }
                }
            }
            .fileImporter(isPresented: $showFileImporter, allowedContentTypes: [.item]) { result in
                guard case .success(let url) = result else { return }
                let scoped = url.startAccessingSecurityScopedResource()
                defer { if scoped { url.stopAccessingSecurityScopedResource() } }
                guard let data = try? Data(contentsOf: url) else { return }
                if UTType(filenameExtension: url.pathExtension)?.conforms(to: .image) == true {
                    conversation.sendImage(data, caption: text)
                    text = ""
                } else {
                    conversation.sendFile(data, name: url.lastPathComponent)
                }
            }

            HStack(alignment: .bottom, spacing: 8) {
                // Read-only while the compose bar is open, and that is load-bearing
                // rather than cosmetic. The bar captures the draft ONCE and every chip
                // rewrites THAT text, so a hand edit landing in between would silently
                // rewrite something other than what the user last typed. Freezing the
                // field closes that gap without pushing every keystroke onto the
                // conversation, which is the republish this view exists to avoid.
                //
                // Not dimmed: after a chip runs, this field holds the RESULT, and the
                // user can send it from here directly.
                TextField("Type a message...", text: $text, axis: .vertical)
                    .textFieldStyle(.plain)
                    .font(.system(size: 14))
                    .lineLimit(1...6)
                    .disabled(conversation.isComposeOpen)
                    .padding(.vertical, 3)
                    // Return sends; Shift-Return inserts a newline.
                    .onKeyPress(.return, phases: .down) { press in
                        if press.modifiers.contains(.shift) {
                            text += "\n"
                        } else {
                            send()
                        }
                        return .handled
                    }
                    // A disabled TextField swallows clicks silently. Reaching for
                    // the text is a clear "I'm done with the chips" — so it closes
                    // the bar and hands editing straight back.
                    .overlay {
                        if conversation.isComposeOpen {
                            Color.clear
                                .contentShape(Rectangle())
                                .onTapGesture { conversation.endCompose() }
                        }
                    }

                // Magic Compose — where the design has its emoji button. Shown
                // whether or not there is a draft: an empty box opens the
                // smart-reply chips, a draft opens the rewrite ones. The draft is
                // handed over explicitly because it lives in THIS view's state.
                Button {
                    if conversation.isComposeOpen {
                        conversation.endCompose()
                    } else {
                        conversation.beginCompose(draft: text)
                    }
                } label: {
                    Image(systemName: conversation.isComposeOpen
                          ? "wand.and.stars.inverse" : "wand.and.stars")
                        .font(.system(size: 17))
                        .foregroundStyle(conversation.isComposeOpen ? Theme.accent : .secondary)
                }
                .buttonStyle(.plain)
                .disabled(!isReady || conversation.isBotTyping)
                .help(text.isEmpty ? "Write a reply with AI" : "Rewrite with AI")
            }
            .padding(.horizontal, 18).padding(.vertical, 10)
            .background(Theme.field, in: RoundedRectangle(cornerRadius: 22))

            Button(action: send) {
                Image(systemName: "paperplane")
                    .font(.system(size: 19))
                    .foregroundStyle(canSend ? Theme.accent : .secondary)
            }
            .buttonStyle(.plain)
            .keyboardShortcut(.return, modifiers: .command)
            .help("Send (Return)")
            .disabled(!canSend)
        }
        .padding(.horizontal, 24).padding(.top, 8).padding(.bottom, 16)
        // A tapped suggestion fills `conversation.input`; mirror that into the
        // local field. Send clears input to "", which we ignore here.
        .onChange(of: conversation.input) { newValue in
            if !newValue.isEmpty, newValue != text { text = newValue }
        }
        // Every PROGRAMMATIC write — an applied rewrite, an undo back to an empty
        // draft — arrives as a tick here. The value-based mirror above can't carry
        // those: it deliberately ignores "" so `send`'s clear doesn't fight the
        // field, and an undo to an empty draft is that same value. See
        // `Conversation.setInput`.
        .onChange(of: conversation.inputRevision) { _ in
            text = conversation.input
        }
    }
}
