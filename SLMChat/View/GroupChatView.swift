import SwiftUI

/// The group-chat screen: the user plus three bots. A Start/Stop control (or the
/// typed `/start` / `/stop` commands) drives the autonomous 30s chatter loop.
struct GroupChatView: View {
    @ObservedObject var group: GroupChat
    @ObservedObject var store: ConversationStore
    /// Set by sidebar search: scroll to this message and flash it.
    var jump: JumpRequest? = nil
    /// The message currently flashed after a search jump.
    @State private var flashID: UUID?

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider().opacity(0.5)
            liveBanner
            summaryBanner
            messageList
            GroupInputBar(group: group, isReady: isReady)
        }
        .onAppear { group.isActive = true }
        .onDisappear { group.isActive = false }
        .sheet(isPresented: $group.showSummary) {
            SummarySheet(summary: group.summary ?? "",
                         isStreaming: group.isSummarizing,
                         progress: group.summaryProgress,
                         modelName: store.selectedModelName)
        }
    }

    // MARK: - Summary banner (unread)

    @ViewBuilder
    private var summaryBanner: some View {
        if group.unreadCount > 0 {
            Button {
                Task { await group.summarizeUnread() }
            } label: {
                HStack {
                    Image(systemName: "text.append")
                    Text("Summarize \(group.unreadCount) unread message\(group.unreadCount == 1 ? "" : "s")")
                        .font(.subheadline.weight(.medium))
                    Spacer()
                    if group.isSummarizing {
                        ProgressView().controlSize(.small)
                    } else {
                        Image(systemName: "chevron.right").font(.caption)
                    }
                }
                .bannerCard()
            }
            .buttonStyle(.plain)
            .disabled(group.isBusy || !isReady)
        }
    }

    private var isReady: Bool { store.isReady }

    private var header: some View {
        ChatHeader(title: group.title,
                   subtitle: group.typingBot.map { "\($0) is typing…" }
                       ?? "\(group.botNames.joined(separator: ", ")) and you") {
            AvatarCircle(name: group.title, size: 40, systemImage: "person.3.fill", tint: .indigo)
        } actions: {
            startStopButton
            summarizeButton
            guardrailDemoButton
        }
    }

    // MARK: - Live banner

    @ViewBuilder
    private var liveBanner: some View {
        if group.isRunning {
            HStack(spacing: 6) {
                Circle().fill(.red).frame(width: 8, height: 8)
                Text("Live — bots are chatting (a message every \(GroupChat.tickSeconds)s). Type /stop to pause.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer(minLength: 0)
            }
            .bannerCard(.red)
        } else if let err = group.errorMessage {
            Text(err).font(.caption).foregroundStyle(.red)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal).padding(.vertical, 8)
        }
    }

    // MARK: - Messages

    /// DEMO trigger: replay a scripted GROUP scenario. Stops the tick loop first,
    /// so random bot posts don't interleave and change the verdicts mid-run.
    ///
    /// The group scenarios demonstrate what a 1:1 one can't — attribution across
    /// several authors, bystanders who must stay clean, and a pile-on whose harm
    /// is in the pattern rather than in any single message.
    @ViewBuilder
    private var guardrailDemoButton: some View {
        Menu {
            // Same shape as the 1:1 menu — language nests under the scenario.
            ForEach(GuardrailDemo.group) { scenario in
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
            // Only separates if there is something above it. `group` is empty
            // while the group scenarios sit in `GuardrailDemo.disabled`, and a
            // menu that opens with a divider reads as a missing first item.
            if !GuardrailDemo.group.isEmpty { Divider() }
            // Load test, not a scripted scenario: dumps an unscreened backlog and
            // leaves it to the viewport path. Scroll afterwards to make the
            // screener work, and watch the depth number in the log.
            Button("Flood: 120 messages") { group.floodGroup(count: 120) }
            Button("Flood: 500 messages") { group.floodGroup(count: 500) }
        } label: {
            if group.isPlayingDemo {
                ProgressView().controlSize(.small)
            } else {
                Image(systemName: "shield.lefthalf.filled")
            }
        }
        .headerMenuStyle()
        .help("Guardrail demo: scripted scenarios and floods")
        .disabled(group.isPlayingDemo)
    }

    /// Play one variant. See `ChatView.play` — the name carries the language so
    /// the `[GuardDemo]` header identifies which pack the run exercised.
    private func play(_ scenario: GuardrailDemo.Scenario,
                      _ variant: GuardrailDemo.Scenario.Variant) {
        Task {
            await group.playGuardrailDemo(variant.lines,
                                          named: scenario.title(variant))
        }
    }

    private var messageList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 6) {
                    OnDeviceNotice()
                    if group.messages.isEmpty {
                        emptyState
                    }
                    let messages = group.messages
                    ForEach(Array(messages.enumerated()), id: \.element.id) { index, message in
                        if ThreadLayout.startsDay(messages, index) {
                            DaySeparator(date: message.date)
                        }
                        MessageBubble(message: message,
                                      nativeLanguage: store.nativeLanguage,
                                      onTranslate: { Task { await group.translate(message) } },
                                      onRetranslate: { Task { await group.translate(message, force: true) } },
                                      onSummarize: { Task { await group.summarizeMessage(message) } },
                                      onReport: { reason, note in
                                          group.report(message, reason: reason, note: note)
                                      })
                        .searchFlash(flashID == message.id)
                        .id(message.id)
                        // Same bulk path as ChatView: history is screened when it
                        // reaches the viewport, not when it arrives.
                        .onAppear { group.screenFromViewport(message) }
                    }
                    if let bot = group.typingBot {
                        HStack(spacing: 6) {
                            ProgressView().controlSize(.small)
                            Text("\(bot) is typing…").font(.caption).foregroundStyle(.secondary)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .id("typing")
                    }
                }
                .padding(.horizontal, 24).padding(.bottom, 12)
            }
            .dismissKeyboardOnTap()
            .scrollDismissesKeyboard(.interactively)
            .onAppear {
                // Open at the first unread bot message — see ChatView for why
                // this decides which direction the screener has to look ahead in.
                // The group had no open-scroll at all, so it always landed at the
                // top of the history, which is the worst of both.
                DispatchQueue.main.async {
                    if SearchJump.perform(jump, in: group.messages.map(\.id),
                                          proxy: proxy, flash: $flashID) { return }
                    let firstUnread = group.messages
                        .first { $0.sender == .friend && !$0.isRead }
                    if let firstUnread {
                        proxy.scrollTo(firstUnread.id, anchor: .top)
                    } else {
                        proxy.scrollTo(group.messages.last?.id, anchor: .bottom)
                    }
                }
            }
            .onChange(of: jump) { request in
                _ = SearchJump.perform(request, in: group.messages.map(\.id),
                                       proxy: proxy, flash: $flashID)
            }
            .onChange(of: group.messages.count) { _ in
                withAnimation { proxy.scrollTo(group.messages.last?.id, anchor: .bottom) }
            }
        }
        // Per-message summary, presented as a sheet (like the conversation summary).
        .sheet(item: $group.summaryMessage) {
            MessageSummarySheet(message: $0, modelName: store.selectedModelName)
        }
    }

    private var emptyState: some View {
        VStack(spacing: 6) {
            Image(systemName: "person.3.fill").font(.largeTitle).foregroundStyle(.secondary)
            Text("Group with \(group.botNames.joined(separator: ", "))")
                .font(.subheadline.weight(.medium))
            Text("Type /start to let the bots chat, or send a message.")
                .font(.caption).foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
        .padding(.top, 40)
    }

    // MARK: - Toolbar

    /// Always-available summary of the recent group history (independent of the
    /// unread banner). Reuses `summarizeUnread`, which falls back to the recent
    /// thread when nothing is unread.
    private var summarizeButton: some View {
        Button {
            Task { await group.summarizeUnread() }
        } label: {
            if group.isSummarizing {
                ProgressView().controlSize(.small)
            } else {
                Image(systemName: "list.bullet.rectangle")
            }
        }
        .headerButtonStyle()
        .help("Summarize the recent conversation")
        .disabled(group.isBusy || !isReady || group.messages.isEmpty)
    }

    private var startStopButton: some View {
        Button {
            group.isRunning ? group.stop() : group.start()
        } label: {
            Image(systemName: group.isRunning ? "stop.circle" : "play.circle")
                .foregroundStyle(group.isRunning ? .red : .primary)
        }
        .headerButtonStyle()
        .help(group.isRunning ? "Stop the bots (/stop)" : "Let the bots chat (/start)")
        .disabled(!isReady)
    }

}

/// Group input bar with **local** text state — same rationale as `InputBar`:
/// binding the field to `group.input` republished the whole `GroupChat` on every
/// keystroke, re-rendering the entire screen. Keeping the text local confines
/// typing to this small view.
private struct GroupInputBar: View {
    @ObservedObject var group: GroupChat
    let isReady: Bool
    @State private var text = ""

    private var canSend: Bool {
        isReady && !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private func send() {
        guard canSend else { return }
        group.handleInput(text)
        text = ""
    }

    var body: some View {
        HStack(spacing: 12) {
            HStack(alignment: .bottom, spacing: 8) {
                TextField("Type a message... · /start · /stop", text: $text, axis: .vertical)
                    .textFieldStyle(.plain)
                    .font(.system(size: 14))
                    .lineLimit(1...6)
                    .padding(.vertical, 3)
                    // Return sends; Shift-Return inserts a newline.
                    .onKeyPress(.return, phases: .down) { press in
                        if press.modifiers.contains(.shift) { text += "\n" } else { send() }
                        return .handled
                    }
            }
            .padding(.horizontal, 18).padding(.vertical, 10)
            .background(Theme.field, in: RoundedRectangle(cornerRadius: 22))

            Button(action: send) {
                Image(systemName: "paperplane")
                    .font(.system(size: 19))
                    .foregroundStyle(canSend ? Theme.accent : .secondary)
            }
            .buttonStyle(.plain)
            .help("Send (Return)")
            .disabled(!canSend)
        }
        .padding(.horizontal, 24).padding(.top, 8).padding(.bottom, 16)
    }
}
