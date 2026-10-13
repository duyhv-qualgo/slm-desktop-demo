import SwiftUI

/// The main window: a narrow icon rail, the inbox sidebar (search, filter chips,
/// collapsible Group / Direct sections) and the selected chat. On iOS this was a
/// `NavigationStack` push; on macOS selection replaces navigation.
struct ConversationListView: View {
    @ObservedObject var store: ConversationStore

    enum Selection: Hashable {
        case group
        case conversation(Conversation)
    }

    enum Filter: String, CaseIterable, Identifiable {
        case all = "All", unread = "Unread", direct = "Direct", groups = "Groups"
        var id: Self { self }
    }

    @State private var selection: Selection?
    @State private var filter: Filter = .all
    @State private var query = ""
    @State private var groupExpanded = true
    @State private var directExpanded = true
    @State private var showNewChat = false
    @State private var showImageSearch = false
    @FocusState private var searchFocused: Bool

    var body: some View {
        HStack(spacing: 0) {
            rail
            HStack(spacing: 0) {
                sidebar
                    .frame(width: 300)
                    .background(Theme.sidebar)
                detail
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(Theme.chat)
            }
            .clipShape(UnevenRoundedRectangle(topLeadingRadius: Theme.panelRadius,
                                              bottomLeadingRadius: Theme.panelRadius))
            .padding(.vertical, 8)
        }
        .background(Theme.rail)
        .ignoresSafeArea()
        .tint(Theme.accent)
        .sheet(isPresented: $showNewChat) {
            NewChatView(store: store, isPresented: $showNewChat) { conversation in
                selection = .conversation(conversation)
            }
            .frame(minWidth: 360, minHeight: 440)
        }
        .sheet(isPresented: $showImageSearch) {
            ImageSearchView(store: store)
                .frame(minWidth: 560, minHeight: 480)
        }
        .focusedSceneValue(\.newChatAction) { showNewChat = true }
        .task { await store.warmUpIfNeeded() }
        .onAppear {
            // Don't land the cursor in search at launch; ⌘N / a click get there.
            DispatchQueue.main.async { searchFocused = false }
            #if DEBUG
            // Dev aid for snapshots: SLMCHAT_OPEN=first|group opens a chat.
            switch ProcessInfo.processInfo.environment["SLMCHAT_OPEN"] {
            case "first": selection = store.sortedConversations.first.map { .conversation($0) }
            case "group": selection = .group
            default: break
            }
            #endif
        }
    }

    // MARK: - Rail

    private var rail: some View {
        VStack(spacing: 14) {
            // Clear the traffic lights: the title bar is hidden and the rail
            // runs underneath it.
            Spacer().frame(height: 34)

            AvatarCircle(name: "You", size: 40, systemImage: "person.fill", tint: .gray)
                .help("You")

            // The one "workspace": this app. Selected style, like the design.
            Image(systemName: "bubble.left.and.text.bubble.right.fill")
                .font(.system(size: 18, weight: .semibold))
                .foregroundStyle(.white)
                .frame(width: 40, height: 40)
                .background(Theme.accent.gradient, in: RoundedRectangle(cornerRadius: 11))
                .padding(4)
                .overlay(RoundedRectangle(cornerRadius: 14).stroke(Theme.accent, lineWidth: 2))
                .help("SLM Chat")

            RailButton(systemImage: "plus", help: "New chat (⌘N)") { showNewChat = true }

            Spacer()

            RailButton(systemImage: "photo.stack", help: "Search photos (⇧⌘F)") {
                showImageSearch = true
            }
            .keyboardShortcut("f", modifiers: [.command, .shift])

            RailButton(systemImage: "bell", help: "Show unread") { filter = .unread }

            SettingsLink {
                Image(systemName: "gearshape")
                    .font(.system(size: 18))
                    .frame(width: 40, height: 40)
                    .background(Theme.sidebar, in: RoundedRectangle(cornerRadius: 10))
                    .overlay(alignment: .topTrailing) { engineDot }
            }
            .buttonStyle(.plain)
            .help(engineHelp)
        }
        .padding(.bottom, 16)
        .frame(width: 72)
    }

    /// Green when the on-device models are loaded, amber while warming up, red on
    /// error — the design's presence dot, repurposed as engine status.
    private var engineDot: some View {
        Circle()
            .fill(store.errorMessage != nil ? Color.red : store.isReady ? Color.green : Color.orange)
            .frame(width: 8, height: 8)
            .padding(5)
    }

    private var engineHelp: String {
        if let err = store.errorMessage { return "Settings — \(err)" }
        return store.isReady ? "Settings — on-device models ready"
                             : "Settings — \(store.statusText ?? "loading models…")"
    }

    // MARK: - Sidebar

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("SLM Chat").font(.system(size: 20, weight: .semibold))
                Spacer()
                Button { showNewChat = true } label: {
                    Image(systemName: "square.and.pencil").font(.system(size: 17))
                }
                .buttonStyle(.plain)
                .help("New chat (⌘N)")
            }
            .padding(.horizontal, 20)
            .padding(.top, 22)

            searchField
                .padding(.horizontal, 16)
                .padding(.top, 16)

            HStack(spacing: 6) {
                ForEach(Filter.allCases) { f in
                    FilterChip(title: f.rawValue, selected: filter == f) { filter = f }
                }
            }
            .padding(.horizontal, 16)
            .padding(.top, 12)

            ScrollView {
                LazyVStack(alignment: .leading, spacing: 2) {
                    if showsGroup {
                        SectionHeader(title: "Group", expanded: $groupExpanded)
                        if groupExpanded {
                            SidebarRow(selected: selection == .group) {
                                selection = .group
                            } content: {
                                GroupRowContent(group: store.group)
                            }
                        }
                    }
                    if filter != .groups {
                        SectionHeader(title: "Direct", expanded: $directExpanded)
                        if directExpanded {
                            ForEach(visibleConversations) { conversation in
                                SidebarRow(selected: selection == .conversation(conversation)) {
                                    selection = .conversation(conversation)
                                } content: {
                                    ConversationRowContent(conversation: conversation)
                                }
                                .contextMenu { rowMenu(conversation) }
                            }
                            if visibleConversations.isEmpty {
                                Text(query.isEmpty ? "No chats" : "No matches")
                                    .font(.system(size: 12))
                                    .foregroundStyle(Theme.textSecondary)
                                    .padding(.horizontal, 22).padding(.vertical, 6)
                            }
                        }
                    }
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 12)
            }

            statusLabel
        }
    }

    private var searchField: some View {
        HStack(spacing: 8) {
            Image(systemName: "magnifyingglass").foregroundStyle(Theme.textSecondary)
            TextField("Search people or messages", text: $query)
                .textFieldStyle(.plain)
                .focused($searchFocused)
            if !query.isEmpty {
                Button { query = "" } label: {
                    Image(systemName: "xmark.circle.fill").foregroundStyle(Theme.textSecondary)
                }
                .buttonStyle(.plain)
            }
        }
        .font(.system(size: 13.5))
        .padding(.horizontal, 14)
        .frame(height: 38)
        .background(Capsule().stroke(Theme.fieldBorder))
    }

    private var showsGroup: Bool {
        switch filter {
        case .direct: return false
        case .unread: return store.group.unreadCount > 0 && matches(store.group.title)
        default:      return matches(store.group.title)
                          || store.group.botNames.contains(where: matches)
        }
    }

    private var visibleConversations: [Conversation] {
        store.sortedConversations.filter { c in
            (filter != .unread || c.unreadCount > 0)
                && (matches(c.contactName) || matches(c.lastMessagePreview))
        }
    }

    private func matches(_ text: String) -> Bool {
        let q = query.trimmingCharacters(in: .whitespaces)
        return q.isEmpty || text.localizedCaseInsensitiveContains(q)
    }

    @ViewBuilder
    private func rowMenu(_ conversation: Conversation) -> some View {
        if conversation.unreadCount == 0 {
            Button {
                conversation.markUnread()
            } label: {
                Label("Mark as Unread", systemImage: "envelope.badge")
            }
        }
        Divider()
        Button(role: .destructive) {
            if selection == .conversation(conversation) { selection = nil }
            store.delete(conversation)
        } label: {
            Label("Delete Chat", systemImage: "trash")
        }
    }

    @ViewBuilder
    private var statusLabel: some View {
        if let err = store.errorMessage {
            Text(err).font(.caption).foregroundStyle(.red).lineLimit(2)
                .padding(.horizontal, 20).padding(.vertical, 10)
        } else if !store.isReady, let s = store.statusText {
            HStack(spacing: 6) {
                ProgressView().controlSize(.small)
                Text(s).font(.caption).foregroundStyle(Theme.textSecondary).lineLimit(1)
            }
            .padding(.horizontal, 20).padding(.vertical, 10)
        }
    }

    // MARK: - Detail

    @ViewBuilder
    private var detail: some View {
        switch selection {
        case .group:
            GroupChatView(group: store.group, store: store)
        case .conversation(let conversation):
            // Fresh identity per chat, so `ChatView.onAppear/onDisappear` fire on
            // every switch — they drive `isActive`, auto-suggest and compose teardown.
            ChatView(conversation: conversation, store: store)
                .id(conversation.id)
        case nil:
            VStack(spacing: 10) {
                Image(systemName: "bubble.left.and.bubble.right")
                    .font(.system(size: 40, weight: .light))
                Text("Select a chat to start messaging")
                    .font(.system(size: 14, weight: .medium))
                Text("or press ⌘N for a new one")
                    .font(.system(size: 12))
            }
            .foregroundStyle(Theme.textSecondary)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }
}

// MARK: - Sidebar pieces

private struct RailButton: View {
    let systemImage: String
    let help: String
    let action: () -> Void
    @State private var hovering = false

    var body: some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .font(.system(size: 18))
                .frame(width: 40, height: 40)
                .background(hovering ? Theme.rowHover : Theme.sidebar.opacity(0),
                            in: RoundedRectangle(cornerRadius: 10))
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
        .help(help)
    }
}

private struct FilterChip: View {
    let title: String
    let selected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.system(size: 12.5, weight: .semibold))
                .foregroundStyle(selected ? Color.black : .primary)
                .padding(.horizontal, 11).frame(height: 26)
                .background(selected ? Color.white : .clear, in: Capsule())
                .overlay(Capsule().stroke(selected ? .clear : Theme.fieldBorder))
                .contentShape(Capsule())
        }
        .buttonStyle(.plain)
    }
}

private struct SectionHeader: View {
    let title: String
    @Binding var expanded: Bool

    var body: some View {
        Button { withAnimation(.easeInOut(duration: 0.15)) { expanded.toggle() } } label: {
            HStack(spacing: 6) {
                Image(systemName: "chevron.down")
                    .font(.system(size: 10, weight: .semibold))
                    .rotationEffect(.degrees(expanded ? 0 : -90))
                Text(title).font(.system(size: 13))
                Spacer()
            }
            .foregroundStyle(Theme.textSecondary)
            .padding(.horizontal, 8).padding(.top, 10).padding(.bottom, 4)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}

private struct SidebarRow<Content: View>: View {
    let selected: Bool
    let action: () -> Void
    @ViewBuilder let content: () -> Content
    @State private var hovering = false

    var body: some View {
        Button(action: action) {
            content()
                .padding(.horizontal, 12)
                .frame(height: 42)
                .background(selected ? Theme.rowSelected : hovering ? Theme.rowHover : .clear,
                            in: RoundedRectangle(cornerRadius: 10))
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
    }
}

/// Observes the group so its unread badge and live dot update in real time.
private struct GroupRowContent: View {
    @ObservedObject var group: GroupChat

    var body: some View {
        HStack(spacing: 10) {
            AvatarCircle(name: group.title, size: 26, systemImage: "person.3.fill", tint: .indigo)
            Text("\(group.title) · \(group.botNames.joined(separator: ", "))")
                .font(.system(size: 14))
                .lineLimit(1)
            Spacer(minLength: 6)
            if group.isRunning {
                Circle().fill(.red).frame(width: 7, height: 7).help("Bots are chatting")
            }
            UnreadBadge(count: group.unreadCount)
        }
    }
}

/// Observes the conversation so the unread badge updates live as replies stream in.
private struct ConversationRowContent: View {
    @ObservedObject var conversation: Conversation

    var body: some View {
        HStack(spacing: 10) {
            AvatarCircle(name: conversation.contactName, size: 26, tint: conversation.tint)
            Text(conversation.contactName)
                .font(.system(size: 14, weight: conversation.unreadCount > 0 ? .semibold : .regular))
                .lineLimit(1)
            Spacer(minLength: 6)
            UnreadBadge(count: conversation.unreadCount)
        }
        .help(conversation.lastMessagePreview)
    }
}

private struct UnreadBadge: View {
    let count: Int

    var body: some View {
        if count > 0 {
            Text("\(count)")
                .font(.system(size: 11, weight: .bold))
                .foregroundStyle(.white)
                .padding(.horizontal, 6)
                .frame(minWidth: 20, minHeight: 18)
                .background(Theme.accent, in: Capsule())
        }
    }
}

/// ⌘N from the menu bar reaches the window's "new chat" sheet through this.
struct NewChatActionKey: FocusedValueKey {
    typealias Value = () -> Void
}

extension FocusedValues {
    var newChatAction: (() -> Void)? {
        get { self[NewChatActionKey.self] }
        set { self[NewChatActionKey.self] = newValue }
    }
}
