import SwiftUI

/// Telegram-style contact profile: avatar + name, then a **Media | Files** switcher.
/// Media is a grid of shared images, searchable by the on-device image tags
/// (Vietnamese queries translated to English first). Files is a list of shared
/// documents, filtered live by filename. Both are scoped to this one conversation.
struct ContactInfoView: View {
    @ObservedObject var conversation: Conversation
    @ObservedObject var store: ConversationStore

    enum Tab: String, CaseIterable { case media = "Media", files = "Files" }
    @State private var tab: Tab = .media
    @State private var query = ""
    /// Media search results (`nil` = not searching). Files filter live instead.
    @State private var results: [ConversationStore.ImageSearchHit]?
    @State private var isSearching = false

    private let columns = Array(repeating: GridItem(.flexible(), spacing: 2), count: 3)

    private var allImages: [ChatMessage] {
        conversation.messages.filter { $0.hasImage }.reversed()
    }
    private var allFiles: [ChatMessage] {
        conversation.messages.filter { $0.hasFile }.reversed()
    }
    private var filteredFiles: [ChatMessage] {
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !q.isEmpty else { return allFiles }
        return allFiles.filter { ($0.fileName ?? "").lowercased().contains(q) }
    }

    var body: some View {
        ScrollView {
            VStack(spacing: 16) {
                header
                picker
                searchField
                content
            }
            .padding(.vertical, 24)
        }
        .background(Theme.sidebar)
    }

    // MARK: - Header + switcher

    private var header: some View {
        VStack(spacing: 10) {
            AvatarCircle(name: conversation.contactName, size: 88, tint: conversation.tint)
            Text(conversation.contactName)
                .font(.title2.weight(.semibold))
        }
        .frame(maxWidth: .infinity)
    }

    private var picker: some View {
        Picker("", selection: $tab) {
            Text("Media (\(allImages.count))").tag(Tab.media)
            Text("Files (\(allFiles.count))").tag(Tab.files)
        }
        .pickerStyle(.segmented)
        .padding(.horizontal)
        .onChange(of: tab) { _ in query = ""; results = nil }
    }

    private var searchField: some View {
        HStack(spacing: 8) {
            Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
            TextField(tab == .media ? "Search photos by keyword…" : "Search files by name…",
                      text: $query)
                .textFieldStyle(.plain)
                .submitLabel(.search)
                .onSubmit { if tab == .media { runSearch(debounce: false) } }
                .onChange(of: query) { _ in if tab == .media { runSearch(debounce: true) } }
            if isSearching { ProgressView().controlSize(.small) }
            else if !query.isEmpty {
                Button { query = ""; results = nil } label: {
                    Image(systemName: "xmark.circle.fill").foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.horizontal, 12).padding(.vertical, 9)
        .background(Color.secondaryBackground)
        .clipShape(RoundedRectangle(cornerRadius: 10))
        .padding(.horizontal)
    }

    // MARK: - Content

    @ViewBuilder
    private var content: some View {
        switch tab {
        case .media: mediaContent
        case .files: filesContent
        }
    }

    @ViewBuilder
    private var mediaContent: some View {
        if let hits = results {
            if hits.isEmpty && !isSearching {
                note("No photo in this chat matched “\(query)”.")
            } else {
                grid(hits.map(\.message))
            }
        } else if allImages.isEmpty {
            note("No photos shared yet. Attach one in the chat — it's tagged on-device so you can search it here.")
        } else {
            grid(allImages)
        }
    }

    @ViewBuilder
    private var filesContent: some View {
        if allFiles.isEmpty {
            note("No files shared yet. Tap the paperclip in the chat to send a document.")
        } else if filteredFiles.isEmpty {
            note("No file matched “\(query)”.")
        } else {
            VStack(spacing: 0) {
                ForEach(filteredFiles) { message in
                    fileRow(message)
                    if message.id != filteredFiles.last?.id { Divider().padding(.leading, 56) }
                }
            }
            .padding(.horizontal)
        }
    }

    private func grid(_ messages: [ChatMessage]) -> some View {
        LazyVGrid(columns: columns, spacing: 2) {
            ForEach(messages) { message in
                Color.clear
                    .aspectRatio(1, contentMode: .fit)
                    .overlay { StoredImageThumb(imageID: message.imageID) }
                    .clipShape(RoundedRectangle(cornerRadius: 4))
            }
        }
        .padding(.horizontal, 2)
    }

    private func fileRow(_ message: ChatMessage) -> some View {
        HStack(spacing: 12) {
            Image(systemName: "doc.fill")
                .font(.title2)
                .foregroundStyle(Color.accentColor)
                .frame(width: 44, height: 44)
                .background(Color.accentColor.opacity(0.12))
                .clipShape(RoundedRectangle(cornerRadius: 8))
            VStack(alignment: .leading, spacing: 2) {
                Text(message.fileName ?? "File")
                    .font(.subheadline.weight(.medium))
                    .lineLimit(1)
                HStack(spacing: 6) {
                    if let size = message.fileSize {
                        Text(ByteCountFormatter.string(fromByteCount: Int64(size), countStyle: .file))
                    }
                    Text(message.date, format: .dateTime.day().month().year())
                }
                .font(.caption2)
                .foregroundStyle(.secondary)
            }
            Spacer(minLength: 0)
        }
        .padding(.vertical, 8)
    }

    private func note(_ text: String) -> some View {
        Text(text)
            .font(.subheadline)
            .foregroundStyle(.secondary)
            .multilineTextAlignment(.center)
            .frame(maxWidth: .infinity)
            .padding(.horizontal).padding(.vertical, 24)
    }

    /// Search this thread's media as the user types. `debounce` waits ~300 ms and
    /// bails if the query changed meanwhile, so live typing doesn't fire an SLM
    /// translation on every keystroke; results are also dropped if stale.
    private func runSearch(debounce: Bool) {
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty else { results = nil; isSearching = false; return }
        isSearching = true
        Task {
            if debounce {
                try? await Task.sleep(nanoseconds: 300_000_000)
                guard query.trimmingCharacters(in: .whitespacesAndNewlines) == q else { return }
            }
            let hits = await store.searchImages(q, in: conversation)
            // Apply only if the field still holds this query (drop stale results).
            guard query.trimmingCharacters(in: .whitespacesAndNewlines) == q else { return }
            results = hits
            isSearching = false
        }
    }
}
