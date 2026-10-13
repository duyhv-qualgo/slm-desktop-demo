import SwiftUI

/// Offline image search over chat-attached photos, using the tags the on-device
/// EdgeNeXt model assigned when each image was sent. Vietnamese queries are
/// translated to English (the tag vocabulary is English) before matching — see
/// `ConversationStore.searchImages`.
struct ImageSearchView: View {
    @ObservedObject var store: ConversationStore
    @Environment(\.dismiss) private var dismiss

    @State private var query = ""
    @State private var results: [ConversationStore.ImageSearchHit] = []
    @State private var isSearching = false
    @State private var hasSearched = false

    private let columns = [GridItem(.adaptive(minimum: 108), spacing: 8)]

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                searchField
                Divider()
                content
            }
            .navigationTitle("Search Photos")
            .inlineNavigationTitle()
            .toolbar {
                ToolbarItem(placement: .trailingBar) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    private var searchField: some View {
        HStack(spacing: 8) {
            Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
            TextField("dog, beach, birthday cake…", text: $query)
                .textFieldStyle(.plain)
                .submitLabel(.search)
                .onSubmit { runSearch() }
            if isSearching { ProgressView().controlSize(.small) }
            else if !query.isEmpty {
                Button { query = ""; results = []; hasSearched = false } label: {
                    Image(systemName: "xmark.circle.fill").foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.horizontal, 12).padding(.vertical, 10)
        .background(Color.secondaryBackground)
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .padding()
    }

    @ViewBuilder
    private var content: some View {
        if !store.hasTaggedImages {
            emptyState(icon: "photo.on.rectangle.angled",
                       title: "No photos yet",
                       message: "Attach a photo in any chat. It's tagged on-device so you can search it here.")
        } else if hasSearched && results.isEmpty && !isSearching {
            emptyState(icon: "magnifyingglass",
                       title: "No matches",
                       message: "No attached photo matched “\(query)”. Try a different word.")
        } else {
            ScrollView {
                LazyVGrid(columns: columns, spacing: 8) {
                    ForEach(results) { hit in
                        resultCell(hit)
                    }
                }
                .padding()
            }
        }
    }

    private func resultCell(_ hit: ConversationStore.ImageSearchHit) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            StoredImageThumb(imageID: hit.message.imageID)
                .frame(height: 108)
                .frame(maxWidth: .infinity)
                .clipShape(RoundedRectangle(cornerRadius: 10))
            Text(hit.conversation.contactName)
                .font(.caption2.weight(.semibold))
                .lineLimit(1)
            Text(hit.matchedTags.prefix(3).joined(separator: ", "))
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(1)
        }
    }

    private func emptyState(icon: String, title: String, message: String) -> some View {
        VStack(spacing: 10) {
            Image(systemName: icon).font(.largeTitle).foregroundStyle(.secondary)
            Text(title).font(.headline)
            Text(message)
                .font(.subheadline).foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding(40)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func runSearch() {
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty else { results = []; hasSearched = false; return }
        isSearching = true
        Task {
            let hits = await store.searchImages(q)
            results = hits
            hasSearched = true
            isSearching = false
        }
    }
}

/// A square thumbnail loaded from `ImageStore` off the main thread. Shared by the
/// global image search and the per-contact media grid (`ContactInfoView`).
struct StoredImageThumb: View {
    let imageID: String?
    @State private var image: PlatformImage?

    var body: some View {
        Group {
            if let image {
                Image(platformImage: image).resizable().scaledToFill()
            } else {
                Rectangle().fill(Color.gray.opacity(0.2)).overlay(ProgressView())
            }
        }
        .task(id: imageID) {
            guard image == nil, let id = imageID else { return }
            image = await Task.detached(priority: .utility) {
                ImageStore.data(for: id).flatMap(PlatformImage.init(data:))
            }.value
        }
    }
}
