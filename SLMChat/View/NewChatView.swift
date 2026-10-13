import SwiftUI

/// "New message" people picker: a searchable list of the predefined roster. Tapping
/// a person opens (or creates) a conversation with them and pushes it in the inbox.
struct NewChatView: View {
    @ObservedObject var store: ConversationStore
    @Binding var isPresented: Bool
    /// Called with the opened/created conversation so the caller can navigate to it.
    var onSelect: (Conversation) -> Void

    @State private var query = ""

    private var results: [String] {
        query.isEmpty
            ? store.roster
            : store.roster.filter { $0.localizedCaseInsensitiveContains(query) }
    }

    var body: some View {
        NavigationStack {
            List(results, id: \.self) { name in
                Button {
                    onSelect(store.startConversation(with: name))
                    isPresented = false
                } label: {
                    personRow(name)
                }
                .buttonStyle(.plain)
            }
            .listStyle(.plain)
            .searchable(text: $query, prompt: "Search people")
            .navigationTitle("New Message")
            .inlineNavigationTitle()
            .toolbar {
                ToolbarItem(placement: .leadingBar) {
                    Button("Cancel") { isPresented = false }
                }
            }
        }
    }

    private func personRow(_ name: String) -> some View {
        HStack(spacing: 12) {
            Text(Avatar.initials(name))
                .font(.headline.weight(.semibold))
                .foregroundStyle(.white)
                .frame(width: 42, height: 42)
                .background(Avatar.tint(name).gradient)
                .clipShape(Circle())

            VStack(alignment: .leading, spacing: 2) {
                Text(name).font(.body.weight(.medium))
                if store.conversations.contains(where: { $0.contactName == name }) {
                    Text("Existing chat")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
            Spacer()
        }
        .padding(.vertical, 2)
        .contentShape(Rectangle())
    }
}
