import Foundation

/// Where a search hit lives, so the sidebar can open it.
enum MessageLocation: Hashable {
    case group
    case conversation(Conversation)
}

/// One message found by search.
struct MessageHit: Identifiable {
    let message: ChatMessage
    let location: MessageLocation
    /// Title of the chat it's in ("Mai", "Weekend crew").
    let chatTitle: String
    var id: UUID { message.id }
}

/// Ask a thread to scroll to a message and flash it. `token` makes a second
/// click on the same result count as a new request.
struct JumpRequest: Equatable {
    let messageID: UUID
    let token = UUID()
}

/// Exact text search over every message in the inbox and the group, newest
/// first. Reads the messages in memory; keeps no index and no copy of them.
@MainActor
final class MessageSearch {

    private static let maxResults = 20

    private unowned let store: ConversationStore

    init(store: ConversationStore) {
        self.store = store
    }

    func search(_ rawQuery: String) -> [MessageHit] {
        let query = rawQuery.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !query.isEmpty else { return [] }
        var hits: [MessageHit] = []
        for c in store.conversations {
            for m in c.messages where m.text.localizedCaseInsensitiveContains(query) {
                hits.append(MessageHit(message: m, location: .conversation(c), chatTitle: c.contactName))
            }
        }
        for m in store.group.messages where m.text.localizedCaseInsensitiveContains(query) {
            hits.append(MessageHit(message: m, location: .group, chatTitle: store.group.title))
        }
        hits.sort { $0.message.date > $1.message.date }
        return Array(hits.prefix(Self.maxResults))
    }
}
