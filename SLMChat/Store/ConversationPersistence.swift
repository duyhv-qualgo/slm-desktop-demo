import Foundation

/// Codable snapshot of one conversation for on-disk storage. The `Conversation`
/// class itself stays a `@MainActor ObservableObject`; we serialize this flat
/// value type instead so persistence stays independent of Combine/UI state.
struct StoredConversation: Codable {
    var contactName: String
    var messages: [StoredMessage]
}

/// Reads/writes the whole inbox as a single JSON file in the app's Documents
/// directory. Small payloads (a few KB), so a plain synchronous atomic write on
/// the main actor is fine for this demo.
enum ConversationPersistence {

    private static let filename = "conversations.json"

    private static var fileURL: URL {
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return dir.appendingPathComponent(filename)
    }

    /// Returns the stored inbox, or `nil` if nothing has been saved yet (first
    /// launch) or the file is unreadable/corrupt.
    static func load() -> [StoredConversation]? {
        guard let data = try? Data(contentsOf: fileURL) else { return nil }
        return try? JSONDecoder().decode([StoredConversation].self, from: data)
    }

    /// Persist the whole inbox, overwriting any previous file.
    static func save(_ conversations: [StoredConversation]) {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted]
        guard let data = try? encoder.encode(conversations) else { return }
        try? data.write(to: fileURL, options: .atomic)
    }
}
