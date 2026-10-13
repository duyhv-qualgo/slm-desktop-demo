import Foundation

/// On-disk store for chat image attachments. Image bytes are kept as files under
/// `Documents/images/<id>.jpg` and referenced from a message by their `id` only,
/// so the conversation JSON (see `ConversationPersistence`) stays small — it never
/// inlines pixel data.
enum ImageStore {

    private static var dir: URL {
        let base = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let d = base.appendingPathComponent("images", isDirectory: true)
        if !FileManager.default.fileExists(atPath: d.path) {
            try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        }
        return d
    }

    static func url(for id: String) -> URL {
        dir.appendingPathComponent("\(id).jpg")
    }

    /// Persist image bytes and return the new id, or nil on write failure.
    static func save(_ data: Data) -> String? {
        let id = UUID().uuidString
        do {
            try data.write(to: url(for: id), options: .atomic)
            return id
        } catch {
            NSLog("[ImageStore] save failed: %@", error.localizedDescription)
            return nil
        }
    }

    /// Load previously stored image bytes, or nil if the file is gone.
    static func data(for id: String) -> Data? {
        try? Data(contentsOf: url(for: id))
    }

    /// Remove an image file (best-effort; used when a message/conversation is deleted).
    static func delete(_ id: String) {
        try? FileManager.default.removeItem(at: url(for: id))
    }
}
