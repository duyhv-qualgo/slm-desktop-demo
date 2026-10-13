import Foundation

/// On-disk store for chat file (document) attachments — the non-image counterpart
/// of `ImageStore`. Bytes live under `Documents/files/<id>` and a message keeps only
/// the `id` plus the original display name, so the conversation JSON stays small.
enum FileStore {

    private static var dir: URL {
        let base = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let d = base.appendingPathComponent("files", isDirectory: true)
        if !FileManager.default.fileExists(atPath: d.path) {
            try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        }
        return d
    }

    static func url(for id: String) -> URL {
        dir.appendingPathComponent(id)
    }

    /// Persist file bytes and return the new id, or nil on write failure.
    static func save(_ data: Data) -> String? {
        let id = UUID().uuidString
        do {
            try data.write(to: url(for: id), options: .atomic)
            return id
        } catch {
            NSLog("[FileStore] save failed: %@", error.localizedDescription)
            return nil
        }
    }

    static func data(for id: String) -> Data? {
        try? Data(contentsOf: url(for: id))
    }

    static func delete(_ id: String) {
        try? FileManager.default.removeItem(at: url(for: id))
    }
}
