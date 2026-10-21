import Foundation
import Combine
import CryptoKit

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
    /// Cosine similarity for meaning matches; 1 for exact text matches.
    let score: Float
    /// Matched by text (true) or only by meaning (false).
    let exact: Bool
    /// The meaning match came from the photo's pixels, not its text.
    var viaPhoto = false
    var id: UUID { message.id }
}

/// Ask a thread to scroll to a message and flash it. `token` makes a second
/// click on the same result count as a new request.
struct JumpRequest: Equatable {
    let messageID: UUID
    let token = UUID()
}

/// Semantic + exact search over every message in the inbox and the group.
///
/// Vectors come from `MessageEmbedder`. Text is keyed by a hash of the indexed
/// text, photos by their `ImageStore` id — never by message id, which is
/// regenerated on every launch. A photo message gets both: its caption/tags
/// and the picture itself. The index is built
/// on this device from the messages and nothing else: on a fresh install it
/// starts empty and fills in the background, newest messages first.
///
/// E2EE: the store holds a near-copy of every message, so it stays on device
/// (excluded from backup), never logs text, and drops a vector as soon as no
/// message with that text remains. Production TODO: encrypt the file at rest
/// with a device-only Keychain key; not done in this sample.
@MainActor
final class MessageSearch: ObservableObject {

    /// Messages with a vector / messages eligible for one. Drives the sidebar's
    /// "Indexing…" line.
    @Published private(set) var indexedCount = 0
    @Published private(set) var eligibleCount = 0
    /// The embedding model loaded; meaning matches are available.
    @Published private(set) var isAvailable = false

    var isIndexing: Bool { isAvailable && indexedCount < eligibleCount }

    private unowned let store: ConversationStore
    private let embedder = MessageEmbedder()
    private var vectors: [String: [Float]] = [:]
    private var refreshTask: Task<Void, Never>?
    private var refreshAgain = false
    private var pendingRefresh: Task<Void, Never>?
    private var groupWatch: AnyCancellable?

    // Ranking. Measured on the EN/VI checks in MessageEmbedderTests: relevant
    // messages score 0.75–0.91, unrelated ones up to ~0.72, and very short
    // messages ("ok", "haha") 0.66–0.77 against anything — hence the length
    // floor, the absolute floor and the cut relative to the best hit.
    private static let minScore: Float = 0.74
    private static let maxDropFromBest: Float = 0.08
    // Text→photo similarities run lower: the right photo scored 0.67–0.81,
    // the runner-up 0.60–0.80 (often also relevant: "string instrument" →
    // violin, then guitar). Photos are cut on their own scale, then nudged up
    // by the gap so they interleave fairly with text hits.
    private static let photoMinScore: Float = 0.68
    private static let photoMaxDropFromBest: Float = 0.06
    private static let photoScoreOffset: Float = 0.06
    private static let maxResults = 20
    private static let batchSize = 16

    init(store: ConversationStore) {
        self.store = store
    }

    // MARK: - Lifecycle

    /// Load the model and the stored index, then index whatever is missing.
    /// Call after the chat model has warmed up, so this never delays it.
    func start() async {
        guard !isAvailable else { return }
        #if DEBUG
        // Dev aid for memory comparisons: SLMCHAT_NO_SEARCH=1 skips the embedder.
        if ProcessInfo.processInfo.environment["SLMCHAT_NO_SEARCH"] != nil { return }
        #endif
        do {
            try await embedder.load()
        } catch {
            print("MessageSearch: embedding model unavailable — exact search only")
            return
        }
        vectors = EmbeddingIndexFile.load() ?? [:]
        isAvailable = true
        groupWatch = store.group.$messages
            .debounce(for: .seconds(1.5), scheduler: RunLoop.main)
            .sink { [weak self] _ in self?.refresh() }
        refresh()
    }

    func shutdown() async {
        refreshTask?.cancel()
        await embedder.unload()
    }

    /// Index new messages and drop vectors no message uses any more. Cheap to
    /// call often: coalesced, and a no-op when nothing changed.
    func scheduleRefresh() {
        pendingRefresh?.cancel()
        pendingRefresh = Task { [weak self] in
            // Let a streamed reply finish before it's embedded.
            try? await Task.sleep(for: .seconds(1.5))
            guard !Task.isCancelled else { return }
            self?.refresh()
        }
    }

    private func refresh() {
        guard isAvailable else { return }
        if refreshTask != nil { refreshAgain = true; return }
        refreshTask = Task { [weak self] in
            await self?.runRefresh()
            guard let self else { return }
            self.refreshTask = nil
            if self.refreshAgain { self.refreshAgain = false; self.refresh() }
        }
    }

    private func runRefresh() async {
        let entries = allMessages()
        let textKeys = Set(entries.compactMap { Self.indexText($0.message).map(Self.key) })
        let photoIDs = entries.compactMap { e in e.message.imageID.flatMap { FileManager.default.fileExists(atPath: ImageStore.url(for: $0).path) ? $0 : nil } }
        let photoKeys = Set(photoIDs.map(Self.photoKey))
        eligibleCount = textKeys.count + photoKeys.count

        // Deleted messages take their vectors with them.
        var changed = false
        for k in vectors.keys where !textKeys.contains(k) && !photoKeys.contains(k) {
            vectors[k] = nil; changed = true
        }

        // Newest first, so recent chats become searchable soonest. Text first:
        // it's ~40 ms a message against ~0.5 s a photo.
        var seen = Set<String>()
        let texts = entries.reversed()
            .compactMap { Self.indexText($0.message) }
            .filter { vectors[Self.key($0)] == nil && seen.insert(Self.key($0)).inserted }
        let photos = photoIDs.reversed()
            .filter { vectors[Self.photoKey($0)] == nil && seen.insert(Self.photoKey($0)).inserted }
        indexedCount = eligibleCount - texts.count - photos.count

        for start in stride(from: 0, to: texts.count, by: Self.batchSize) {
            if Task.isCancelled { break }
            let batch = Array(texts[start..<min(start + Self.batchSize, texts.count)])
            guard let vs = try? await embedder.embed(documents: batch) else { break }
            for (text, v) in zip(batch, vs) { vectors[Self.key(text)] = v }
            indexedCount += batch.count
            changed = true
            EmbeddingIndexFile.save(vectors)   // a quit mid-way loses nothing
        }
        for id in photos {
            if Task.isCancelled { break }
            guard let data = ImageStore.data(for: id),
                  let v = try? await embedder.embed(image: data) else { continue }
            vectors[Self.photoKey(id)] = v
            indexedCount += 1
            changed = true
            EmbeddingIndexFile.save(vectors)
        }
        if changed { EmbeddingIndexFile.save(vectors) }
    }

    // MARK: - Search

    func search(_ rawQuery: String) async -> [MessageHit] {
        let query = rawQuery.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !query.isEmpty else { return [] }
        let entries = allMessages()

        var hits: [MessageHit] = []
        var exactIDs = Set<UUID>()
        for e in entries.reversed() where e.message.text.localizedCaseInsensitiveContains(query) {
            hits.append(e.hit(score: 1, exact: true))
            exactIDs.insert(e.message.id)
        }

        // Meaning matches: not for one- or two-letter queries, which embed to noise.
        guard isAvailable, query.count >= 3,
              let qv = try? await embedder.embed(query: query) else {
            return Array(hits.prefix(Self.maxResults))
        }
        var textScores: [(Entry, Float)] = []
        var photoScores: [(Entry, Float)] = []
        for e in entries where !exactIDs.contains(e.message.id) {
            if let text = Self.indexText(e.message), let v = vectors[Self.key(text)] {
                textScores.append((e, Self.dot(qv, v)))
            }
            if let id = e.message.imageID, let v = vectors[Self.photoKey(id)] {
                photoScores.append((e, Self.dot(qv, v)))
            }
        }
        let textHits = Self.passing(textScores, minScore: Self.minScore, maxDrop: Self.maxDropFromBest)
            .map { $0.0.hit(score: $0.1, exact: false) }
        let photoHits = Self.passing(photoScores, minScore: Self.photoMinScore, maxDrop: Self.photoMaxDropFromBest)
            .map { var h = $0.0.hit(score: $0.1 + Self.photoScoreOffset, exact: false); h.viaPhoto = true; return h }

        // One row per message: a photo whose caption also matched keeps the
        // better of the two.
        var best: [UUID: MessageHit] = [:]
        for h in textHits + photoHits where (best[h.id]?.score ?? -1) < h.score { best[h.id] = h }
        hits += best.values.sorted { $0.score > $1.score }
        return Array(hits.prefix(Self.maxResults))
    }

    // MARK: - Helpers

    private struct Entry {
        let message: ChatMessage
        let location: MessageLocation
        let title: String
        func hit(score: Float, exact: Bool) -> MessageHit {
            MessageHit(message: message, location: location, chatTitle: title, score: score, exact: exact)
        }
    }

    /// Every message, oldest first per chat.
    private func allMessages() -> [Entry] {
        var out: [Entry] = []
        for c in store.conversations {
            out += c.messages.map { Entry(message: $0, location: .conversation(c), title: c.contactName) }
        }
        out += store.group.messages.map { Entry(message: $0, location: .group, title: store.group.title) }
        return out
    }

    /// What gets embedded for a message: its text, plus image tags and file name
    /// so "beach" finds a photo. `nil` = too short to mean anything ("ok", "👍").
    static func indexText(_ m: ChatMessage) -> String? {
        var parts = [m.text.trimmingCharacters(in: .whitespacesAndNewlines)]
        if let tags = m.imageTags, !tags.isEmpty { parts.append("photo: " + tags.prefix(8).joined(separator: ", ")) }
        if let name = m.fileName { parts.append("file: " + name) }
        let text = parts.filter { !$0.isEmpty }.joined(separator: "\n")
        let letters = text.unicodeScalars.filter { CharacterSet.letters.contains($0) }.count
        let words = text.split(whereSeparator: \.isWhitespace).count
        return letters >= 8 && words >= 2 ? text : nil
    }

    /// Above the absolute floor and within `maxDrop` of the best score.
    private static func passing(_ scored: [(Entry, Float)], minScore: Float, maxDrop: Float) -> [(Entry, Float)] {
        guard let top = scored.map(\.1).max() else { return [] }
        let floor = max(minScore, top - maxDrop)
        return scored.filter { $0.1 >= floor }
    }

    private static func photoKey(_ imageID: String) -> String { "img:" + imageID }

    private static func key(_ text: String) -> String {
        SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    private static func dot(_ a: [Float], _ b: [Float]) -> Float {
        var s: Float = 0
        for i in 0..<min(a.count, b.count) { s += a[i] * b[i] }
        return s
    }
}

/// On-disk vectors: `Application Support/SLMChat/embeddings.plist`, excluded from
/// backup. Tagged with `MessageEmbedder.version`; a file from another model or
/// prefix scheme is discarded rather than mixed with new vectors.
private enum EmbeddingIndexFile {
    private struct Stored: Codable {
        var version: String
        var vectors: [String: Data]
    }

    private static var url: URL {
        let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("SLMChat", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("embeddings.plist")
    }

    static func load() -> [String: [Float]]? {
        guard let data = try? Data(contentsOf: url),
              let stored = try? PropertyListDecoder().decode(Stored.self, from: data),
              stored.version == MessageEmbedder.version else { return nil }
        return stored.vectors.mapValues { d in
            d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        }
    }

    static func save(_ vectors: [String: [Float]]) {
        let stored = Stored(version: MessageEmbedder.version,
                            vectors: vectors.mapValues { v in v.withUnsafeBufferPointer { Data(buffer: $0) } })
        let encoder = PropertyListEncoder()
        encoder.outputFormat = .binary
        guard let data = try? encoder.encode(stored) else { return }
        var target = url
        try? data.write(to: target, options: .atomic)
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try? target.setResourceValues(values)
    }
}
