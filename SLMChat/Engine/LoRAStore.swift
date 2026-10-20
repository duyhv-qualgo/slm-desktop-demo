import CryptoKit
import Foundation

/// Install state of one language pack, as Settings renders it.
enum LoRAInstallState: Equatable {
    case notInstalled
    /// 0…1, or nil while the server hasn't declared a length.
    case downloading(progress: Double?)
    case installed
    case failed(String)

    var isFailure: Bool {
        if case .failed = self { return true }
        return false
    }
}

/// Downloads, verifies and installs LoRA language packs.
///
/// Installation is atomic: the file is fetched to a temp location, hashed, and
/// only then moved into place. A partial or corrupt download therefore can't
/// become the file the engine loads — which matters more here than for most
/// downloads, because llama.cpp will happily accept a structurally valid adapter
/// that is the wrong one and produce fluent, wrong output rather than an error.
@MainActor
final class LoRAStore: ObservableObject {

    /// Install state keyed by `LoRAAdapter.id`. Adapters absent from the map are
    /// `.notInstalled`; use `state(of:)` rather than reading this directly.
    @Published private(set) var states: [String: LoRAInstallState] = [:]

    private var tasks: [String: Task<Void, Never>] = [:]

    init() { refresh() }

    /// Re-derive install state from disk. Cheap (a few `fileExists` calls), so
    /// it's safe to call whenever Settings appears.
    func refresh() {
        for adapter in LoRACatalog.allAdapters {
            // Don't stomp a download in flight with a disk check that is, by
            // definition, about to be out of date.
            if case .downloading = states[adapter.id] { continue }
            states[adapter.id] = LoRACatalog.isInstalled(adapter) ? .installed : .notInstalled
        }
    }

    func state(of adapter: LoRAAdapter) -> LoRAInstallState {
        states[adapter.id] ?? .notInstalled
    }

    var isBusy: Bool {
        states.values.contains { if case .downloading = $0 { return true }; return false }
    }

    /// Fetch and install a pack. No-op if it's already installed or in flight.
    func download(_ adapter: LoRAAdapter) {
        guard tasks[adapter.id] == nil else { return }
        if case .installed = state(of: adapter) { return }

        var source: URL?
        if !adapter.simulatedDownload {
            guard let url = adapter.remoteURL else {
                states[adapter.id] = .failed("No download URL — set LoRACatalog.remoteBaseURL, "
                                           + "or copy \(adapter.fileName) into Documents/.")
                return
            }
            source = url
        }

        states[adapter.id] = .downloading(progress: nil)
        tasks[adapter.id] = Task { [weak self] in
            do {
                let report: (Double?) -> Void = { fraction in
                    Task { @MainActor in
                        self?.reportProgress(fraction, for: adapter.id)
                    }
                }
                if let source {
                    try await Self.fetchAndInstall(adapter, from: source, onProgress: report)
                } else {
                    try await Self.simulateInstall(adapter, onProgress: report)
                }
                self?.states[adapter.id] = .installed
            } catch is CancellationError {
                self?.states[adapter.id] = .notInstalled
            } catch {
                NSLog("[LoRA] download failed for %@: %@", adapter.id, error.localizedDescription)
                self?.states[adapter.id] = .failed(error.localizedDescription)
            }
            self?.tasks[adapter.id] = nil
        }
    }

    /// Publish a progress tick, but only while the pack is still downloading.
    ///
    /// Progress is produced off the main actor (a background copy loop, or
    /// URLSession's delegate queue) and reaches this class as a separate hop, so
    /// the FINAL tick can land after the install has already completed. Without
    /// this guard that late tick overwrites `.installed` with
    /// `.downloading(1.0)` — the row sticks at 100% forever and anything waiting
    /// on completion never sees it finish.
    private func reportProgress(_ fraction: Double?, for id: String) {
        guard case .downloading = states[id] else { return }
        states[id] = .downloading(progress: fraction)
    }

    /// Download (or simulate) a pack and return once it has settled — installed,
    /// failed or cancelled. Preferred over `download` plus polling: the terminal
    /// state can't be missed, and there's no poll interval to tune.
    func downloadAndWait(_ adapter: LoRAAdapter) async {
        download(adapter)
        await tasks[adapter.id]?.value
    }

    func cancel(_ adapter: LoRAAdapter) {
        tasks[adapter.id]?.cancel()
        tasks[adapter.id] = nil
        states[adapter.id] = .notInstalled
    }

    /// Delete a downloaded pack. The engine keeps using it until the next reload,
    /// so callers that care should reload after this.
    func remove(_ adapter: LoRAAdapter) {
        do {
            try LoRACatalog.removeDownload(adapter)
            // A side-loaded or bundled copy may still satisfy this adapter, so
            // ask the catalog rather than assuming it's gone.
            states[adapter.id] = LoRACatalog.isInstalled(adapter) ? .installed : .notInstalled
        } catch {
            states[adapter.id] = .failed(error.localizedDescription)
        }
    }

    // MARK: - Download

    private static func fetchAndInstall(_ adapter: LoRAAdapter,
                                        from url: URL,
                                        onProgress: @escaping (Double?) -> Void) async throws {
        let progress = ProgressForwarder(onProgress)
        let (tempURL, response) = try await URLSession.shared.download(from: url, delegate: progress)

        // download() deletes tempURL as soon as we return, so everything below
        // has to happen before this function unwinds.
        if let http = response as? HTTPURLResponse, !(200...299).contains(http.statusCode) {
            throw LoRAError.http(http.statusCode)
        }
        try Task.checkCancellation()

        if let expected = adapter.sha256 {
            let actual = try sha256Hex(of: tempURL)
            guard actual.caseInsensitiveCompare(expected) == .orderedSame else {
                throw LoRAError.hashMismatch(expected: expected, actual: actual)
            }
        }

        let destination = LoRACatalog.installURL(for: adapter)
        let fm = FileManager.default
        if fm.fileExists(atPath: destination.path) {
            try fm.removeItem(at: destination)
        }
        try fm.moveItem(at: tempURL, to: destination)
        NSLog("[LoRA] installed %@ (%@)", adapter.fileName, adapter.displaySize)
    }

    /// Install a `simulatedDownload` pack from the app bundle, paced so the UI
    /// shows the same available → downloading → installed progression a real fetch
    /// would.
    ///
    /// Only the transport is faked. The file still lands in the install directory
    /// through the same path a download uses, so what gets loaded afterwards, and
    /// the reload it triggers, are exactly what a hosted pack would produce.
    private static func simulateInstall(_ adapter: LoRAAdapter,
                                        onProgress: @escaping (Double?) -> Void) async throws {
        guard let source = LoRACatalog.bundledPath(for: adapter) else {
            throw LoRAError.missingBundledSource(adapter.fileName)
        }

        let steps = 24
        let perStep = UInt64(simulatedDuration / Double(steps) * 1_000_000_000)
        for step in 1...steps {
            try Task.checkCancellation()
            try await Task.sleep(nanoseconds: perStep)
            onProgress(Double(step) / Double(steps))
        }
        try Task.checkCancellation()

        let destination = LoRACatalog.installURL(for: adapter)
        let fm = FileManager.default
        if fm.fileExists(atPath: destination.path) {
            try fm.removeItem(at: destination)
        }
        try fm.copyItem(at: URL(fileURLWithPath: source), to: destination)
        NSLog("[LoRA] installed %@ (%@, simulated)", adapter.fileName, adapter.displaySize)
    }

    /// How long a simulated install takes. Long enough to read the progress, short
    /// enough not to stall a demo.
    private static let simulatedDuration: Double = 2.5

    /// Streaming SHA-256 — a language pack is tens of megabytes and there is no
    /// reason to hold one in memory to hash it.
    private static func sha256Hex(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let chunk = try handle.read(upToCount: 1 << 20), !chunk.isEmpty {
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}

enum LoRAError: LocalizedError {
    case http(Int)
    case hashMismatch(expected: String, actual: String)
    case missingBundledSource(String)

    var errorDescription: String? {
        switch self {
        case .http(let code):
            return "Server returned HTTP \(code)."
        case .hashMismatch(let expected, let actual):
            return "Checksum mismatch — the file is not the expected adapter "
                 + "(wanted \(expected.prefix(12))…, got \(actual.prefix(12))…)."
        case .missingBundledSource(let name):
            return "\(name) isn't in the app bundle — a simulated pack needs its "
                 + "file in Resources/."
        }
    }
}

/// Bridges `URLSession`'s delegate-based progress onto a closure so the async
/// `download(from:delegate:)` call can still report bytes as they arrive.
private final class ProgressForwarder: NSObject, URLSessionDownloadDelegate {
    private let onProgress: (Double?) -> Void

    init(_ onProgress: @escaping (Double?) -> Void) {
        self.onProgress = onProgress
    }

    func urlSession(_ session: URLSession,
                    downloadTask: URLSessionDownloadTask,
                    didWriteData bytesWritten: Int64,
                    totalBytesWritten: Int64,
                    totalBytesExpectedToWrite totalBytesExpectedToWrite: Int64) {
        guard totalBytesExpectedToWrite > 0 else { onProgress(nil); return }
        onProgress(Double(totalBytesWritten) / Double(totalBytesExpectedToWrite))
    }

    /// Required by the protocol; the async `download` API consumes the file, so
    /// there is nothing to do here.
    func urlSession(_ session: URLSession,
                    downloadTask: URLSessionDownloadTask,
                    didFinishDownloadingTo location: URL) {}
}
