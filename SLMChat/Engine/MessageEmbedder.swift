import Foundation
import ImageIO
import UniformTypeIdentifiers
import LiteRTLM

/// On-device embeddings (EmbeddingGemma 2 text+vision, 440M, via LiteRT-LM) for
/// semantic message search. Text and photos land in one space, in any of 100+
/// languages: "hoa hướng dương" finds a photo of a sunflower, and "dinner
/// plans" finds "tối nay ăn gì?".
///
/// E2EE: messages, photos and their vectors never leave this process — no
/// network, no logging of content. A vector is close to a copy of what it was
/// made from (inversion attacks recover much of it), so whatever stores them
/// must treat them like the messages themselves.
///
/// Runs on the **CPU** backend on purpose. GPU is ~2× faster per text (20 vs
/// 37 ms on M-series) but takes ~2.7 s to initialise and would contend with
/// llama.cpp for Metal, which owns the chat latency the user actually feels.
/// Search embeds one short query, so 37 ms is invisible; a photo takes ~0.5 s,
/// once, in the background.
///
/// The text tower is the same as the text-only 270M model (identical vectors),
/// so this model costs only the 170M vision encoder on top.
actor MessageEmbedder {

    /// Vector size kept per message. Matryoshka: 256 scores the same as 768 on
    /// our EN/VI check (MRR 1.0) at a third of the storage; 128 drops to 0.96.
    static let dimension = 256
    /// Bumped whenever anything that changes vectors changes (model, prefixes,
    /// dimension, image preprocessing), so a stored index built the old way is
    /// discarded, not mixed.
    static let version = "embeddinggemma-2-tv-440m/prefix-v1/img-1024/\(dimension)"

    private static let modelName = "embeddinggemma-2-text-vision-440m"
    /// Tokens per text input. Chat messages are short; longer ones are
    /// truncated, which is fine for search.
    private static let maxInputLength = 256
    /// Photos are downscaled to this longest side before embedding. The encoder
    /// resizes to its own input anyway; this only keeps a 48 MP photo from being
    /// decoded at full size.
    private static let maxImageSide = 1024

    private var engine: EmbeddingEngine?
    private let options = EmbeddingOptions(normalize: true, outputSize: dimension)

    var isLoaded: Bool { engine != nil }

    /// Load the model. Throws if the `.litertlm` is missing from the bundle.
    func load() async throws {
        guard engine == nil else { return }
        guard let path = ResourceLookup.path(Self.modelName, ext: "litertlm") else {
            throw CocoaError(.fileNoSuchFile,
                             userInfo: [NSFilePathErrorKey: "\(Self.modelName).litertlm"])
        }
        let e = EmbeddingEngine(config: EmbeddingEngineConfig(
            modelPath: path, backend: .cpu(), visionBackend: .cpu(),
            cacheDir: Self.cacheDirectory(), maxInputLength: Self.maxInputLength))
        try await e.initialize()
        engine = e
    }

    /// Where XNNPACK keeps its prepacked weights (~one file per encoder). By
    /// default LiteRT-LM writes them next to the model — inside the app bundle,
    /// which breaks its code signature and is read-only once installed.
    /// Caches is right: the files are derived and rebuilt if deleted.
    private static func cacheDirectory() -> String? {
        guard let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
        else { return nil }
        let dir = base.appendingPathComponent("SLMChat/litert", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.path
    }

    func unload() async {
        await engine?.close()
        engine = nil
    }

    /// Embed a search query. Unit length, `dimension` floats.
    func embed(query: String) async throws -> [Float] {
        try await run([[.text(Self.queryPrompt(query))]])[0]
    }

    /// Embed messages for the index, in one batch call. Unit length.
    func embed(documents: [String]) async throws -> [[Float]] {
        guard !documents.isEmpty else { return [] }
        return try await run(documents.map { [.text(Self.documentPrompt($0))] })
    }

    /// Embed a photo (any format ImageIO reads). Unit length.
    func embed(image data: Data) async throws -> [Float] {
        try await run([[.imageData(Self.downscaled(data) ?? data)]])[0]
    }

    // EmbeddingGemma's asymmetric retrieval prompts. The v2 model card doesn't
    // list any; these are v1's, and they matter: on our EN/VI check top-1 goes
    // from 9/12 without them to 12/12 with them. Photos go in bare.
    static func queryPrompt(_ q: String) -> String { "task: search result | query: \(q)" }
    static func documentPrompt(_ d: String) -> String { "title: none | text: \(d)" }

    private func run(_ inputs: [[Content]]) async throws -> [[Float]] {
        guard let engine else { throw CocoaError(.featureUnsupported) }
        let responses = try await engine.computeEmbeddingBatch(contentsBatch: inputs, options: options)
        return responses.map(\.embedding)
    }

    /// JPEG at most `maxImageSide` on the long edge, EXIF orientation applied.
    private static func downscaled(_ data: Data) -> Data? {
        guard let src = CGImageSourceCreateWithData(data as CFData, nil),
              let image = CGImageSourceCreateThumbnailAtIndex(src, 0, [
                  kCGImageSourceCreateThumbnailFromImageAlways: true,
                  kCGImageSourceCreateThumbnailWithTransform: true,
                  kCGImageSourceThumbnailMaxPixelSize: maxImageSide,
              ] as CFDictionary) else { return nil }
        let out = NSMutableData()
        guard let dst = CGImageDestinationCreateWithData(out, UTType.jpeg.identifier as CFString, 1, nil)
        else { return nil }
        CGImageDestinationAddImage(dst, image, [kCGImageDestinationLossyCompressionQuality: 0.9] as CFDictionary)
        return CGImageDestinationFinalize(dst) ? out as Data : nil
    }
}
