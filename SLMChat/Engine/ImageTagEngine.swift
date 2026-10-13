import Foundation
#if canImport(UIKit)
import UIKit
#else
import ImageIO
#endif

/// One predicted tag with its probability.
struct ImageTag: Equatable {
    let name: String
    let score: Float
}

/// On-device image-tagging engine: preprocessing + the EdgeNeXt TFLite classifier
/// (via `ImageTagLM`), wrapped in a serialized async API — the same shape as
/// `GuardEngine`, but for vision instead of text.
///
/// Runs on its own **`.utility`** queue, deliberately lower priority than the LLM
/// engines, so tagging an attached photo never contends the UI. Multi-label:
/// every tag gets an independent probability (`sigmoid`), and we keep those above a
/// threshold, capped at `topK`, sorted by score.
final class ImageTagEngine: @unchecked Sendable {

    /// Input tensor channel layout, detected from the model at load.
    private enum Layout { case nhwc, nchw }

    /// Pixel-value scaling applied before the model. The right one depends on how
    /// the `.tflite` was exported: if normalization was baked into the graph (like
    /// the sigmoid was), the model wants raw `unit`/`signed` input and `imagenet`
    /// double-normalizes. `debugCompareNorms` logs all three so we can tell which.
    private enum Norm {
        case imagenet   // (v/255 - mean)/std   — README's external transform
        case unit       // v/255                — [0,1], if norm is baked in
        case signed     // v/127.5 - 1          — [-1,1], another common baked-in range
    }

    /// When true, `tag` runs the image through all three `Norm` variants and logs
    /// the top tags for each, to diagnose a baked-in-normalization mismatch. Set
    /// back to false once the correct `productionNorm` is confirmed.
    private static let debugCompareNorms = false
    /// The scaling used for real results (and the index written to messages).
    /// This `.tflite` was exported with BOTH the ImageNet normalization and the
    /// final sigmoid baked into the graph, so it wants raw `[0,1]` pixels — applying
    /// ImageNet normalization here double-normalizes and collapses every image to
    /// generic tags (verified on-device: `.unit` gave table/food/plate/red at 0.85+
    /// where `.imagenet` gave blue/sky/light). See `decode` for the matching
    /// already-sigmoided output handling.
    private static let productionNorm: Norm = .unit

    /// When true, logs decode/preprocess geometry and preprocessed-tensor stats
    /// (per-channel means, min/max, a sample pixel) so the app's input tensor can be
    /// diffed against the host script's. Turn off once parity is confirmed.
    private static let debugVerbose = false

    /// A/B DEBUG ONLY: skip EXIF orientation and feed the RAW sensor pixels, matching
    /// the host's `--no-exif-transpose` run — reproduces the sideways "bicycle/wall"
    /// output for parity checks. **MUST be false in production**: it tags rotated
    /// photos sideways (~0.42 probability error).
    private static let debugIgnoreExif = false

    // All mutable state is touched only on this serial queue, so the type is safe
    // to share across the async boundary.
    private let queue = DispatchQueue(label: "com.example.slmchat.imagetag", qos: .utility)
    private let bridge = ImageTagLM()
    private var tags: [String] = []
    private var side = 224
    private var layout: Layout = .nhwc
    private var loaded = false

    /// ImageNet normalization stats — the training-time eval transform (see the
    /// image-tagging release README). Must match, or scores drift.
    private static let mean: [Float] = [0.485, 0.456, 0.406]
    private static let std:  [Float] = [0.229, 0.224, 0.225]

    /// The eval transform (torchvision `T.Resize(256)`) scales the image's SHORTER
    /// side to this, then `T.CenterCrop(224)` takes the middle `side` (224). See
    /// `preprocess`, which reproduces resize-then-crop to match the host script.
    private static let resizeShortSide = 256

    /// Default per-tag probability cutoff. Raised above the contract's 0.5 to keep
    /// only high-confidence tags (fewer, cleaner tags per image → more precise
    /// search). Applies to real probabilities (the graph's sigmoid output; see
    /// `decode`).
    static let defaultThreshold: Float = 0.7
    /// Max tags kept per image, so a busy photo doesn't bloat the index.
    static let defaultTopK = 20

    // MARK: - Load

    /// Load the classifier + tag list once. Serialized on the engine queue.
    func warmUp(modelName: String, tagListName: String) async -> (ok: Bool, error: String?) {
        await withCheckedContinuation { cont in
            queue.async { [weak self] in
                guard let self else { cont.resume(returning: (false, "deallocated")); return }
                guard let modelPath = ResourceLookup.path(modelName, ext: "tflite") else {
                    cont.resume(returning: (false, "\(modelName).tflite not in bundle — add it to Resources/"))
                    return
                }
                guard let listURL = ResourceLookup.url(tagListName, ext: "txt"),
                      let raw = try? String(contentsOf: listURL, encoding: .utf8) else {
                    cont.resume(returning: (false, "\(tagListName).txt missing or unreadable"))
                    return
                }
                guard self.bridge.load(modelPath: modelPath) else {
                    cont.resume(returning: (false, self.bridge.lastError))
                    return
                }

                self.tags = raw.split(separator: "\n", omittingEmptySubsequences: false)
                    .map { $0.trimmingCharacters(in: .whitespaces) }
                    .filter { !$0.isEmpty }

                // Detect spatial size + channel layout from the input tensor shape,
                // so an NHWC or NCHW export both work without a rebuild.
                let shape = self.bridge.inputShape.map { $0.intValue }
                if shape.count == 4 {
                    if shape[3] == 3 { self.layout = .nhwc; self.side = shape[1] }
                    else if shape[1] == 3 { self.layout = .nchw; self.side = shape[2] }
                }

                // The tag list is the model's output contract: line i names logit i.
                // A mismatch silently renames every tag, so fail loudly — the engine
                // stays unloaded and tagging is disabled rather than mislabeling.
                guard self.tags.count == self.bridge.tagCount else {
                    cont.resume(returning: (false,
                        "tag list has \(self.tags.count) entries but model emits \(self.bridge.tagCount) — must match"))
                    return
                }

                self.loaded = true
                cont.resume(returning: (true, nil))
            }
        }
    }

    // MARK: - Tagging

    /// Tag one image (decoded from `imageData`). Returns the predicted tags sorted
    /// by descending score. Fails soft to `[]` if the engine isn't loaded, the image
    /// can't be decoded, or the model can't run.
    func tag(imageData: Data,
             threshold: Float = ImageTagEngine.defaultThreshold,
             topK: Int = ImageTagEngine.defaultTopK) async -> [ImageTag] {
        await withCheckedContinuation { cont in
            queue.async { [weak self] in
                guard let self, self.loaded else { cont.resume(returning: []); return }
                guard let cg = Self.decode(imageData) else {
                    NSLog("[ImageTagEngine] decode failed"); cont.resume(returning: []); return
                }
                if Self.debugCompareNorms { self.debugCompareNorms(cg) }
                guard let pixels = self.preprocess(cg, norm: Self.productionNorm) else {
                    NSLog("[ImageTagEngine] preprocess failed"); cont.resume(returning: []); return
                }
                guard let raw = self.bridge.tag(pixels: pixels.withUnsafeBytes { Data($0) }) else {
                    NSLog("[ImageTagEngine] inference failed: %@", self.bridge.lastError)
                    cont.resume(returning: []); return
                }
                cont.resume(returning: self.decode(output: raw, threshold: threshold, topK: topK))
            }
        }
    }

    // MARK: - Output decoding

    /// Convert one raw model output to per-tag probabilities. The `.tflite` may emit
    /// raw logits or (as this export does) already-sigmoid probabilities in [0,1] —
    /// detect from the value range and only sigmoid true logits. Applying sigmoid to
    /// already-probabilities double-squashes every score into ~0.62–0.73.
    private func probabilities(from raw: [NSNumber]) -> [Float] {
        let count = min(raw.count, tags.count)
        guard count > 0 else { return [] }
        var lo = Float.greatestFiniteMagnitude, hi = -Float.greatestFiniteMagnitude
        for i in 0..<count {
            let v = raw[i].floatValue
            if v < lo { lo = v }; if v > hi { hi = v }
        }
        let alreadyProb = lo >= -0.001 && hi <= 1.001
        var out = [Float](repeating: 0, count: count)
        for i in 0..<count {
            let v = raw[i].floatValue
            out[i] = alreadyProb ? v : 1 / (1 + Foundation.exp(-v))
        }
        return out
    }

    /// Threshold, sort by descending score, and cap a probability vector into tags.
    private func emitTags(from probs: [Float], threshold: Float, topK: Int) -> [ImageTag] {
        let count = min(probs.count, tags.count)
        var out: [ImageTag] = []
        out.reserveCapacity(topK)
        for i in 0..<count where probs[i] >= threshold {
            out.append(ImageTag(name: tags[i], score: probs[i]))
        }
        out.sort { $0.score > $1.score }
        if out.count > topK { out.removeLast(out.count - topK) }
        return out
    }

    /// Single-output convenience used by `debugCompareNorms`.
    private func decode(output raw: [NSNumber], threshold: Float, topK: Int) -> [ImageTag] {
        emitTags(from: probabilities(from: raw), threshold: threshold, topK: topK)
    }

    // MARK: - Preprocessing

    /// Decode image bytes into a CGImage, honoring EXIF orientation so a photo
    /// taken sideways is tagged upright.
    private static func decode(_ data: Data) -> CGImage? {
        #if canImport(UIKit)
        return decodeUIKit(data)
        #else
        // macOS: ImageIO bakes the EXIF orientation in when asked for a
        // "thumbnail" with the transform applied, at full size. Alpha survives,
        // so `preprocess` still composites transparent PNGs onto white.
        guard let src = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
        if debugIgnoreExif {
            NSLog("[ImageTagEngine] decode IGNORING EXIF — raw sensor pixels (sideways for rotated photos)")
            return CGImageSourceCreateImageAtIndex(src, 0, nil)
        }
        let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any]
        let w = props?[kCGImagePropertyPixelWidth] as? Int ?? 0
        let h = props?[kCGImagePropertyPixelHeight] as? Int ?? 0
        let opts: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: max(w, h, 1),
        ]
        return CGImageSourceCreateThumbnailAtIndex(src, 0, opts as CFDictionary)
        #endif
    }

    #if canImport(UIKit)
    private static func decodeUIKit(_ data: Data) -> CGImage? {
        guard let ui = UIImage(data: data) else { return nil }
        if debugIgnoreExif {
            // A/B only: raw sensor pixels, EXIF ignored (host --no-exif-transpose).
            // Sideways for rotated photos → the "bicycle/wall" reading. NOT production.
            NSLog("[ImageTagEngine] decode IGNORING EXIF — raw sensor pixels (sideways for rotated photos)")
            return ui.cgImage
        }
        // Redraw through UIKit so the orientation flag is baked into the pixels.
        // Keep it NON-opaque so any transparency survives to `preprocess`, which
        // composites onto white (per the model contract) — an opaque renderer would
        // put BLACK behind transparent PNGs and defeat that white composite.
        let format = UIGraphicsImageRendererFormat.default()
        format.scale = 1
        format.opaque = false
        let renderer = UIGraphicsImageRenderer(size: ui.size, format: format)
        let normalized = renderer.image { _ in ui.draw(at: .zero) }
        if debugVerbose {
            // orient rawValue: .up=0 .down=1 .left=2 .right=3 (EXIF 6/90°CW → .right)
            NSLog("[ImageTagEngine] decode uiSize=%.0fx%.0f orient=%ld → upright=%.0fx%.0f",
                  ui.size.width, ui.size.height, ui.imageOrientation.rawValue,
                  normalized.size.width, normalized.size.height)
        }
        return normalized.cgImage
    }
    #endif

    /// Run one image through every `Norm` variant and log the top tags for each, so
    /// a baked-in-normalization mismatch is obvious: the correct scaling gives sharp,
    /// image-specific tags; the wrong ones give vague, low-confidence ones.
    private func debugCompareNorms(_ cg: CGImage) {
        for norm in [Norm.imagenet, .unit, .signed] {
            guard let px = preprocess(cg, norm: norm),
                  let raw = bridge.tag(pixels: px.withUnsafeBytes { Data($0) }) else { continue }
            let top = decode(output: raw, threshold: 0, topK: 6)
            let desc = top.map { "\($0.name):\(String(format: "%.2f", $0.score))" }.joined(separator: ", ")
            NSLog("[ImageTagEngine] NORM %@ → %@", "\(norm)", desc)
        }
    }

    /// Reproduce the eval transform (short-side resize 256 → center-crop 224 →
    /// scale to [0,1] → normalize) and lay the result out to match the model's input
    /// tensor. RGBA is flattened on white per the release README. `norm` selects the
    /// value scaling (see `Norm`).
    private func preprocess(_ cg: CGImage, norm: Norm) -> [Float]? {
        let w = cg.width, h = cg.height
        guard w > 0, h > 0 else { return nil }

        // Match the host script's torchvision transform exactly:
        //   T.Resize(256)     → scale the SHORTER side to 256, longer follows (int-truncated)
        //   T.CenterCrop(224)
        // Fuse resize+crop into ONE resample by drawing the scaled image centered in
        // the 224×224 output; whatever falls outside is the center crop. (`cg` is
        // already EXIF-upright — see `decode`.)
        let target = Self.resizeShortSide            // 256
        let scale = Double(target) / Double(min(w, h))
        let scaledW = (w <= h) ? target : Int(Double(w) * scale)
        let scaledH = (w <= h) ? Int(Double(h) * scale) : target
        // Center-crop offsets, matching torchvision CenterCrop's rounding.
        let originX = -Int((Double(scaledW - side) / 2).rounded())
        let originY = -Int((Double(scaledH - side) / 2).rounded())

        // Draw into a side×side RGBA8 bitmap (resize + crop happen in this one pass).
        let bytesPerRow = side * 4
        var rgba = [UInt8](repeating: 0, count: side * side * 4)
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
        guard let ctx = rgba.withUnsafeMutableBytes({ raw -> CGContext? in
            CGContext(data: raw.baseAddress, width: side, height: side,
                      bitsPerComponent: 8, bytesPerRow: bytesPerRow,
                      space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: bitmapInfo)
        }) else { return nil }
        ctx.interpolationQuality = .high     // closest to PIL's antialiased bilinear
        // Flip to a top-left origin so row 0 is the top of the image (matches PIL).
        ctx.translateBy(x: 0, y: CGFloat(side))
        ctx.scaleBy(x: 1, y: -1)
        // Flatten any transparency onto white first, then draw the scaled image over it.
        ctx.setFillColor(CGColor(gray: 1, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: side, height: side))
        ctx.draw(cg, in: CGRect(x: originX, y: originY, width: scaledW, height: scaledH))

        // Normalize into the model's expected channel layout.
        let plane = side * side
        var out = [Float](repeating: 0, count: plane * 3)
        for r in 0..<side {
            for c in 0..<side {
                let o = (r * side + c) * 4
                let rgb: [Float] = [Float(rgba[o]), Float(rgba[o + 1]), Float(rgba[o + 2])]
                for ch in 0..<3 {
                    let v: Float
                    switch norm {
                    case .imagenet: v = (rgb[ch] / 255 - Self.mean[ch]) / Self.std[ch]
                    case .unit:     v = rgb[ch] / 255
                    case .signed:   v = rgb[ch] / 127.5 - 1
                    }
                    switch layout {
                    case .nhwc: out[(r * side + c) * 3 + ch] = v
                    case .nchw: out[ch * plane + r * side + c] = v
                    }
                }
            }
        }

        if Self.debugVerbose {
            // Per-channel means + range let us diff the app's input tensor against the
            // host's `x`: in Python add `print(x.mean((0,2,3)), x.min(), x.max())`.
            // If R̄/Ḡ/B̄ match, preprocessing is equivalent and any output gap is the
            // resampler/model, not our pipeline.
            func planeMean(_ ch: Int) -> Float {
                var s: Float = 0
                for i in 0..<plane {
                    s += (layout == .nchw) ? out[ch * plane + i] : out[i * 3 + ch]
                }
                return s / Float(plane)
            }
            // Center-pixel RGB (post-scale) — a concrete value to eyeball against host.
            let cIdx = (side / 2) * side + (side / 2)
            let center: [Float] = layout == .nchw
                ? [out[cIdx], out[plane + cIdx], out[2 * plane + cIdx]]
                : [out[cIdx * 3], out[cIdx * 3 + 1], out[cIdx * 3 + 2]]
            NSLog("[ImageTagEngine] preprocess upright=%dx%d scaled=%dx%d origin=(%d,%d) norm=%@",
                  w, h, scaledW, scaledH, originX, originY, "\(norm)")
            NSLog("[ImageTagEngine] tensor R̄=%.4f Ḡ=%.4f B̄=%.4f min=%.4f max=%.4f center=[%.4f,%.4f,%.4f]",
                  planeMean(0), planeMean(1), planeMean(2),
                  out.min() ?? 0, out.max() ?? 0, center[0], center[1], center[2])
        }
        return out
    }
}
