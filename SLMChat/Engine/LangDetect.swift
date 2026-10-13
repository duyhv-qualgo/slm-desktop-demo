import Foundation

/// On-device language identification: MediaPipe's Language Detector, run without
/// MediaPipe.
///
/// Replaces `NLLanguageRecognizer` as the engine behind `Prompts.identify`. The
/// reason is not raw accuracy — Apple's model is larger and separates close
/// neighbours better on two-word fragments — it is that **this one abstains
/// where Apple's cascade asserted**. The old stage 2 (`vietnameseRatio`) could
/// only answer Vietnamese or English, so every Latin-script language the
/// recognizer declined to name was claimed as Vietnamese: measured on 44 chat
/// messages across 17 unserved Latin languages, Portuguese, Romanian, Czech,
/// Catalan and Indonesian were all routed to the VIETNAMESE guardrail pack. A
/// classifier that never saw the language still returns a confident verdict, so
/// that failure is silent. This model refuses all 44.
///
/// It is also ~20x faster (0.029 ms vs 0.56 ms per message, measured on the same
/// corpus), which matters because every screened message pays for it.
///
/// ## Architecture
///
/// PRADO/SGNN, 111 BCP-47 labels, no word vocabulary at all:
///
///     text → Plan9 lowercase → tokenize → MurmurHash64 over character n-grams
///          (n = 1,2,3,4; vocab 5500/5500/13000/13000)
///          → k-means-compressed embedding lookup → mean-pool
///          → FC[96→160] ReLU → FC[160→200] ReLU → FC[200→111] → softmax
///
/// 70,031 parameters in the MLP. The hashing cannot be a tensor op — it is
/// string processing over UTF-8 — which is why MediaPipe ships it as a custom
/// op and why it lives here in Swift rather than in a graph.
///
/// ## Fidelity
///
/// Ported 1:1 from the MediaPipe extraction's `langdet/model.py`, which that
/// release verifies bit-exact against LiteRT. That source is NOT in this repo —
/// it is gitignored under `language-detection/` and ships via `$SLMCHAT_ASSETS`,
/// because nothing at build or run time reads it. `LangDetectConformanceTests`
/// pins this port against reference vectors captured from it, and is the only
/// thing standing between a plausible-looking refactor and silent misrouting.
/// Three details are load-bearing and each fails quietly if changed:
///
///  1. The three dense layers must run TFLite's **hybrid / dynamic-range** int8
///     kernel. A plain float matmul drifts up to 1.3e-02, worst on short text.
///  2. Lowercasing uses the **Plan9** tables in `LangDetectTables`, not
///     `String.lowercased()`. They predate modern Unicode and disagree on a
///     number of codepoints; the model was trained against these.
///  3. Iteration is over **unicode scalars**, not Swift `Character`s. A grapheme
///     cluster lowercases as a unit and diverges from the C loop.
final class LangDetect {

    /// Shared instance. Loading is lazy and happens once; a failure leaves
    /// `isLoaded` false and every lookup returns no hypotheses, which
    /// `Prompts.identify` reads as "can't tell" and resolves to the thread's
    /// language. A missing asset must never be able to crash the chat.
    static let shared = LangDetect()

    private let model: Model?

    var isLoaded: Bool { model != nil }

    private init() {
        guard let binURL = ResourceLookup.url("langdetect-weights", ext: "bin"),
              let jsonURL = ResourceLookup.url("langdetect-weights", ext: "json"),
              let labelsURL = ResourceLookup.url("langdetect-labels", ext: "txt") else {
            NSLog("[LangDetect] assets not in bundle — add langdetect-weights.{bin,json} "
                + "and langdetect-labels.txt to Resources/ (scripts/bootstrap.sh)")
            model = nil
            return
        }
        do {
            model = try Model(bin: binURL, json: jsonURL, labels: labelsURL)
            NSLog("[LangDetect] loaded (%d labels)", model?.labels.count ?? 0)
        } catch {
            NSLog("[LangDetect] load failed: %@", String(describing: error))
            model = nil
        }
    }

    /// Language hypotheses for `text`, most likely first, as BCP-47 codes paired
    /// with softmax probabilities. Empty when the model isn't loaded.
    ///
    /// Deliberately returns the FULL ranked list over all 111 labels rather than
    /// a single answer, and deliberately is not filtered to the languages the app
    /// serves. Narrowing here would renormalise probability mass over the allowed
    /// set, which is exactly how a language we have no pack for gets forced onto
    /// the nearest one we do. `Prompts.identify` walks this list and skips what it
    /// cannot use; `unknown` is a real label in it, not an error.
    ///
    /// Thread-safe: every call allocates its own scratch. Measured at 2% over a
    /// shared buffer, which is not worth a lock on a path reached from both the
    /// main actor and `ChatService`'s background work.
    func ranked(_ text: String) -> [(code: String, probability: Double)] {
        guard let model else { return [] }
        let probs = model.probabilities(text)
        return zip(model.labels, probs)
            .map { (code: $0.0, probability: Double($0.1)) }
            .sorted { $0.probability > $1.probability }
    }

    /// The raw n-gram id matrix, row-major `[4][tokens]`. Conformance seam —
    /// an exact-integer assertion on this catches every preprocessing drift,
    /// where a probability comparison only catches the large ones.
    func ngramIDs(_ text: String) -> (ids: [Int32], tokens: Int) {
        guard let model else { return ([], 0) }
        let scratch = Scratch()
        let n = model.hash(text, into: scratch)
        return (Array(scratch.ids[0..<(4 * n)]), n)
    }
}

// MARK: - Plan9 rune classification

extension LangDetect {

    /// Port of `utf_isalpharune`.
    static func isAlphaRune(_ c: Int32) -> Bool {
        let r = LangDetectTables.alphaRanges
        var lo = 0, hi = r.count / 2 - 1
        while lo <= hi {
            let mid = (lo + hi) / 2
            if c < r[mid * 2] { hi = mid - 1 }
            else if c > r[mid * 2 + 1] { lo = mid + 1 }
            else { return true }
        }
        let s = LangDetectTables.alphaSingles
        lo = 0; hi = s.count - 1
        while lo <= hi {
            let mid = (lo + hi) / 2
            if c < s[mid] { hi = mid - 1 }
            else if c > s[mid] { lo = mid + 1 }
            else { return true }
        }
        return false
    }

    /// Port of `utf_tolowerrune`.
    static func toLowerRune(_ c: Int32) -> Int32 {
        let bias = LangDetectTables.bias
        let r = LangDetectTables.lowerRanges
        var lo = 0, hi = r.count / 3 - 1
        while lo <= hi {
            let mid = (lo + hi) / 2
            if c < r[mid * 3] { hi = mid - 1 }
            else if c > r[mid * 3 + 1] { lo = mid + 1 }
            else { return c + r[mid * 3 + 2] - bias }
        }
        // In these ranges only alternating codepoints are uppercase.
        let p = LangDetectTables.lowerPairs
        lo = 0; hi = p.count / 3 - 1
        while lo <= hi {
            let mid = (lo + hi) / 2
            if c < p[mid * 3] { hi = mid - 1 }
            else if c > p[mid * 3 + 1] { lo = mid + 1 }
            else { return (c - p[mid * 3]) & 1 == 0 ? c + p[mid * 3 + 2] - bias : c }
        }
        let g = LangDetectTables.lowerSingles
        lo = 0; hi = g.count / 2 - 1
        while lo <= hi {
            let mid = (lo + hi) / 2
            if c < g[mid * 2] { hi = mid - 1 }
            else if c > g[mid * 2] { lo = mid + 1 }
            else { return c + g[mid * 2 + 1] - bias }
        }
        return c
    }
}

// MARK: - MurmurHash64WithSeed

extension LangDetect {

    private static let kMul: UInt64 = 0xC6A4A793_5BD1E995

    @inline(__always)
    private static func shiftMix(_ v: UInt64) -> UInt64 { v ^ (v >> 47) }

    /// Murmur 2.0, 64-bit, little-endian. Wrapping arithmetic throughout — the
    /// C original relies on unsigned overflow.
    static func murmurHash64(_ buf: UnsafePointer<UInt8>, _ n: Int, seed: UInt64) -> UInt64 {
        var h = seed ^ (UInt64(n) &* kMul)
        let aligned = n & ~0x7
        var i = 0
        while i < aligned {
            var word: UInt64 = 0
            withUnsafeMutableBytes(of: &word) {
                $0.copyMemory(from: UnsafeRawBufferPointer(start: buf + i, count: 8))
            }
            h ^= shiftMix(UInt64(littleEndian: word) &* kMul) &* kMul
            h = h &* kMul
            i += 8
        }
        let rem = n & 0x7
        if rem != 0 {
            var tail: UInt64 = 0
            for k in 0..<rem { tail |= UInt64(buf[aligned + k]) << (8 * UInt64(k)) }
            h ^= tail
            h = h &* kMul
        }
        return shiftMix(shiftMix(h) &* kMul)
    }
}

// MARK: - UTF-8 decoding (utf_charntorune)

extension LangDetect {

    static let runeError: Int32 = 0xFFFD
    private static let runeMax: Int32 = 0x10FFFF

    /// Returns `(rune, bytesConsumed)`.
    ///
    /// The two error cases are NOT interchangeable and the distinction is the
    /// whole reason this is hand-ported. A malformed sequence consumes 1 byte
    /// and the scan continues; a well-formed but TRUNCATED one — the buffer ends
    /// mid-character — consumes 0, which stops tokenization. MediaPipe reads past
    /// its buffer when lowercasing shortens the text, and lands in that second
    /// case; reproducing it is what keeps this port faithful on `İ` and `ẞ`.
    @inline(__always)
    static func decodeRune(_ d: UnsafePointer<UInt8>, _ i: Int, _ length: Int) -> (Int32, Int) {
        if length <= 0 { return (runeError, 0) }
        let c = Int32(d[i])
        if c < 0x80 { return (c, 1) }
        if length <= 1 { return (runeError, 0) }
        let c1 = Int32(d[i + 1]) ^ 0x80
        if c1 & 0xC0 != 0 { return (runeError, 1) }
        if c < 0xE0 {
            if c < 0xC0 { return (runeError, 1) }
            let l = ((c << 6) | c1) & 0x7FF
            return l <= 0x7F ? (runeError, 1) : (l, 2)
        }
        if length <= 2 { return (runeError, 0) }
        let c2 = Int32(d[i + 2]) ^ 0x80
        if c2 & 0xC0 != 0 { return (runeError, 1) }
        if c < 0xF0 {
            let l = ((((c << 6) | c1) << 6) | c2) & 0xFFFF
            return l <= 0x7FF ? (runeError, 1) : (l, 3)
        }
        if length <= 3 { return (runeError, 0) }
        let c3 = Int32(d[i + 3]) ^ 0x80
        if c3 & 0xC0 != 0 { return (runeError, 1) }
        if c < 0xF8 {
            let l = ((((((c << 6) | c1) << 6) | c2) << 6) | c3) & 0x1FFFFF
            return (l <= 0xFFFF || l > runeMax) ? (runeError, 1) : (l, 4)
        }
        return (runeError, 1)
    }
}

// MARK: - Per-call scratch

extension LangDetect {

    /// Working buffers for one `probabilities` call. Private to the call, so the
    /// detector holds no mutable state and needs no synchronisation.
    final class Scratch {
        var lower: [UInt8] = []
        var buf = [UInt8](repeating: 0, count: 640)
        var tokOff = [Int](repeating: 0, count: 130)
        var tokLen = [Int](repeating: 0, count: 130)
        var ids = [Int32](repeating: 0, count: 4 * 130)
        init() { lower.reserveCapacity(256) }
    }
}

// MARK: - The model

extension LangDetect {

    enum LoadError: Error {
        case malformedMetadata(String)
        case truncatedBlob(tensor: String)
    }

    final class Model {
        let seed: UInt64
        let ngramLengths: [Int]
        let vocabSizes: [Int]
        let maxSplits: Int
        let labels: [String]

        private let cMatrix: [[UInt8]]     // [head] codebook indices, [vocab, 6]
        private let bMatrix: [[Float]]     // [head] codebook,         [256, 4]
        private let fcWeight: [[Int8]]
        private let fcBias: [[Float]]
        private let fcScale: [Double]
        private let fcOut: [Int]
        private let fcIn: [Int]

        init(bin: URL, json: URL, labels labelsURL: URL) throws {
            let blob = try Data(contentsOf: bin)
            guard let meta = try JSONSerialization.jsonObject(with: Data(contentsOf: json))
                    as? [String: Any],
                  let tensors = meta["tensors"] as? [String: [String: Any]],
                  let seedValue = meta["seed"] as? Int,
                  let lengths = meta["ngram_lengths"] as? [Int],
                  let vocabs = meta["vocab_sizes"] as? [Int],
                  let splits = meta["max_splits"] as? Int,
                  let scales = meta["fc_scales"] as? [Double] else {
                throw LoadError.malformedMetadata(json.lastPathComponent)
            }
            seed = UInt64(seedValue)
            ngramLengths = lengths
            vocabSizes = vocabs
            maxSplits = splits
            fcScale = scales

            func extent(_ name: String, _ stride: Int) throws -> (Int, Int) {
                guard let t = tensors[name],
                      let off = t["off"] as? Int, let count = t["count"] as? Int else {
                    throw LoadError.malformedMetadata(name)
                }
                guard off + count * stride <= blob.count else {
                    throw LoadError.truncatedBlob(tensor: name)
                }
                return (off, count)
            }
            func floats(_ name: String) throws -> [Float] {
                let (off, count) = try extent(name, MemoryLayout<Float>.size)
                return blob.withUnsafeBytes { raw in
                    Array(UnsafeBufferPointer(
                        start: raw.baseAddress!.advanced(by: off).assumingMemoryBound(to: Float.self),
                        count: count))
                }
            }
            func bytes(_ name: String) throws -> [UInt8] {
                let (off, count) = try extent(name, 1)
                return Array(blob[off..<(off + count)])
            }
            func shape(_ name: String) throws -> [Int] {
                guard let s = tensors[name]?["shape"] as? [Int] else {
                    throw LoadError.malformedMetadata("\(name).shape")
                }
                return s
            }

            cMatrix = try (0..<4).map { try bytes("c\($0)") }
            bMatrix = try (0..<4).map { try floats("b\($0)") }
            fcWeight = try (0..<3).map { try bytes("fc\($0)_w").map { Int8(bitPattern: $0) } }
            fcBias = try (0..<3).map { try floats("fc\($0)_b") }
            let shapes = try (0..<3).map { try shape("fc\($0)_w") }
            fcOut = shapes.map { $0[0] }
            fcIn = shapes.map { $0[1] }
            labels = try String(contentsOf: labelsURL, encoding: .utf8)
                .split(separator: "\n")
                .map { $0.trimmingCharacters(in: .whitespaces) }
                .filter { !$0.isEmpty }
        }

        // MARK: NGramHash

        /// `LowercaseUnicodeStr` + `Tokenize` + MurmurHash, writing ids into
        /// `scratch.ids` as `[4][tokens]` row-major. Returns the token count.
        @discardableResult
        func hash(_ text: String, into s: Scratch) -> Int {
            // Lowercase per Plan9, scalar by scalar. Unicode SCALARS, not Swift
            // Characters — see the type comment.
            s.lower.removeAll(keepingCapacity: true)
            var rawLen = 0
            for u in text.unicodeScalars {
                let v = u.value
                rawLen += v < 0x80 ? 1 : (v < 0x800 ? 2 : (v < 0x10000 ? 3 : 4))
                let c = Int32(bitPattern: v)
                let mapped = LangDetect.isAlphaRune(c) ? LangDetect.toLowerRune(c) : c
                UTF8.encode(Unicode.Scalar(UInt32(bitPattern: mapped)) ?? u) { s.lower.append($0) }
            }
            // MediaPipe hands Tokenize the ORIGINAL byte length together with the
            // LOWERCASED buffer. When lowercasing shortens the text (İ→i, ẞ→ß) the
            // C loop reads past the end into std::string's zeroed slack. Padding
            // with NUL reproduces the cold-call behaviour; those bytes decode to
            // non-alpha runes and become " " tokens. Dropping this pad silently
            // changes the answer for Turkish and German text.
            if rawLen > s.lower.count {
                s.lower.append(contentsOf: repeatElement(0, count: rawLen - s.lower.count))
            }
            if s.buf.count < rawLen + 2 {
                s.buf = [UInt8](repeating: 0, count: (rawLen + 2) * 2)
            }

            var nTok = 0, bufLen = 0, start = 1, i = 0
            s.buf[0] = UInt8(ascii: "^"); bufLen = 1
            s.tokOff[0] = 0; s.tokLen[0] = 1; nTok = 1

            s.lower.withUnsafeBufferPointer { lp in
                guard let d = lp.baseAddress else { return }
                while i < rawLen {
                    if nTok + 1 >= maxSplits { break }        // leave room for "$"
                    let (rune, nbytes) = LangDetect.decodeRune(d, i, rawLen - i)
                    if nbytes == 0 { break }
                    if LangDetect.isAlphaRune(rune) {
                        for k in 0..<nbytes { s.buf[bufLen + k] = d[i + k] }
                        bufLen += nbytes
                        s.tokOff[nTok] = start; s.tokLen[nTok] = nbytes
                        start += nbytes
                    } else {
                        s.buf[bufLen] = UInt8(ascii: " "); bufLen += 1
                        s.tokOff[nTok] = start; s.tokLen[nTok] = 1
                        start += 1
                    }
                    nTok += 1
                    i += nbytes
                }
            }
            s.buf[bufLen] = UInt8(ascii: "$"); bufLen += 1
            s.tokOff[nTok] = start; s.tokLen[nTok] = 1; nTok += 1

            if s.ids.count < 4 * nTok { s.ids = [Int32](repeating: 0, count: 4 * nTok) }
            s.buf.withUnsafeBufferPointer { bp in
                guard let b = bp.baseAddress else { return }
                for gi in 0..<ngramLengths.count {
                    let glen = ngramLengths[gi], vocab = UInt64(vocabSizes[gi])
                    for t in 0..<nTok {
                        var nbytes = 0
                        for k in t..<min(nTok, t + glen) { nbytes += s.tokLen[k] }
                        let h = LangDetect.murmurHash64(b + s.tokOff[t], nbytes, seed: seed)
                        s.ids[gi * nTok + t] = Int32(h % vocab) + 1
                    }
                }
            }
            return nTok
        }

        // MARK: Graph

        /// TFLite's hybrid FullyConnected with `asymmetric_quantize_inputs=true`.
        ///
        /// Reproduces `tensor_utils::AsymmetricQuantizeFloats`: quantize the
        /// activation row to int8 against a nudged zero point, multiply in the
        /// integer domain, scale back. A plain float matmul is NOT equivalent —
        /// it drifts up to 1.3e-02 and does so worst on short text, which is
        /// precisely the regime chat messages live in.
        private func hybridFC(_ x: [Float], _ layer: Int, into out: inout [Float]) {
            let w = fcWeight[layer], bias = fcBias[layer]
            let nOut = fcOut[layer], nIn = fcIn[layer]
            var rmin = 0.0, rmax = 0.0
            for v in x { rmin = min(rmin, Double(v)); rmax = max(rmax, Double(v)) }
            if rmin == rmax {
                for o in 0..<nOut { out[o] = bias[o] }
                return
            }
            let scale = (rmax - rmin) / 255.0
            let zpFromMin = -128.0 - rmin / scale
            let zeroPoint: Int32 = zpFromMin < -128.0 ? -128
                                 : (zpFromMin > 127.0 ? 127
                                 : Int32(zpFromMin.rounded(.toNearestOrEven)))
            let inv = 1.0 / scale
            var q = [Int32](repeating: 0, count: nIn)
            for k in 0..<nIn {
                let r = (Double(x[k]) * inv).rounded(.toNearestOrEven) + Double(zeroPoint)
                q[k] = Int32(max(-128.0, min(127.0, r)))
            }
            let combined = Float(scale * fcScale[layer])
            w.withUnsafeBufferPointer { wp in
                q.withUnsafeBufferPointer { qp in
                    for o in 0..<nOut {
                        var acc: Int32 = 0, rowSum: Int32 = 0
                        let base = o * nIn
                        for k in 0..<nIn {
                            let weight = Int32(wp[base + k])
                            acc &+= weight &* qp[k]
                            rowSum &+= weight
                        }
                        out[o] = Float(acc - zeroPoint * rowSum) * combined + bias[o]
                    }
                }
            }
        }

        func probabilities(_ text: String) -> [Float] {
            let s = Scratch()
            let nTok = hash(text, into: s)

            // KmeansEmbeddingLookup x4 -> concat. Each head decompresses 6 codebook
            // indices into 24 floats and means them over the tokens, stopping at
            // the first 0 id (the padding marker).
            var x = [Float](repeating: 0, count: 96)
            for head in 0..<4 {
                let codes = cMatrix[head], book = bMatrix[head]
                var acc = [Float](repeating: 0, count: 24)
                var n = 0
                for t in 0..<nTok {
                    let id = s.ids[head * nTok + t]
                    if id == 0 { break }
                    n += 1
                    let base = Int(id) * 6
                    for e in 0..<6 {
                        let code = Int(codes[base + e]) * 4
                        for j in 0..<4 { acc[e * 4 + j] += book[code + j] }
                    }
                }
                let divisor = Float(max(n, 1))
                for e in 0..<24 { x[head * 24 + e] = acc[e] / divisor }
            }

            var h1 = [Float](repeating: 0, count: fcOut[0])
            hybridFC(x, 0, into: &h1)
            for i in 0..<h1.count { h1[i] = max(h1[i], 0) }          // ReLU, fused in TFLite
            var h2 = [Float](repeating: 0, count: fcOut[1])
            hybridFC(h1, 1, into: &h2)
            for i in 0..<h2.count { h2[i] = max(h2[i], 0) }
            var logits = [Float](repeating: 0, count: fcOut[2])
            hybridFC(h2, 2, into: &logits)

            let peak = logits.max() ?? 0
            var total: Float = 0
            for i in 0..<logits.count { logits[i] = expf(logits[i] - peak); total += logits[i] }
            for i in 0..<logits.count { logits[i] /= total }
            return logits
        }
    }
}
