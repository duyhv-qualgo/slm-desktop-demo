import Foundation

/// Byte-level Byte-Pair-Encoding tokenizer for the text-guardrail model, ported
/// 1:1 from the pack's Hugging Face `tokenizer.json` pipeline:
///
///   1. Normalize: NFKC. **No lowercasing** — the v16 packs are case-sensitive.
///   2. Pre-tokenize: the GPT-2 split regex, which keeps a leading space attached
///      to the word that follows it.
///   3. Byte-level map: UTF-8 bytes → a 256-character printable alphabet, so a
///      space becomes `Ġ` and every byte sequence is representable.
///   4. BPE: rank-based merge per pre-token over those mapped characters.
///   5. Truncate to `maxLen`, pad with `[PAD]`=0, build the attention mask.
///
/// **This replaced a completely different pipeline** and the difference was not
/// cosmetic. The pre-v16 packs normalized with NFKC **+ Lowercase** and
/// pre-tokenized on `\w+|[^\w\s]+` (a `Whitespace` pre-tokenizer), so the vocab
/// was keyed by bare lowercase word pieces. v16 is `ByteLevel`, case-preserving,
/// and its vocab is keyed by byte-mapped strings. Running the old algorithm over
/// the new vocab does not fail — it produces ids that mostly miss, and the model
/// returns a confident verdict about text it never saw. Measured while porting:
/// "Tao sẽ giết mày" scored p_unsafe 0.0004 under the old tokenizer against
/// 0.979 from the reference implementation. Nothing crashed.
///
/// Validated against the real HF tokenizer on the fixtures in
/// `GuardrailTests/tokenizer_fixtures.json` (regenerate with
/// `scripts/build_guard_fixtures.py`). Do not "optimize" the byte mapping or the
/// merge order without re-running that parity check.
struct GuardTokenizer {

    private let vocab: [String: Int32]
    /// Merge rank keyed by "left\u{1}right"; lower rank = higher merge priority.
    private let ranks: [String: Int]
    private let unkId: Int32
    let maxLen: Int

    /// The GPT-2 pre-tokenizer pattern, used when `pre_tokenizer.use_regex` is
    /// true (it is, for every v16 pack).
    ///
    /// The leading ` ?` in the letter/number/symbol branches is the part that
    /// matters: a space is captured WITH the following word rather than dropped,
    /// which is what makes `Ġ`-prefixed vocab entries reachable. Splitting on
    /// whitespace first — as the old pipeline did — makes every such entry dead
    /// and silently degrades everything to single-character pieces.
    private static let pretokRegex = try! NSRegularExpression(
        pattern: #"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#)

    /// byte value → the character that stands in for it.
    ///
    /// GPT-2's trick for making arbitrary bytes printable: the 188 bytes that are
    /// already printable ASCII/Latin-1 map to themselves, and the remaining 68
    /// (control codes, space, delete, and the Latin-1 gaps) are displaced into
    /// U+0100 upward. So 0x20 (space) becomes U+0120 `Ġ`, and no byte maps to a
    /// character that could be confused with a real one.
    private static let byteEncoder: [UInt8: Character] = {
        var bs: [Int] = []
        bs.append(contentsOf: Int(UInt8(ascii: "!"))...Int(UInt8(ascii: "~")))
        bs.append(contentsOf: 0xA1...0xAC)
        bs.append(contentsOf: 0xAE...0xFF)

        var map = [UInt8: Character](minimumCapacity: 256)
        for b in bs { map[UInt8(b)] = Character(UnicodeScalar(UInt8(b))) }

        // Everything not already covered gets the next free slot from U+0100.
        var next = 0
        for b in 0...255 where !bs.contains(b) {
            map[UInt8(b)] = Character(UnicodeScalar(256 + next)!)
            next += 1
        }
        return map
    }()

    /// Load vocab + merges from a Hugging Face `tokenizer.json`. Returns nil if
    /// the file is missing or malformed.
    init?(tokenizerJSONURL: URL, maxLen: Int = 128) {
        guard let data = try? Data(contentsOf: tokenizerJSONURL),
              let root = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
              let model = root["model"] as? [String: Any],
              let vocabRaw = model["vocab"] as? [String: Any],
              let mergesRaw = model["merges"] as? [Any] else {
            return nil
        }

        var vocab = [String: Int32](minimumCapacity: vocabRaw.count)
        for (tok, v) in vocabRaw {
            if let n = v as? NSNumber { vocab[tok] = n.int32Value }
        }
        self.vocab = vocab
        self.unkId = vocab["[UNK]"] ?? 1
        self.maxLen = maxLen

        var ranks = [String: Int](minimumCapacity: mergesRaw.count)
        for (i, m) in mergesRaw.enumerated() {
            // merges entries are ["left","right"] pairs (some exports use "left right").
            if let pair = m as? [String], pair.count == 2 {
                ranks[Self.key(pair[0], pair[1])] = i
            } else if let s = m as? String {
                let parts = s.split(separator: " ", maxSplits: 1).map(String.init)
                if parts.count == 2 { ranks[Self.key(parts[0], parts[1])] = i }
            }
        }
        self.ranks = ranks
    }

    private static func key(_ a: String, _ b: String) -> String { a + "\u{1}" + b }

    /// Encode text into fixed-length `input_ids` + `attention_mask` (both Int32).
    func encode(_ text: String) -> (ids: [Int32], mask: [Int32]) {
        var ids: [Int32] = []
        for token in preTokenize(Self.normalize(text)) {
            ids.append(contentsOf: bpe(token))
            if ids.count >= maxLen { break }
        }
        if ids.count > maxLen { ids = Array(ids.prefix(maxLen)) }

        let real = ids.count
        if real < maxLen { ids.append(contentsOf: repeatElement(0, count: maxLen - real)) }
        var mask = [Int32](repeating: 0, count: maxLen)
        for i in 0..<real { mask[i] = 1 }
        return (ids, mask)
    }

    /// How many tokens `text` produces, WITHOUT the `maxLen` cap.
    ///
    /// Needed to split a long message into windows the model can actually see:
    /// `encode` truncates, so it can only ever answer "128 or fewer" and can't
    /// say how much was thrown away.
    ///
    /// **Super-additive across whitespace, where the old pipeline was exactly
    /// additive.** A byte-level pre-tokenizer binds the separating space to the
    /// following word, so `"a b"` tokenizes as `"a"` + `"Ġb"` — and `"Ġb"` is one
    /// token where the segment `"b"` measured alone is also one. Summing segment
    /// counts therefore OVER-estimates a joined string, never under. Callers pack
    /// windows by summing, so the error is in the safe direction: slightly
    /// smaller windows, never an overflow past 128.
    func tokenCount(of text: String) -> Int {
        var n = 0
        for token in preTokenize(Self.normalize(text)) { n += bpe(token).count }
        return n
    }

    /// NFKC only. The `Lowercase` step the previous packs applied is deliberately
    /// absent — v16's `normalizer` is a bare `{"type": "NFKC"}` and its vocab
    /// holds cased entries, so folding case here would make every capitalized
    /// word miss its vocab entry and fall back to character pieces.
    private static func normalize(_ s: String) -> String {
        s.precomposedStringWithCompatibilityMapping
    }

    private func preTokenize(_ s: String) -> [String] {
        let ns = s as NSString
        let matches = Self.pretokRegex.matches(in: s, range: NSRange(location: 0, length: ns.length))
        return matches.map { ns.substring(with: $0.range) }
    }

    /// UTF-8 bytes → the printable stand-in alphabet. Operates on BYTES, not
    /// characters: that is the whole point of a byte-level tokenizer, and it is
    /// why a Vietnamese or Cyrillic string never needs an `[UNK]`.
    private func byteMap(_ token: String) -> [String] {
        token.utf8.map { String(Self.byteEncoder[$0] ?? "?") }
    }

    /// Classic BPE over the byte-mapped characters. Repeatedly merges the
    /// highest-priority (lowest-rank) adjacent pair present in the merge table.
    private func bpe(_ token: String) -> [Int32] {
        var word = byteMap(token)
        if word.isEmpty { return [] }

        while word.count >= 2 {
            var bestRank = Int.max
            var bestIdx = -1
            for i in 0..<(word.count - 1) {
                if let r = ranks[Self.key(word[i], word[i + 1])], r < bestRank {
                    bestRank = r
                    bestIdx = i
                }
            }
            if bestIdx == -1 { break }

            let a = word[bestIdx]
            let b = word[bestIdx + 1]
            var merged: [String] = []
            merged.reserveCapacity(word.count)
            var i = 0
            while i < word.count {
                if i < word.count - 1 && word[i] == a && word[i + 1] == b {
                    merged.append(a + b)
                    i += 2
                } else {
                    merged.append(word[i])
                    i += 1
                }
            }
            word = merged
        }

        return word.map { vocab[$0] ?? unkId }
    }
}
