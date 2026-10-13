import Foundation

/// Rule-based half of the text guardrail: a fixed list of regular expressions
/// that mark a message unsafe outright, ahead of the BERT-Tiny classifier.
///
/// Ported from the guardrail release's `infer.py`, which runs the same patterns
/// as "Step 1 (fast filter)" before the model. Two things make the rules worth
/// keeping alongside a trained classifier rather than folding into it:
///
///  - **They are certain.** The model is probabilistic and gates on a 0.5
///    confidence threshold, so a known slur can slip through at 0.49. A pattern
///    either matches or it doesn't.
///  - **They see the whole message.** `GuardTokenizer` truncates at 128 tokens,
///    so the model is blind to anything past that. A regex scans the full text
///    (up to `scanLimit`), which is where a long message hides its tail.
///
/// Rules can only ever RAISE a verdict — a pattern firing means unsafe, and no
/// pattern can mark something clean. So running them first is safe: the model
/// still decides everything they don't catch.
///
/// Immutable after `init`, and `NSRegularExpression` is thread-safe for
/// matching, so instances are safe to share.
final class GuardRules {

    /// One pattern that fired, and what it means.
    struct Hit {
        let label: SafetyLabel
        /// How serious this specific pattern is, not its category's ceiling.
        let severity: Severity
        /// The pattern's own text — logged so a false positive can be traced to
        /// the exact rule and fixed in `guardrails/regions/<region>.yaml`.
        let pattern: String
    }

    private struct Document: Decodable {
        struct Pattern: Decodable {
            let re: String
            let severity: Int
        }
        struct Category: Decodable {
            let category: String
            let patterns: [Pattern]
        }
        let categories: [Category]
        /// False when the region contributed nothing of its own and is running on
        /// the universal baseline alone. Optional-decoded so an older file still
        /// loads; absent is treated as authored, since only the new per-region
        /// generator emits the flag.
        let authored: Bool?
    }

    /// Whether this region's rule set has actually been written, as opposed to
    /// inheriting only the universal patterns every region gets. Reported in
    /// Settings so a placeholder region isn't mistaken for real coverage.
    private(set) var isAuthored = true

    /// Rule-file category names, mapped onto the app's taxonomy.
    ///
    /// Must stay in step with `ORDER` in `scripts/build_regex_patterns.py`. A
    /// category present in the JSON but absent here is LOGGED AND SKIPPED, not
    /// rejected — so a typo costs a whole category of coverage silently, which
    /// is why `GuardRulesTests.testEveryBuiltCategoryIsMapped` exists.
    ///
    /// The upstream Vietnamese release predates the 10-class model and only
    /// populates four of these; the UK set added the other five.
    private static let labels: [String: SafetyLabel] = [
        "violence": .violence,
        "sexual": .sexual,
        "political": .political,
        "child": .child,
        "self_harm": .suicide,
        "hate": .hate,
        "harassment": .harassment,
        "illegal_goods": .illegal,
        "scam_fraud": .scam,
    ]

    /// Longest prefix scanned, in UTF-16 units.
    ///
    /// Regex cost grows with input length: measured ~10 ms for all 200 patterns
    /// over 2.4k chars, so an unbounded scan would let a pasted wall of text
    /// (the app accepts them — see the long-message summarizer) stall the guard
    /// queue for seconds. 20k is far past any real message while keeping the
    /// worst case under ~100 ms, and it is still ~40x more text than the model's
    /// 128-token window sees.
    static let scanLimit = 20_000

    /// Patterns in match-precedence order, most-severe category first. The JSON
    /// carries that order in an array (not an object) precisely because Swift
    /// dictionaries would lose it — see `scripts/build_regex_patterns.py`.
    private let rules: [(label: SafetyLabel, severity: Severity,
                         pattern: String, regex: NSRegularExpression)]

    var count: Int { rules.count }

    /// Load and compile every pattern. Returns nil only if the file is missing or
    /// malformed; individual patterns that fail to compile are skipped and logged,
    /// so one bad rule in a new release degrades coverage instead of disabling it.
    init?(jsonURL: URL) {
        guard let data = try? Data(contentsOf: jsonURL),
              let doc = try? JSONDecoder().decode(Document.self, from: data) else {
            NSLog("[GuardRules] %@ missing or malformed", jsonURL.lastPathComponent)
            return nil
        }

        var compiled: [(SafetyLabel, Severity, String, NSRegularExpression)] = []
        for group in doc.categories {
            guard let label = Self.labels[group.category] else {
                NSLog("[GuardRules] unknown category '%@' — skipping %d patterns",
                      group.category, group.patterns.count)
                continue
            }
            for item in group.patterns {
                // Case-insensitive matches the upstream `re.IGNORECASE`. The
                // build script compiles every pattern under Python's `re`, and
                // GuardRulesTests checks they all survive ICU too.
                guard let regex = try? NSRegularExpression(pattern: item.re,
                                                           options: [.caseInsensitive]) else {
                    NSLog("[GuardRules] pattern failed to compile: %@", item.re)
                    continue
                }
                // An out-of-range severity means the generator and the app
                // disagree; treat it as the most serious reading rather than
                // dropping the rule.
                let severity = Severity(rawValue: item.severity) ?? .critical
                compiled.append((label, severity, item.re, regex))
            }
        }

        guard !compiled.isEmpty else {
            NSLog("[GuardRules] no usable patterns in %@", jsonURL.lastPathComponent)
            return nil
        }
        self.rules = compiled
        self.isAuthored = doc.authored ?? true
    }

    /// The most serious pattern that matches `text`, or nil if none do.
    ///
    /// Ties break on category precedence — the file is ordered most-severe-first,
    /// so text matching both a sexual and a child-safety pattern at equal
    /// severity reports child safety.
    ///
    /// Severity, not position, decides because patterns now carry their own:
    /// returning the first hit would let a `caution` rule in an early category
    /// (the bare name of a banned knife) mask a `critical` one later (an offer to
    /// sell it). Scanning stops the moment something Critical fires, since
    /// nothing can outrank it — and a message that matches nothing costs the same
    /// full scan it always did.
    func firstMatch(in text: String) -> Hit? {
        let ns = text as NSString
        let range = NSRange(location: 0, length: min(ns.length, Self.scanLimit))
        guard range.length > 0 else { return nil }

        var best: Hit?
        for rule in rules {
            if let best, best.severity.rawValue >= rule.severity.rawValue { continue }
            guard rule.regex.firstMatch(in: text, options: [], range: range) != nil else { continue }
            let hit = Hit(label: rule.label, severity: rule.severity, pattern: rule.pattern)
            if rule.severity == .critical { return hit }
            best = hit
        }
        return best
    }
}
