import Foundation

/// On-device text-guardrail engine: the tokenizer + the TFLite BERT-Tiny
/// classifier (via `GuardLM`), wrapped in a serialized async API — the same
/// shape as `SLMEngine`, but for a fixed classifier instead of the generative
/// model.
///
/// Runs on its own **`.utility`** queue, deliberately lower priority than the
/// LLM engines, so per-message safety classification never contends the UI.
/// Replaces the previous LLM-prompt guardrail — classification is now
/// deterministic and ~1–5 ms, and doesn't consume a task-model generation.
final class GuardEngine: @unchecked Sendable {

    /// One language's loaded assets. The interpreter is NOT thread-safe, which is
    /// the other reason everything here stays on the engine's serial queue.
    private final class LoadedPack {
        let bridge = GuardLM()
        let tokenizer: GuardTokenizer
        /// Bumped on every use; the smallest value is evicted first.
        var lastUsed: UInt64 = 0

        init(tokenizer: GuardTokenizer) { self.tokenizer = tokenizer }
    }

    // All mutable state is touched only on this serial queue, so the type is safe
    // to share across the async boundary — and concurrent classify calls for a
    // not-yet-loaded language coalesce onto one load for free.
    private let queue = DispatchQueue(label: "com.example.slmchat.guard", qos: .utility)

    /// Loaded packs, keyed by language, capped at `maxResidentPacks`.
    private var packs: [TranslateLanguage: LoadedPack] = [:]
    private var useClock: UInt64 = 0

    /// How many classifiers stay in memory at once.
    ///
    /// Sized to hold every language the app plans to support, because a pack costs
    /// far less resident memory than its file size suggests: **measured at ~3 MB**,
    /// not the ~10 MB of the `.tflite`, since TFLite memory-maps the model and only
    /// the tensor arena and tokenizer vocabulary land in the footprint. Eight slots
    /// is therefore ~25 MB — noise beside the SLM.
    ///
    /// This was 2, chosen against a wrong ~10 MB-per-pack assumption. Two slots
    /// thrash badly once the catalog grows: with seven languages, a chat cycling
    /// through three or more evicts on every message and pays the ~50 ms reload
    /// each time, forever. Holding them all turns that into ~50 ms once per
    /// language per app run.
    ///
    /// Packs are still loaded LAZILY — only the native language is preloaded at
    /// warm-up. Eagerly loading all seven would put ~600–750 ms of work into
    /// startup for languages the user may never receive a message in.
    static let maxResidentPacks = 8

    /// Rule sets for the active region, keyed by the language they were WRITTEN
    /// in — which is the language they can match.
    ///
    /// One set per region was the original design and it was wrong in a way that
    /// only showed up across languages: `uk-en` cannot match a Vietnamese scam,
    /// so a UK reader with Vietnamese contacts had no rule coverage at all. See
    /// `GuardRegion.ruleLanguages`.
    private var rules: [TranslateLanguage: GuardRules] = [:]

    /// Fallback for a language this region has no authored set for. Baseline
    /// coverage (the universal child-safety terms) beats none.
    private var universalRules: GuardRules?

    /// How many patterns are live across every language of the active region —
    /// surfaced in Settings so "rules are loaded" is visible rather than assumed.
    private(set) var ruleCount = 0

    /// Most severe hit across EVERY rule set the region ships. Queue-only.
    ///
    /// All sets, not just the one matching the detected language. Selecting one
    /// by language was the obvious reading of "keyed by (region, language)" and it
    /// is wrong: it makes rule coverage depend on language detection being
    /// correct, and detection abstains on short messages — exactly the ones the
    /// rules are best at. A misdetected language would silently drop to a set with
    /// no patterns for the text.
    ///
    /// Running them all costs nothing to correctness, because a pattern only
    /// matches text in its own language: English scam patterns cannot fire on
    /// Vietnamese and vice versa. The language key organises AUTHORING — one file
    /// per language a region serves — not dispatch.
    ///
    /// Cost is linear in total patterns: the UK is 127 + 21 = 148, against a
    /// measured ~6.5 ms for 200. Under the cost of a single inference window.
    private func ruleHit(in text: String) -> GuardRules.Hit? {
        let sets = rules.isEmpty ? [universalRules].compactMap { $0 } : Array(rules.values)
        var best: GuardRules.Hit?
        for set in sets {
            guard let hit = set.firstMatch(in: text) else { continue }
            if best == nil || hit.severity.rawValue > best!.severity.rawValue { best = hit }
            if best?.severity == .critical { break }
        }
        return best
    }

    /// Load every rule set the region declares, plus the fallback. Queue-only.
    ///
    /// Eager rather than lazy, unlike the model packs: a region has at most a
    /// handful of sets, they are tens of KB, and compiling them costs a few ms —
    /// where a pack is ~10 MB on disk and 3 MB resident.
    private func loadRuleSets(for region: GuardRegion?) {
        rules.removeAll()
        universalRules = nil
        ruleCount = 0
        rulesAuthored = true
        guard let region else { return }

        if let url = ResourceLookup.url(Self.universalRulesResource, ext: "json") {
            universalRules = GuardRules(jsonURL: url)
        }

        var authoredAny = false
        for language in region.ruleLanguages {
            let name = region.rulesResource(for: language)
            guard let url = ResourceLookup.url(name, ext: "json"),
                  let set = GuardRules(jsonURL: url) else {
                NSLog("[GuardRules] %@.json missing — %@ falls back to universal",
                      name, language.name)
                continue
            }
            rules[language] = set
            ruleCount += set.count
            authoredAny = authoredAny || set.isAuthored
            NSLog("[GuardRules] %@: %d patterns (authored: %@)",
                  name, set.count, set.isAuthored ? "yes" : "no")
        }
        // "Authored" describes the REGION: it has contributed patterns of its own
        // in at least one language. A region running purely on the fallback says
        // so in Settings rather than presenting 3 inherited patterns as coverage.
        rulesAuthored = authoredAny
        if rules.isEmpty { ruleCount = universalRules?.count ?? 0 }
    }

    private static let universalRulesResource = GuardRegion.universalRulesResource

    /// False when the active region ships only the universal baseline, i.e. its
    /// rule set hasn't been written yet.
    private(set) var rulesAuthored = true

    /// Classes the active region acts on — see `GuardRegion.actionableClasses`.
    /// Defaults to all of them, so an engine warmed without a region behaves
    /// exactly as it did before this existed.
    private var actionable: Set<SafetyLabel> = Set(SafetyLabel.allCases)

    /// Minimum `p_unsafe` to treat a model prediction as Unsafe.
    ///
    /// 0.95, measured against the `v16ls` (label-smoothed) checkpoint. Re-swept
    /// with `scripts/sweep_guard_threshold.py` over the 285 labelled demo lines
    /// using this engine's own windowing, model-only (rules excluded, so a loose
    /// gate cannot hide behind a pattern):
    ///
    ///     threshold   caught /104   recovered /25   false positives /156
    ///        0.75          82             9                7
    ///        0.85          79             8                6
    ///        0.90          79             8                6
    ///        0.95          74             7                4
    ///        0.97          70             6                3
    ///        0.99           0             0                0
    ///
    /// Three columns, because `shouldFlag: false` in `GuardrailDemo` does not
    /// mean "harmless" — it also covers lines the demo documents as known misses.
    /// *Recovered* is that set: genuinely unsafe text the previous checkpoint
    /// could not reach. *False positives* is the genuinely clean set only.
    ///
    /// ⚠️ **False positives no longer reach zero at any usable threshold, and
    /// that is a regression.** The previous checkpoint had 0 across 0.75–0.95;
    /// this one has 4 even at 0.95. Two of them matter more than the count:
    ///
    ///  - "Para ya." — the Spanish victim replying *stop it* — scores 0.9837
    ///    `suicide`. It scored 0.712 before, and `GuardrailDemo` calls it out by
    ///    name as the nearest thing to a false positive in the whole corpus.
    ///  - "Cuộc sống nặng nề quá. Mình cứ nghĩ đến việc nếu mình…" scores 0.9113
    ///    `suicide`. First-person distress, which is exactly the shape v16 was
    ///    brought in to stop flagging.
    ///
    /// 0.95 is therefore the least-bad point on a worse curve rather than a knee:
    /// there is no knee. It is chosen over the manifest's 0.9 because two fewer
    /// clean lines flagged is worth five true positives here — the rules catch
    /// most of those five, and nothing catches a false positive. The trade the
    /// checkpoint offers overall is 8 recovered misses for 4 false positives; if
    /// that is judged the wrong way round, the fix is to roll the checkpoint back
    /// to `2026-08-26`, not to move this number further.
    ///
    /// 0.99 catches nothing because label smoothing (0.1) caps the achievable
    /// `p_unsafe` at about 0.9875. The usable range is roughly 0.5–0.98, and a
    /// threshold set above that ceiling silently disables the model gate
    /// entirely — which is why 0.99 is in the swept table rather than left out.
    ///
    /// Note this number is not comparable to the 0.5 it replaced three releases
    /// ago. That gate read the WINNING CLASS's probability; this one reads total
    /// unsafe mass.
    ///
    /// The per-class `severityRange` table that used to sit here is gone with the
    /// severity head. It clamped head 2's output into a per-class band; there is
    /// no head 2 to clamp, and reconstructing severity from the class is exactly
    /// what `Severity`'s documentation explains we are not doing.
    static let threshold: Float = 0.95

    // MARK: - Backlog depth

    /// Messages enqueued but not yet classified, and the high-water mark.
    ///
    /// The single most useful number under load, and the one neither of the other
    /// two clocks reports. `work` stays flat at ~4.5 ms however busy things get,
    /// and `queued` only tells you how long ONE message waited — by the time it
    /// is large the pile-up has already happened. Depth says how far behind the
    /// screener is RIGHT NOW, which is what a flood control would have to act on.
    ///
    /// Guarded by a lock rather than the engine queue, because it has to be
    /// incremented by the CALLER, before the work is submitted — counting it on
    /// the queue would only ever observe 1.
    private let depthLock = NSLock()
    private var pending = 0
    private var peakPending = 0

    /// Current and peak queue depth. `resetPeak` clears the high-water mark so a
    /// measurement run can start from a known point.
    func backlogDepth(resetPeak: Bool = false) -> (pending: Int, peak: Int) {
        depthLock.lock()
        defer {
            if resetPeak { peakPending = pending }
            depthLock.unlock()
        }
        return (pending, peakPending)
    }

    private func enterQueue() {
        depthLock.lock()
        pending += 1
        peakPending = max(peakPending, pending)
        depthLock.unlock()
    }

    private func leaveQueue() -> Int {
        depthLock.lock()
        pending -= 1
        let now = pending
        depthLock.unlock()
        return now
    }

    /// Load the rules and the pack for `language`. Serialized on the engine queue.
    ///
    /// Only that one pack is preloaded — the user's own language, so the common
    /// path never waits on a load. Every other language loads lazily on first use.
    ///
    /// `rulesName` is best-effort: a missing pattern file leaves `rules` nil and
    /// the engine model-only, reported through `ruleCount` rather than as a load
    /// failure. A missing MODEL is a real failure, because without it nothing is
    /// screened at all.
    func warmUp(language: TranslateLanguage,
                region: GuardRegion?) async -> (ok: Bool, error: String?) {
        await withCheckedContinuation { cont in
            queue.async { [weak self] in
                guard let self else { cont.resume(returning: (false, "deallocated")); return }
                self.actionable = region?.actionableClasses ?? Set(SafetyLabel.allCases)
                self.loadRuleSets(for: region)

                guard let pack = GuardPackCatalog.pack(for: language) else {
                    cont.resume(returning: (false, "no guardrail pack for \(language.name)"))
                    return
                }
                switch self.loadPack(pack) {
                case .success:          cont.resume(returning: (true, nil))
                case .failure(let why): cont.resume(returning: (false, why))
                }
            }
        }
    }

    /// Swap in a different region's rule set, keeping the loaded model packs.
    ///
    /// Rules are the only part of the guardrail a region change touches — the
    /// classifier is per-language and says the same thing wherever it runs, so
    /// reloading it here would cost ~10 MB of work to reach the same verdict.
    ///
    /// Returns the new pattern count. A region whose file is missing leaves the
    /// engine model-only rather than keeping the previous region's rules, which
    /// would silently screen the user under a jurisdiction they just left.
    @discardableResult
    func loadRules(for region: GuardRegion?) async -> Int {
        await withCheckedContinuation { cont in
            queue.async { [weak self] in
                guard let self else { cont.resume(returning: 0); return }
                self.actionable = region?.actionableClasses ?? Set(SafetyLabel.allCases)
                self.loadRuleSets(for: region)
                cont.resume(returning: self.ruleCount)
            }
        }
    }

    // MARK: - Pack cache

    /// Fetch `language`'s pack, loading it if it isn't resident. Queue-only.
    /// Returns nil when the language has no pack or its assets won't load.
    private func residentPack(for language: TranslateLanguage) -> LoadedPack? {
        if let existing = packs[language] {
            useClock += 1
            existing.lastUsed = useClock
            return existing
        }
        guard let pack = GuardPackCatalog.pack(for: language) else { return nil }
        guard case .success(let loaded) = loadPack(pack) else { return nil }
        return loaded
    }

    /// Outcome of a pack load. A plain enum rather than `Result`, whose failure
    /// type must be an `Error` — the reason here is a message for the Settings
    /// sheet, not something anyone throws.
    private enum PackLoad {
        case success(LoadedPack)
        case failure(String)
    }

    /// Load one pack and admit it to the cache, evicting the least-recently-used
    /// pack if that would exceed `maxResidentPacks`. Queue-only.
    @discardableResult
    private func loadPack(_ pack: GuardPack) -> PackLoad {
        if let existing = packs[pack.language] { return .success(existing) }

        guard let modelPath = ResourceLookup.path(pack.modelResource, ext: "tflite") else {
            return .failure("\(pack.modelResource).tflite not in bundle — add it to Resources/")
        }
        guard let tokURL = ResourceLookup.url(pack.tokenizerResource, ext: "json"),
              let tok = GuardTokenizer(tokenizerJSONURL: tokURL) else {
            return .failure("\(pack.tokenizerResource).json missing or invalid")
        }
        let loaded = LoadedPack(tokenizer: tok)
        guard loaded.bridge.load(modelPath: modelPath) else {
            return .failure(loaded.bridge.lastError)
        }

        // Evict BEFORE inserting, so the cache never briefly exceeds its cap.
        while packs.count >= Self.maxResidentPacks,
              let oldest = packs.min(by: { $0.value.lastUsed < $1.value.lastUsed })?.key {
            packs.removeValue(forKey: oldest)
            NSLog("[GuardEngine] evicted %@ pack", oldest.name)
        }

        useClock += 1
        loaded.lastUsed = useClock
        packs[pack.language] = loaded
        NSLog("[GuardEngine] loaded %@ pack (%@)", pack.language.name, pack.modelResource)
        return .success(loaded)
    }

    /// Classify one message: rules first, then the model.
    ///
    /// A rule hit short-circuits — the tokenizer and the interpreter never run,
    /// mirroring the release's `infer.py`, where the regex pass is "Step 1" and a
    /// match returns UNSAFE without consulting the model. That ordering is only
    /// safe because rules can exclusively raise a verdict; nothing a pattern does
    /// can clear a message the model would have flagged.
    ///
    /// Fails open to `.clean` if the engine isn't loaded or the model can't run,
    /// so a transient failure never blocks the chat (a later message re-triggers
    /// classification).
    /// `language` selects the classifier. A language with no pack is NOT screened
    /// by the model — the rules still run, since a pattern that doesn't belong to
    /// the text's language simply won't match.
    func classify(_ text: String, language: TranslateLanguage) async -> SafetyResult {
        await classify(text, language: language, useRules: true)
    }

    /// The model's own verdict, with the rule pre-filter bypassed.
    ///
    /// Exists for `GuardEngineTests`: its fixtures are reference outputs of the
    /// ORIGINAL Python model, so they pin the Swift port of THAT. Letting rules
    /// intercept them would test the patterns instead and quietly stop testing
    /// the thing the fixtures were captured for. Not used by the app.
    func classifyModelOnly(_ text: String, language: TranslateLanguage) async -> SafetyResult {
        await classify(text, language: language, useRules: false)
    }

    private func classify(_ text: String,
                          language: TranslateLanguage,
                          useRules: Bool) async -> SafetyResult {
        await withCheckedContinuation { cont in
            let queued = DispatchTime.now().uptimeNanoseconds
            enterQueue()
            queue.async { [weak self] in
                guard let self else { _ = self?.leaveQueue(); cont.resume(returning: .clean); return }

                // Two clocks, because they answer different questions and the
                // difference between them IS the story on a busy thread. `work`
                // is what one message costs the CPU. `wait` is how long it sat
                // behind other messages on this serial queue — which grows
                // linearly during a look-ahead sweep and is invisible if you time
                // only the inference.
                //
                // Both travel out on the verdict rather than being logged here.
                // This used to print its own `[GuardEngine] Xms work, Yms queued`
                // line, which meant a slow `[Guard]` line could only be explained
                // by finding its partner further up the log — and under load the
                // queue interleaves messages, so the two lines next to each other
                // need not describe the same message.
                let began = DispatchTime.now().uptimeNanoseconds
                let beganCPU = clock_gettime_nsec_np(CLOCK_THREAD_CPUTIME_ID)
                var windowCount = 0
                let report: (SafetyResult) -> Void = { verdict in
                    let now = DispatchTime.now().uptimeNanoseconds
                    let nowCPU = clock_gettime_nsec_np(CLOCK_THREAD_CPUTIME_ID)
                    var stamped = verdict
                    stamped.timing = GuardTiming(
                        waitMS: Double(began - queued) / 1_000_000,
                        workMS: Double(now - began) / 1_000_000,
                        cpuMS: Double(nowCPU - beganCPU) / 1_000_000,
                        windows: windowCount,
                        behind: self.leaveQueue())
                    cont.resume(returning: stamped)
                }

                let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                if trimmed.isEmpty { report(.clean); return }

                // Rules first, and independent of the pack: they scan the full
                // text where the model sees 128 tokens, and they are the only
                // coverage a language without a pack gets.
                if useRules, let hit = self.ruleHit(in: trimmed) {
                    if self.actionable.contains(hit.label) {
                        report(Self.verdict(for: hit)); return
                    }
                    // Matched, but this region doesn't act on the class. Fall
                    // through to the model rather than returning clean — another
                    // pattern isn't consulted, but the classifier still is.
                    NSLog("[GuardRules] matched %@ but region does not act on it — ignoring",
                          hit.label.displayName)
                }

                guard let pack = self.residentPack(for: language) else {
                    NSLog("[GuardEngine] no pack for %@ — not screened by the model",
                          language.name)
                    report(.clean); return
                }

                // Long text is screened in windows: the tokenizer truncates at 128
                // tokens, so a single pass is blind to everything past that. Short
                // text yields one chunk and takes the identical path it always did.
                let chunks = Self.windows(trimmed, tokenizer: pack.tokenizer)
                var worst: SafetyResult?

                for chunk in chunks {
                    windowCount += 1
                    let enc = pack.tokenizer.encode(chunk)
                    let idsData = enc.ids.withUnsafeBytes { Data($0) }
                    let maskData = enc.mask.withUnsafeBytes { Data($0) }
                    guard let heads = pack.bridge.classify(inputIds: idsData,
                                                           attentionMask: maskData) else {
                        continue    // a failed window shouldn't discard the others
                    }
                    let verdict = Self.decide(heads: heads, actionable: self.actionable)
                    worst = Self.moreSevere(worst, verdict)
                    // Nothing above Critical, so further windows can't change the
                    // outcome — stop paying for them.
                    if worst?.severity == .critical { break }
                }

                report(worst ?? .clean)
            }
        }
    }

    /// Languages with a pack currently in memory. Test/diagnostic seam — a verdict
    /// can't reveal which pack ran while the packs are byte-identical copies.
    func residentLanguages() async -> Set<TranslateLanguage> {
        await withCheckedContinuation { cont in
            queue.async { [weak self] in
                cont.resume(returning: Set(self?.packs.keys ?? [:].keys))
            }
        }
    }

    // MARK: - Windowing long text

    /// Most windows screened for one message.
    ///
    /// Every window costs a FULL forward pass — `encode` pads to a fixed 128
    /// tokens, so a 13-token sentence is exactly as expensive as a full one.
    /// Measured on the simulator at ~9–11 ms per window, so this cap puts the
    /// worst case around 160 ms of background work for one message.
    ///
    /// ⚠️ Known limit: a message with more than this many sentences is screened
    /// only up to the cap, so padding with benign sentences can push content out
    /// of the model's reach. The rules are the backstop — they scan the whole
    /// text up to `GuardRules.scanLimit` (20k chars) and are unaffected by this.
    /// Raising it trades latency for reach, linearly.
    static let maxWindows = 16

    /// Split `text` into the pieces the model is asked to judge.
    ///
    /// **One sentence per window, not "as much as fits".** The obvious design —
    /// pack sentences up to the 128-token limit — was measured and does not work:
    /// the model judges a window by its dominant content, so
    /// `"Tao sẽ giết mày."` (Violence/Violation on its own, conf 0.911) comes back
    /// **Clean** when packed into a 31-token window behind one benign sentence.
    /// A threat buried in chat is exactly the case this has to catch, so the unit
    /// has to be small enough that a violation dominates its own window.
    ///
    /// The whole message is ALSO screened as one window whenever it fits, so
    /// anything that needs the surrounding context to read as unsafe is still
    /// caught the way it was before sentences were split out.
    ///
    /// Single-sentence text — most messages — returns `[text]` and takes exactly
    /// the path it always did.
    static func windows(_ text: String, tokenizer: GuardTokenizer) -> [String] {
        let limit = tokenizer.maxLen
        let pieces = sentences(text)

        guard pieces.count > 1 else {
            // One sentence: only split if it alone overflows the window.
            return tokenizer.tokenCount(of: text) > limit
                ? Array(hardSplit(text, tokenizer: tokenizer, limit: limit).prefix(maxWindows))
                : [text]
        }

        var out: [String] = []
        // The context pass. Skipped when the message overflows the window, since
        // `encode` would truncate it and the sentence windows cover it anyway.
        if tokenizer.tokenCount(of: text) <= limit { out.append(text) }

        for piece in pieces {
            if out.count >= maxWindows { break }
            if tokenizer.tokenCount(of: piece) > limit {
                for part in hardSplit(piece, tokenizer: tokenizer, limit: limit) {
                    if out.count >= maxWindows { break }
                    out.append(part)
                }
            } else {
                out.append(piece)
            }
        }
        return out.isEmpty ? [text] : out
    }

    /// Sentence-ish units: a run ending at a terminator or newline.
    private static func sentences(_ text: String) -> [String] {
        var out: [String] = []
        var current = ""
        for ch in text {
            current.append(ch)
            if ch == "\n" || ch == "." || ch == "!" || ch == "?" || ch == "。" {
                let t = current.trimmingCharacters(in: .whitespacesAndNewlines)
                if !t.isEmpty { out.append(t) }
                current = ""
            }
        }
        let tail = current.trimmingCharacters(in: .whitespacesAndNewlines)
        if !tail.isEmpty { out.append(tail) }
        return out
    }

    /// Last resort for a single sentence that exceeds the window: pack by words.
    private static func hardSplit(_ s: String,
                                  tokenizer: GuardTokenizer,
                                  limit: Int) -> [String] {
        var out: [String] = []
        var current: [String] = []
        var count = 0
        for word in s.split(separator: " ").map(String.init) {
            let n = tokenizer.tokenCount(of: word)
            if count + n > limit, !current.isEmpty {
                out.append(current.joined(separator: " "))
                current = []
                count = 0
            }
            current.append(word)
            count += n
        }
        if !current.isEmpty { out.append(current.joined(separator: " ")) }
        return out
    }

    /// Fold two window verdicts, keeping the more serious.
    ///
    /// Unsafe beats clean, then higher severity, then higher confidence. A message
    /// is as unsafe as its worst part — averaging or majority-voting would let a
    /// long benign message bury one flagged paragraph.
    private static func moreSevere(_ a: SafetyResult?, _ b: SafetyResult) -> SafetyResult {
        guard let a else { return b }
        if a.isUnsafe != b.isUnsafe { return a.isUnsafe ? a : b }
        // Severity ranks only when BOTH windows have one. Since v16 that is the
        // clean-vs-clean case; flagged model verdicts carry nil, and a rule hit
        // never reaches this fold because it short-circuits before the model
        // runs. Treating nil as 0 would have made an unassessed flagged window
        // lose to any window that happened to carry `.safe`.
        if let sa = a.severity, let sb = b.severity, sa != sb {
            return sa.rawValue > sb.rawValue ? a : b
        }
        return a.confidence >= b.confidence ? a : b
    }

    // MARK: - Decision logic

    /// Turn a matched rule into a verdict.
    ///
    /// Severity comes from the PATTERN, not from the class's ceiling. Every
    /// pattern used to land on the ceiling, which held while the only rule set
    /// was the precision-tuned Vietnamese one — its phrasings ("tao sẽ giết
    /// mày", "child porn") are not the borderline cases severity exists to
    /// soften. The UK set broke that assumption: it deliberately contains rules
    /// that are evidence rather than proof (a reclaimed slur, the mere name of a
    /// banned weapon), and blurring those outright is how a guardrail loses the
    /// user's trust. The generator still defaults a pattern to its category
    /// ceiling, so a file of bare strings behaves exactly as it did before.
    ///
    /// Confidence is 1 because a rule is not probabilistic. It is NOT compared
    /// against `threshold`: that gate exists to suppress the model's low-
    /// confidence guesses, and a pattern match has no equivalent doubt.
    ///
    /// The pattern travels with the verdict rather than only being logged here,
    /// so the one `[Guard]` line the caller prints can name the rule. Previously
    /// that meant correlating two log statements from different subsystems, which
    /// is unreliable under load — the guard queue interleaves messages.
    private static func verdict(for hit: GuardRules.Hit) -> SafetyResult {
        SafetyResult(label: hit.label, severity: hit.severity, confidence: 1,
                     source: .rules(pattern: hit.pattern))
    }

    /// Map the v16 model's single output head to a verdict.
    ///
    /// The decision rule is the one the checkpoint was released with, quoted from
    /// its manifest:
    ///
    ///     p_unsafe = 1 - softmax(logits)[0]
    ///     flag unsafe when p_unsafe >= threshold
    ///     final class = argmax(logits) when unsafe, else clean
    ///
    /// Note what the confidence is measured over. The old rule took the WINNING
    /// class's probability, so a message the model read as unsafe but couldn't
    /// place — 0.3 violence, 0.3 harassment, 0.3 scam — scored 0.3 and passed as
    /// clean, even though it put 0.9 on "unsafe" overall. Summing the mass first
    /// fixes exactly that case, and it is why the threshold moved 0.5 → 0.8: the
    /// two numbers are not on the same scale and comparing them is meaningless.
    ///
    /// The head is identified by width. A checkpoint whose head is not
    /// `modelClassCount` wide is REFUSED rather than interpreted, because the
    /// alternative failure is silent: v15's 10-wide head read under v16's rules
    /// mislabels every class from `child` up, and a 4-wide severity head read as
    /// classes returns confident nonsense. Neither throws, and both look like a
    /// working guardrail.
    private static func decide(heads: [[NSNumber]],
                               actionable: Set<SafetyLabel>) -> SafetyResult {
        guard let head = heads.first(where: { $0.count == SafetyLabel.modelClassCount })
        else {
            NSLog("[GuardEngine] no %d-wide head in output (%@) — refusing to "
                + "interpret; message NOT screened by the model",
                  SafetyLabel.modelClassCount,
                  heads.map { String($0.count) }.joined(separator: ","))
            return .clean   // `.notScreened`, not a clean verdict — see below.
        }

        let probs = softmax(head.map { $0.floatValue })
        // Total mass on everything that is not class 0. Complement of clean
        // rather than a sum over the rest: one subtraction, no accumulated
        // floating-point drift across seven terms.
        let pUnsafe = 1 - probs[0]
        let label = SafetyLabel.fromModelIndex(argmax(probs)) ?? .clean

        // The region gate sits alongside the confidence gate, and for the same
        // reason: both decide whether a detection is worth acting on. The model
        // has no notion of jurisdiction, so this is the only place its output can
        // be made regional.
        //
        // `severity: nil` — v16 has no severity head, and inventing one from the
        // class would make the blur gate act on a number no model produced.
        if label.isUnsafe && pUnsafe >= threshold && actionable.contains(label) {
            return SafetyResult(label: label, severity: nil,
                                confidence: pUnsafe, source: .model)
        }
        // Clean, but the model DID run — `.model`, not `.notScreened`. The
        // difference is the whole point of the source field: "screened and found
        // nothing" and "never screened" look identical in the UI and are opposite
        // findings when a miss is being investigated.
        //
        // Severity IS known here, and it is `.safe`: the model actively assessed
        // this text and placed it in class 0. That is a real finding, unlike the
        // nil on a flagged verdict, where nothing assessed how serious it was.
        return SafetyResult(label: .clean, severity: .safe,
                            confidence: pUnsafe, source: .model)
    }

    private static func softmax(_ x: [Float]) -> [Float] {
        let m = x.max() ?? 0
        let e = x.map { Foundation.exp($0 - m) }
        let sum = e.reduce(0, +)
        return sum > 0 ? e.map { $0 / sum } : e
    }

    private static func argmax(_ x: [Float]) -> Int {
        var bestIdx = 0
        var bestVal = -Float.greatestFiniteMagnitude
        for (i, v) in x.enumerated() where v > bestVal { bestVal = v; bestIdx = i }
        return bestIdx
    }
}
