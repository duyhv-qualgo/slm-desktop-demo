import Foundation

/// Target language for the translation feature. Also used to model the user's
/// native language chosen in Settings.
///
/// The list includes languages that are not usable on every checkpoint — see
/// `isSupported(forBaseModel:)`. They are declared here rather than hidden so
/// Settings can show what a model does and doesn't cover; every path that acts on
/// a language must gate on that first.
enum TranslateLanguage: String, CaseIterable, Identifiable {
    case vietnamese
    case english
    case arabic
    case spanish
    case french
    case russian
    case chinese
    /// Guardrail-only, for now.
    ///
    /// German has a classifier pack (`guardrail-v16-de`) but no LoRA language
    /// pack, so `isSupported(forBaseModel:)` returns false for it and Settings
    /// never offers it as a native language — translate and reply suggestions
    /// stay out of reach until a `lora-*-german.gguf` ships.
    ///
    /// It is declared anyway because the alternative was worse than an unused
    /// file. Without a case here, `TranslateLanguage(detectorCode:)` returns nil
    /// for German, so `identify()` walks past `.german` to whatever hypothesis
    /// ranks next — and a German message gets screened by the ENGLISH pack,
    /// which since v16 is a genuinely different model that never saw the
    /// language. Declaring the case routes it to its own pack instead.
    case german

    var id: String { rawValue }

    /// Language name as used in the finetune's prompt ("Vietnamese" / "English").
    var name: String {
        switch self {
        case .vietnamese: return "Vietnamese"
        case .english:    return "English"
        case .arabic:     return "Arabic"
        case .spanish:    return "Spanish"
        case .french:     return "French"
        case .russian:    return "Russian"
        case .chinese:    return "Chinese"
        case .german:     return "German"
        }
    }

    /// Two-letter code used in per-language asset names
    /// (`guardrail-rules-uk-vi.json`, `guardrail-bert-tiny-en.tflite`).
    var shortCode: String {
        switch self {
        case .vietnamese: return "vi"
        case .english:    return "en"
        case .arabic:     return "ar"
        case .spanish:    return "es"
        case .french:     return "fr"
        case .russian:    return "ru"
        case .chinese:    return "zh"
        case .german:     return "de"
        }
    }

    /// Whether the app can run its features in this language on `baseModelID`.
    ///
    /// Support is per (model, language), NOT global: which languages work depends
    /// on the checkpoint that's loaded. Vietnamese and English are the baseline —
    /// every checkpoint in the catalog handles them, and the Gemma-3 task finetune
    /// was trained on exactly that pair. Anything beyond them is earned by a LoRA
    /// language pack, so the catalog is the authority.
    ///
    /// Making this global was fine while one model shipped two languages. It isn't
    /// now: the multilingual Qwen3 v6 packs cover seven, and offering those on
    /// Gemma-3 would run every feature against a language its finetune has never
    /// seen — which fails as plausible-looking wrong output, not as an error.
    func isSupported(forBaseModel baseModelID: String) -> Bool {
        if self == .vietnamese || self == .english { return true }
        return LoRACatalog.adapter(baseModelID: baseModelID, language: self) != nil
    }

    /// Languages a user can actually pick while `baseModelID` is loaded.
    static func supported(forBaseModel baseModelID: String) -> [TranslateLanguage] {
        allCases.filter { $0.isSupported(forBaseModel: baseModelID) }
    }

    /// The other language — the source when this is the translation target.
    /// Everything pairs with English except English itself, which pairs with
    /// Vietnamese (the EN↔VI demo direction).
    var opposite: TranslateLanguage {
        self == .english ? .vietnamese : .english
    }

    /// Bridge from `LangDetect`'s BCP-47 label set, or nil for a language the app
    /// has no pack for (see `Prompts.conversationLanguage`).
    ///
    /// Only the eight languages `GuardPackCatalog` ships a classifier for are
    /// mapped. Everything else — including `unknown`, which is a real label and
    /// not an error — returns nil so `identify` walks past it.
    ///
    /// The detector also emits Latin-script transliteration labels (`zh-Latn`,
    /// `ru-Latn`, `ar-Latn` for pinyin and romanised Cyrillic/Arabic). Those are
    /// deliberately NOT mapped: the guardrail packs and the LoRA language packs
    /// were trained on native script, so routing romanised text to them means
    /// screening it with a model that never saw the form it is written in.
    /// Abstaining sends it to the thread's language instead, which is recoverable.
    init?(detectorCode code: String) {
        switch code {
        case "vi": self = .vietnamese
        case "en": self = .english
        case "ar": self = .arabic
        case "es": self = .spanish
        case "fr": self = .french
        case "ru": self = .russian
        case "zh": self = .chinese
        case "de": self = .german
        default:   return nil
        }
    }
}

/// Pure prompt builders. Every task renders conversation history into the prompt
/// itself because the engine is stateless per call (see `SLMEngine`). Kept short
/// on purpose — each instruction token costs prefill compute on every call, and
/// on a 270M model a tight prompt is a real wall-clock win.
enum Prompts {

    /// Persona name for the chat partner. Shown in the UI and used in prompts.
    static let friendName = "Alex"

    /// The device owner's name, used to label their own messages in the summary
    /// prompt's speaker tags.
    static let userName = "Duy"

    /// Short-term memory: how many trailing messages to feed into a prompt.
    /// ~15 recent turns gives the reply enough context to stay coherent while
    /// still capping prefill cost on the small model.
    static let historyWindow = 15

    /// How many messages the history window advances in ONE step, and why it
    /// isn't 1.
    ///
    /// A plain `suffix(historyWindow)` slides by one message per turn, which
    /// drops the oldest line and rewrites the prompt from its second line on. The
    /// KV cache in `ChatLM` can only reuse an identical LEADING run of tokens, so
    /// a sliding window destroys the reuse entirely — measured: 391-token prompt,
    /// only the 32-token instruction cached, 359 tokens re-prefilled at ~105
    /// tok/s = 3.4s, every single turn.
    ///
    /// Advancing in blocks of this size instead keeps the prompt's prefix byte-
    /// identical for that many turns, so all but the newest messages are served
    /// from cache. The cost is that the window holds up to `stride - 1` extra
    /// messages between steps, i.e. slightly MORE context, never less — a
    /// quality-neutral-to-positive trade for a large latency win.
    ///
    /// Note this sets how OFTEN a full prefill happens, independent of window
    /// size: a chat gains ~2 messages per turn, so the window steps roughly every
    /// `stride / 2` turns. Window size sets what each of those steps COSTS.
    static let historyWindowStride = 5

    /// History window for reply SUGGESTIONS specifically — deliberately shorter
    /// than `historyWindow`.
    ///
    /// Prefill dominates the auto-suggest turn and scales linearly with this: at
    /// ~24 tokens per message and ~100 tok/s on an A12 CPU, 15 messages cost ~3.9s
    /// to re-read, 8 cost ~2.2s. The finetune is picking a reaction to the LATEST
    /// message rather than summarising the thread, so the older turns earn little.
    /// Chat replies keep the full `historyWindow` — they need continuity.
    static let suggestWindow = 8

    /// Window stride for reply SUGGESTIONS, split from `historyWindowStride` so it
    /// can be tuned against the 8-message `suggestWindow` without moving Magic
    /// Compose or chat history.
    ///
    /// A chat gains ~2 messages per suggest (your reply + theirs), so stride `s`
    /// re-anchors about every `s / 2` suggests and the window runs 8 … 8+s−1.
    /// Stride 2 therefore re-anchors on EVERY suggest. Chosen from measured
    /// eval tokens in `SuggestKVHarnessTests`, which was removed along with KV
    /// prefix reuse; the stride is kept so prompts stay unchanged. Without
    /// reuse it no longer affects latency.
    ///
    /// Measured 2026-09-23 (20 suggests, gemma / qwen2.5 / qwen3, pre-warm on):
    /// stride 3 ties 5 and 6 for the lowest suggest-time eval (steady turns max
    /// 31–35 tokens) with every re-anchor served from the pre-warmed slot, and
    /// keeps the window at 8–10 messages. Even strides (2, 4) missed a
    /// re-anchor on a friend burst. What 3 costs over 5 is background work:
    /// ~2,100 vs ~1,400 pre-warm tokens over the run, spent while the reply is
    /// still in flight rather than inside the suggest.
    static let suggestStride = 3

    /// Index of the first message to include, quantized to `historyWindowStride`
    /// so the resulting prompt prefix stays stable across consecutive turns.
    /// See `historyWindowStride` for the reasoning.
    static func windowStart(messageCount: Int,
                            window: Int = historyWindow,
                            stride: Int = historyWindowStride) -> Int {
        guard messageCount > window, stride > 1 else { return max(0, messageCount - window) }
        // Round the ideal start DOWN to a stride boundary: the window only moves
        // when a full block has accumulated, and never drops below `window`.
        return ((messageCount - window) / stride) * stride
    }

    // MARK: - Output cleanup

    /// Strip Qwen3 chain-of-thought and nothing else. First remove closed
    /// `<think>…</think>` blocks. Then, if an UNCLOSED `<think>` remains
    /// (mid-stream, before the closing tag arrives), drop everything from it to
    /// the end so reasoning never shows in the UI. Finally clear any stray tags.
    ///
    /// Split out of `clean` because a parser that matches on `[[TAG]]` markers
    /// has to run BEFORE the tags are stripped, but still after the reasoning is
    /// gone — a reasoning model that muses about `[[REFUSE:…]]` in its scratchpad
    /// would otherwise be read as having actually refused. See `parseCompose`.
    static func stripThinking(_ raw: String) -> String {
        var s = raw.replacingOccurrences(of: #"<think>[\s\S]*?</think>"#,
                                         with: "", options: .regularExpression)
        if let open = s.range(of: "<think>") {
            s.removeSubrange(open.lowerBound..<s.endIndex)
        }
        return s.replacingOccurrences(of: #"</?think>"#, with: "",
                                      options: .regularExpression)
    }

    /// Strip control-tag noise a small model sometimes emits — e.g. `[[INQUIRE]]`,
    /// `[[NO_SUGGEST:SENSITIVE]]` — plus surrounding quotes/whitespace, so chat
    /// bubbles, translations, and summaries read cleanly.
    static func clean(_ raw: String) -> String {
        var s = stripThinking(raw)
        // Remove [[...]] tags.
        while let r = s.range(of: #"\[\[[^\]]*\]\]"#, options: .regularExpression) {
            s.removeSubrange(r)
        }
        // Drop a leading persona label the model may echo ("Alex:" / "You:").
        for label in ["\(friendName):", "You:"] {
            if s.trimmingCharacters(in: .whitespaces).hasPrefix(label) {
                s = String(s.trimmingCharacters(in: .whitespaces).dropFirst(label.count))
            }
        }
        // Collapse runs of blank lines and trim quotes/space.
        s = s.replacingOccurrences(of: #"\n{2,}"#, with: "\n", options: .regularExpression)
        return s.trimmingCharacters(in: CharacterSet(charactersIn: " \n\"'"))
    }

    // MARK: - 1. Chat partner persona (used by the API path)

    /// The persona instruction for a given contact. Kept SHORT on purpose: a 0.6B
    /// model follows a terse system prompt far better than a long bulleted one (a
    /// verbose persona made it emit broken meta-replies like "(Excuses, my head is
    /// spinning…)"). Gemma has no `system` role, so this is folded into the first
    /// user turn. Each conversation supplies its own contact name.
    static func persona(for name: String) -> String {
        """
        You are \(name), texting your close friend. Reply in 1-2 short, warm, \
        casual sentences like a real text message. React naturally — don't interrogate \
        them with questions. Don't repeat earlier messages or prefix your reply with a name.
        """
    }

    // MARK: - 1b. Simulated unread backlog (demo)

    /// How many messages a simulated "while you were away" burst aims for. Enough
    /// to make the "Summarize N unread messages" flow read as a real backlog, few
    /// enough to fit one generation on the small model.
    static let backlogBurstCount = 6

    /// One-shot instruction that generates a BURST of consecutive texts from the
    /// friend — the way someone fires off several short messages while you're
    /// offline — so the "Summarize unread" flow has a realistic list to work on.
    /// Unlike `persona` (one turn = one bubble), this asks for the whole pile in a
    /// SINGLE generation; the caller fans the lines out into separate bubbles with
    /// `splitBacklog`. It's CONTEXT-AWARE: the recent conversation is supplied
    /// alongside it (as Gemini `contents` on the API path, or folded into the prompt
    /// on-device), so the burst continues the real thread instead of a random
    /// check-in. `noThinkSwitch` is intentionally NOT baked in — the on-device
    /// caller appends it after the transcript so it lands on the latest user turn.
    static func friendBacklog(from name: String = friendName,
                              count: Int = backlogBurstCount) -> String {
        """
        You are \(name), texting your close friend \(userName). Continue the \
        conversation: write the next \(count) short messages you send one after \
        another while waiting for \(userName) to reply — building naturally on what \
        was just being discussed (a follow-up, a new thought, a question, some \
        news). If there's no prior context, just check in. Put EACH message on its \
        own line. No numbering, no timestamps, no names, no quotes — just the \
        message text.
        """
    }

    /// Split a one-shot backlog generation into individual message texts — one per
    /// non-empty line, run through `clean` (strips `<think>`/`[[tag]]`/label noise)
    /// and stripped of any leading list marker ("1.", "-", "•") the model adds.
    /// Empty lines are dropped so a stray blank never becomes an empty bubble.
    static func splitBacklog(_ raw: String) -> [String] {
        clean(raw)
            .split(separator: "\n")
            .map { line in
                clean(String(line))
                    .trimmingCharacters(in: CharacterSet(charactersIn: "-•*0123456789.)( "))
            }
            .filter { !$0.isEmpty }
    }

    // MARK: - Group chat

    /// System prompt for one bot in the group chat: stay in character, reply with a
    /// single short casual line reacting to the transcript.
    static func groupSystem(bot: String, others: [String]) -> String {
        """
        You are \(bot), chatting in a group with \(others.joined(separator: ", ")) and \(userName). \
        Reply with ONE short, casual message that naturally continues the conversation and \
        reacts to what was just said. Don't prefix your name or narrate.
        """
    }

    /// Render the recent group transcript as `Name: text` lines (windowed).
    static func groupTranscript(_ messages: [any SLMMessage]) -> String {
        messages.suffix(historyWindow).map { m in
            let who = m.isFromUser ? userName : (m.authorName ?? friendName)
            return "\(who): \(m.text)"
        }.joined(separator: "\n")
    }

    // MARK: - 2. Translation

    /// Translate a single message, in the finetune's format. The model replies with
    /// `{"task":"translation","translation":"…"}` — extract it with `parseTranslation`.
    ///
    /// `source` names the language being translated FROM. It defaults to
    /// `target.opposite`, which is right only while every thread is native↔English:
    /// translating a Spanish message into Vietnamese would otherwise announce it as
    /// "the English sentence", which is a plain lie to the model about its input.
    /// Callers that know the source (they detected it to pick the direction) should
    /// pass it.
    static func translate(_ text: String,
                          from source: TranslateLanguage? = nil,
                          to target: TranslateLanguage) -> String {
        """
        Translate the following \((source ?? target.opposite).name) sentence to \(target.name):
        \(text.trimmingCharacters(in: .whitespacesAndNewlines))
        """
    }

    /// Extract the `translation` field from the finetune's JSON output
    /// (`{"task":"translation","translation":"…"}`), falling back to cleaned raw text
    /// if the model didn't emit valid JSON.
    static func parseTranslation(_ raw: String) -> String {
        if let block = firstBraceBlock(in: raw),
           let data = block.data(using: .utf8),
           let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let t = obj["translation"] as? String,
           !t.trimmingCharacters(in: .whitespaces).isEmpty {
            return clean(t)
        }
        return clean(raw)
    }

    /// Translate direction for the Translate action, relative to the user's
    /// `native` language: text that is NOT already predominantly native translates
    /// INTO native; native text translates OUT to the other language. Undetermined
    /// text (e.g. a bare "ok") resolves to native, so the action is a sensible
    /// no-op rather than a wrong-direction guess. A nav-bar toggle can override.
    static func autoTarget(for text: String,
                           native: TranslateLanguage = .vietnamese) -> TranslateLanguage {
        let detected = detectLanguage(text, native: native)
        return detected == native ? native.opposite : native
    }

    // MARK: Language detection

    /// Vietnamese share of `text`'s *classifiable* words — 0 (all English) … 1
    /// (all Vietnamese) — or nil when nothing could be classified (empty text, or
    /// only ambiguous/unknown tokens). This single ratio backs every
    /// language-dependent decision: translate direction, summary/suggestion
    /// language, and the Translate control's visibility (which reads it per
    /// message via the cached `ChatMessage.vietnameseRatio`).
    ///
    /// How each word votes:
    ///  • **Vietnamese** — carries a Vietnamese diacritic, or is a distinctive
    ///    Vietnamese word (`viStrongWords`). This is what lets us detect casual
    ///    no-diacritic typing ("khong biet", "ban khoe khong"), not just accented
    ///    text.
    ///  • **English** — a common English word (`enCommonWords`), or a token that
    ///    is orthographically impossible as a single Vietnamese syllable (uses
    ///    f/j/w/z, or is longer than any Vietnamese syllable). The orthographic
    ///    rule classifies English content words without an exhaustive list.
    ///  • **Ambiguous / unknown** — valid in both languages, or unrecognised;
    ///    excluded from the ratio so a mixed message scores by its *distinctive*
    ///    words rather than its shared filler.
    static func vietnameseRatio(_ text: String) -> Double? {
        // NFC-normalize so decomposed input (base + combining mark) matches the
        // precomposed scalars in `viDiacritics`, then tokenize on non-letters.
        let words = text.precomposedStringWithCanonicalMapping
            .lowercased()
            .split { !$0.isLetter }
            .map(String.init)
        guard !words.isEmpty else { return nil }

        var vi = 0, en = 0
        for w in words {
            // A token outside the Latin script is neither — abstain before the
            // English test below can misfire on it. `isOrthographicallyNonVietnamese`
            // votes English for anything over 7 letters, which is true of ordinary
            // Russian and Arabic words and of an entire Chinese sentence (Han text
            // has no spaces, so it tokenizes as one long "word"). Without this
            // guard a Russian message scores 1.0 English, the Translate action
            // reads it as English, and asks to translate Russian into Russian.
            guard isLatinScript(w) else { continue }
            if w.unicodeScalars.contains(where: viDiacritics.contains) || viStrongWords.contains(w) {
                vi += 1
            } else if enCommonWords.contains(w) || isOrthographicallyNonVietnamese(w) {
                en += 1
            }
            // else: ambiguous/unknown — excluded from the ratio.
        }
        let total = vi + en
        return total == 0 ? nil : Double(vi) / Double(total)
    }

    /// Resolve `text` to a single language for direction/label decisions, mapping
    /// the undetermined case to the user's `native` language — the safe default,
    /// since undetermined text (short, ambiguous, or filler-only) reads fine as
    /// native and needs no translation.
    static func detectLanguage(_ text: String,
                               native: TranslateLanguage) -> TranslateLanguage {
        identify(text) ?? native
    }

    /// The language `text` is written in, or nil when it can't be told. The
    /// optional form, for callers that need to distinguish "it's the native
    /// language" from "no idea" — `detectLanguage` collapses both to native.
    static func sourceLanguage(of text: String) -> TranslateLanguage? { identify(text) }

    /// Predominantly-Vietnamese yes/no. Thin wrapper kept for call sites that only
    /// need a boolean; undetermined counts as not-Vietnamese.
    static func isVietnamese(_ text: String) -> Bool {
        (vietnameseRatio(text) ?? 0) >= 0.5
    }

    // MARK: Conversation language

    /// The language a conversation is actually being held in.
    ///
    /// `detectLanguage` can only ever answer Vietnamese or English — it is built on
    /// `vietnameseRatio`, a two-way word heuristic. That was the whole world when
    /// two languages shipped. With the seven Qwen3 v6 packs it silently mistargets:
    /// a Spanish thread scores 0.0 "Vietnamese" and comes back `.english`, so
    /// auto-suggest asks the model for replies *in English* to a Spanish
    /// conversation.
    ///
    /// Two stages:
    ///
    ///  1. **`LangDetect`** (see `identify`). Bundled, 111 labels, and deliberately
    ///     NOT filtered to `TranslateLanguage`: narrowing renormalizes probability
    ///     mass over the allowed set, so an Italian message would be forced onto
    ///     whichever of our eight it least dislikes. Unfiltered it can answer
    ///     "Italian", which falls through to stage 2 — the honest outcome, since
    ///     the app has no Italian pack to suggest in.
    ///  2. **The user's native language.** The fallback for undetermined text.
    ///
    /// There used to be a `vietnameseRatio` stage between them, and removing it is
    /// the point of the change rather than a simplification. `viDiacritics`
    /// contains the plain acute and grave vowels (á é í ó ú), which Spanish,
    /// French, Portuguese, Romanian and Czech use constantly, so the heuristic
    /// claimed every accented Latin-script language the previous engine declined
    /// to name: `"Olá, vejo-te amanhã no escritório"` scored 0.75 Vietnamese and
    /// was screened by the Vietnamese guardrail pack. Demoting it below the
    /// recognizer had only moved that failure to the languages the recognizer
    /// abstains on. Ablating it entirely costs 0.7pp on short text and removes
    /// five of fourteen wrong-pack routings.
    ///
    /// The latest INCOMING message is tried on its own first so a mid-thread
    /// language switch is respected, then the recent window as one block — chat
    /// turns are short ("ok", "haha") and language ID needs text to work with.
    static func conversationLanguage(_ history: [any SLMMessage],
                                     native: TranslateLanguage) -> TranslateLanguage {
        let latest = (history.last { !$0.isFromUser } ?? history.last)?.text ?? ""
        if let lang = identify(latest) { return lang }

        // Nothing conclusive in the message being replied to. Fall back to the same
        // window the prompt renders, so the answer describes the conversation the
        // model is about to read.
        let window = history.suffix(suggestWindow)
            .map(\.text)
            .joined(separator: "\n")
        return identify(window) ?? native
    }

    /// One language-ID pass over `text`, or nil when the answer isn't usable —
    /// too little text, too little confidence, or a language the app has no
    /// pack for.
    ///
    /// Runs `LangDetect` (MediaPipe's Language Detector, bundled) rather than
    /// `NLLanguageRecognizer`. Two reasons, both measured over 1,204 chat-length
    /// cases; see the type comment on `LangDetect` for the full argument:
    ///
    ///  - **It abstains instead of guessing.** The old cascade ended in
    ///    `vietnameseRatio`, a two-way Vietnamese/English word heuristic that
    ///    fired whenever the recognizer declined to name a language. Every
    ///    accented Latin-script language it had never heard of came back
    ///    Vietnamese: Portuguese, Romanian, Czech, Catalan and Indonesian all
    ///    routed to the VIETNAMESE guardrail pack, which then returns a confident
    ///    verdict about text it cannot read. Wrong-pack routings across the
    ///    corpus fell from 14 to 6.
    ///  - **It is ~20x cheaper** (0.029 ms vs 0.56 ms per message), which every
    ///    screened message pays.
    ///
    /// What it costs: on one-to-three word fragments the smaller model spreads
    /// probability across close neighbours (`"Mein Laptop"` → nl 0.45 / de 0.32)
    /// and falls below the floor more often than Apple's did. That shows up as
    /// abstention, not misrouting, and `conversationLanguage` already widens to
    /// the thread window and then to native. An abstention is recoverable; a
    /// confident wrong answer is not.
    private static func identify(_ text: String) -> TranslateLanguage? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        // Walk the ranked hypotheses rather than taking the argmax: the top answer
        // is sometimes a language we don't serve while a language we do serve sits
        // just behind it with a comparable score. Filtering the model's output to
        // our eight would renormalise the mass over that set, which is exactly how
        // a language with no pack gets forced onto the nearest one that has one.
        for (code, confidence) in LangDetect.shared.ranked(trimmed) {
            guard confidence >= identificationFloor else { break }
            if let lang = TranslateLanguage(detectorCode: code) { return lang }
        }
        return nil
    }

    /// Minimum confidence to act on a language-ID result. Below this the caller
    /// falls back to the user's native language, which is the safer wrong answer:
    /// suggestions in the language they chose in Settings, rather than in whatever
    /// a three-word message happened to score highest.
    ///
    /// 0.40, swept against the corpora behind `LanguageDetectionTests` rather than
    /// picked. It is below 0.5 on purpose: a short message often splits its
    /// probability mass across a family, and the language we serve is not always
    /// on top. `"До завтра"` scores Ukrainian 0.465 / Russian 0.429 — Ukrainian
    /// has no pack, so the loop falls through to Russian, which a 0.45 floor would
    /// have cut off.
    ///
    /// Re-swept when `LangDetect` replaced `NLLanguageRecognizer`, because a floor
    /// is only meaningful against the model that produced the scores. Across
    /// 0.30–0.40 nothing in the fixtures is misidentified and wrong-pack routings
    /// are flat, so the floor trades only against abstentions — and an abstention
    /// is handled (widen to the window, then native). Raising it past 0.50 starts
    /// costing real answers. Bare `"ok"` still abstains, correctly: Hungarian 0.45
    /// at best, and it genuinely carries no language.
    private static let identificationFloor = 0.40

    /// A token that cannot be a single Vietnamese syllable: it uses a letter
    /// absent from Vietnamese (f, j, w, z), or is longer than the longest
    /// Vietnamese syllable (7 letters, e.g. "nghieng"). Cheap, high-precision
    /// English signal for content words not listed in `enCommonWords`.
    private static func isOrthographicallyNonVietnamese(_ word: String) -> Bool {
        word.count > 7 || word.unicodeScalars.contains { nonVietnameseLetters.contains($0) }
    }

    private static let nonVietnameseLetters = Set("fjwz".unicodeScalars)

    /// Whether every letter in `word` is Latin-script — the precondition for the
    /// Vietnamese/English tests, both of which assume a Latin alphabet.
    ///
    /// The ranges are ASCII, Latin-1 Supplement, Latin Extended-A/B (Spanish and
    /// French accents), Latin Extended Additional (where Vietnamese's precomposed
    /// tone marks live), and combining diacritics, which survive if any caller
    /// skips NFC normalization. Cyrillic, Arabic and Han all fall outside, so a
    /// message in one abstains from the ratio entirely and `vietnameseRatio`
    /// returns nil — the "can't tell" answer, which every call site already
    /// handles as "assume the user's native language".
    private static func isLatinScript(_ word: String) -> Bool {
        word.unicodeScalars.allSatisfy { s in
            (0x41...0x5A).contains(s.value) || (0x61...0x7A).contains(s.value) ||
            (0xC0...0x24F).contains(s.value) || (0x1E00...0x1EFF).contains(s.value) ||
            (0x300...0x36F).contains(s.value)
        }
    }

    /// Lowercase Vietnamese-specific diacritic letters (à á ả ã ạ … đ). Callers
    /// lowercase first, so uppercase forms aren't needed. The full plain-a/e/o/u
    /// tone set is included — an earlier version dropped lowercase à/á/ả/ã/ạ, so
    /// those words went undetected.
    private static let viDiacritics: Set<Unicode.Scalar> =
        Set("àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ".unicodeScalars)

    /// Distinctive diacritic-stripped Vietnamese words — function words, pronouns,
    /// and a few ultra-common content words that rarely collide with English. A
    /// hit is strong evidence of Vietnamese even with no diacritics. Words that
    /// exist in both languages (co, la, me, no, an, ban, to, so, do…) are
    /// deliberately absent — they'd be false signals, so they stay "ambiguous".
    private static let viStrongWords: Set<String> = [
        "khong", "duoc", "minh", "toi", "chung", "nhung", "nhu", "vay", "nhe",
        "nha", "roi", "chua", "dang", "cung", "voi", "cua", "tren", "duoi",
        "trong", "nay", "kia", "dau", "sao", "hom", "gio", "biet", "muon",
        "thich", "lam", "choi", "gap", "luc", "thu", "moi", "rat", "hoi", "lai",
        "nua", "thoi", "oi", "gi", "di", "hoc", "truong", "nguoi", "viec",
        "ngay", "com", "nuoc", "ve", "den", "cho", "khi", "the", // "the" = "thế"
        "them", "qua", "vao", "boi", "tai", "cai", "chiec", "mot", "hai",
    ]

    /// Common English words. Kept to high-frequency function words plus a few
    /// content words; longer/rarer English is caught by the orthographic rule.
    /// Excludes tokens that collide with Vietnamese (see `viStrongWords` note).
    private static let enCommonWords: Set<String> = [
        "and", "or", "but", "is", "are", "am", "was", "were", "be", "been",
        "for", "with", "you", "your", "yours", "we", "he", "she", "it", "they",
        "them", "this", "that", "these", "those", "have", "has", "had", "will",
        "would", "shall", "should", "can", "could", "may", "might", "must",
        "does", "did", "not", "what", "when", "where", "why", "who", "how",
        "if", "get", "got", "let", "lets", "want", "need", "like", "hey", "yes",
        "yeah", "please", "thanks", "thank", "sorry", "see", "know", "think",
        "make", "made", "going", "come", "back", "now", "then", "here", "there",
        "all", "some", "any", "just", "very", "really", "good", "great", "nice",
        "cool", "time", "today", "tonight", "meet", "at", "of", "on", "in", "us",
        "my", "his", "her", "our", "from", "up", "out", "about", "also", "only",
    ]

    // MARK: - 3. Auto-reply suggestions

    /// Auto-reply prompt in the finetune's `auto_suggestion` format: a language +
    /// handle directive ("smart-reply options in <lang> for <handle>") followed by
    /// the recent conversation as `<Handle>: text` lines. The device owner replies
    /// next, so the handle is `userName`. Language follows the CONVERSATION, so a
    /// language switch mid-thread is respected (your reply follows what was just
    /// said). The model returns a single JSON object
    /// (`{"task":"auto_suggestion","suggestions":[{intent,text}],"refused",…}`) —
    /// parse it with `parseSuggestions`.
    /// `language` is the resolved conversation language — callers get it from
    /// `conversationLanguage` and use the SAME value to pick the LoRA pack this
    /// generation runs under, so the prompt and the adapter can't disagree. It is a
    /// parameter rather than something computed in here for exactly that reason.
    static func suggestReplies(history: [any SLMMessage],
                               friendName: String = friendName,
                               language: TranslateLanguage,
                               window: Int = suggestWindow,
                               stride: Int = suggestStride) -> String {
        return """
        Suggest a few short smart-reply options in \(language.name) for <\(userName)>, who replies next in the conversation below.

        Conversation:
        \(renderHandles(history, friendName: friendName, window: window, stride: stride))
        """
    }

    /// Render the recent history as finetune-format `<Handle>: text` lines — the
    /// device owner as `userName`, a friend/bot as its authorName (group) or
    /// `friendName` (1:1).
    /// The window advances in `historyWindowStride` blocks rather than sliding by
    /// one, so consecutive suggest prompts share a byte-identical prefix and the
    /// engine's KV cache can skip re-prefilling it. See `historyWindowStride`.
    static func renderHandles(_ messages: [any SLMMessage], friendName: String = friendName,
                              window: Int = suggestWindow,
                              stride: Int = historyWindowStride) -> String {
        messages.dropFirst(windowStart(messageCount: messages.count,
                                       window: window, stride: stride)).map { m in
            let who = m.isFromUser ? userName : (m.authorName ?? friendName)
            return "<\(who)>: \(m.text)"
        }.joined(separator: "\n")
    }

    /// Structured result of the `auto_suggestion` finetune: the reply chips plus the
    /// refusal signal the model raises for sensitive / bad-actor context.
    struct SuggestionResult {
        var suggestions: [String]
        var refused: Bool
        var reason: String?
    }

    /// Parse the finetune's `auto_suggestion` output into clean, deduped reply chips.
    /// Robust to the forms the model actually emits: the target JSON
    /// (`{"suggestions":[{"intent":…,"text":…}],…}`) whole OR token-truncated, the
    /// raw `[[INTENT]] text` block form, and bare lines — plus both refusal signals
    /// (`[[NO_SUGGEST:REASON]]` and `"refused": true`).
    static func parseSuggestions(_ raw: String) -> SuggestionResult {
        // Refusal: tag form ([[NO_SUGGEST:SENSITIVE]]) or JSON "refused": true.
        if let r = raw.range(of: #"NO_SUGGEST(?::([A-Z_]+))?"#, options: .regularExpression) {
            let reason = raw[r].split(separator: ":").dropFirst().first.map(String.init)
            return SuggestionResult(suggestions: [], refused: true, reason: reason)
        }
        var reason: String?
        if let block = firstBraceBlock(in: raw),
           let data = block.data(using: .utf8),
           let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            if obj["refused"] as? Bool == true {
                return SuggestionResult(suggestions: [], refused: true, reason: obj["reason"] as? String)
            }
            reason = obj["reason"] as? String
        }

        // Candidate texts, most-specific format first: JSON "text" fields (works on
        // valid or truncated JSON), then [[INTENT]] blocks, then bare lines.
        var chips = captured(#""text"\s*:\s*"((?:[^"\\]|\\.)*)""#, in: raw)
        if chips.isEmpty { chips = captured(#"\[\[[A-Za-z_]+\]\]\s*([^\[\n]+)"#, in: raw) }
        if chips.isEmpty { chips = clean(raw).split(separator: "\n").map(String.init) }

        var seen = Set<String>()
        var out: [String] = []
        for c in chips {
            let s = stripListMarker(clean(c))
            guard !s.isEmpty, seen.insert(s.lowercased()).inserted else { continue }
            out.append(s)
        }
        return SuggestionResult(suggestions: out, refused: false, reason: reason)
    }

    /// Drop a leading list marker — `1. `, `2) `, `- `, `• ` — from a candidate
    /// chip, leaving the text itself untouched.
    ///
    /// This was previously a `trimmingCharacters` over `-•*0123456789.")( `, which
    /// trims BOTH ends and treats every digit and period as noise wherever it sits.
    /// Any reply with a number at an edge came out mangled: "Sounds good, see you
    /// at 3:30." lost its trailing "30.", and "3:30 works for me — see you there."
    /// lost its leading "3" and final ".". A marker is only noise at the START, and
    /// only when a separator follows it — which is what the pattern requires, so a
    /// time, price or date anywhere in the chip survives.
    private static func stripListMarker(_ s: String) -> String {
        s.replacingOccurrences(of: #"^\s*(?:[-–—•*]|\d{1,2}[.)])\s+"#,
                               with: "", options: .regularExpression)
            .trimmingCharacters(in: CharacterSet(charactersIn: " \t\n\"'"))
    }

    /// Every capture-group-1 match of `pattern` in `text` (empty if the pattern is
    /// invalid or matches nothing).
    private static func captured(_ pattern: String, in text: String) -> [String] {
        guard let re = try? NSRegularExpression(pattern: pattern) else { return [] }
        let ns = text as NSString
        var out: [String] = []
        re.enumerateMatches(in: text, range: NSRange(location: 0, length: ns.length)) { m, _, _ in
            if let m, m.numberOfRanges > 1 { out.append(ns.substring(with: m.range(at: 1))) }
        }
        return out
    }

    // MARK: - 4. Summarization

    /// How many preceding messages (both sides) to prepend as CONTEXT ahead of an
    /// unread block when summarizing "what you missed", so the new messages are read
    /// in-thread rather than cold. Small on purpose — enough to anchor the topic
    /// without diluting the summary away from the unread messages themselves.
    static let unreadContextWindow = 5

    /// Summarization prompt in the finetune's format: a language directive followed
    /// by speaker-tagged lines. Speakers are labelled with their real names
    /// (`Duy:` / `Alex:`). The summary is always written in the user's `native`
    /// language (regardless of the conversation's language), so it reads in the
    /// language they chose in Settings; pass `language` to override. The model
    /// replies with `{"task":"summarization","summary":"…"}` — extract it with
    /// `parseSummary`.
    static func summarize(_ messages: [any SLMMessage],
                          friendName: String = friendName,
                          language: String? = nil,
                          native: TranslateLanguage = .vietnamese) -> String {
        let lang = language ?? native.name
        return """
        Summarize the following conversation in \(lang):

        \(renderNamed(messages, friendName: friendName))
        """
    }

    /// Summarize ONLY the new (unread) messages, using the earlier messages purely as
    /// background to interpret them — the context itself is NOT summarized. Backs the
    /// "catch up before you reply" flow, so the summary reflects what you actually
    /// missed (e.g. "Alex found a hotel deal and may book today") instead of also
    /// re-summarizing the thread you already read. When there's no prior context this
    /// collapses to the plain `summarize` of the new messages. Same JSON output —
    /// parse with `parseSummary`.
    static func summarizeUnread(context: [any SLMMessage],
                                unread: [any SLMMessage],
                                friendName: String = friendName,
                                language: String? = nil,
                                native: TranslateLanguage = .vietnamese) -> String {
        guard !context.isEmpty else {
            return summarize(unread, friendName: friendName, language: language, native: native)
        }
        let lang = language ?? native.name
        return """
        Summarize the following conversation in \(lang). Use the earlier context just to \
        understand what they refer to — do NOT summarize the context itself.

        Context:
        \(renderNamed(context, friendName: friendName))

        Conversation:
        \(renderNamed(unread, friendName: friendName))
        """
    }

    /// Render messages as `[Name]:` lines using each speaker's real name — the
    /// device owner (`userName`) for your messages, the contact for theirs.
    private static func renderNamed(_ messages: [any SLMMessage], friendName: String) -> String {
        messages.map { m in
            // In a group, each bot message carries its own author name.
            let who = m.isFromUser ? userName : (m.authorName ?? friendName)
            return "[\(who)]: \(m.text)"
        }.joined(separator: "\n")
    }

    /// Extract the `summary` field from the finetune's JSON output
    /// (`{"task":"summarization","summary":"…"}`), falling back to the cleaned raw
    /// text if the model didn't emit valid JSON. Speaker names are unwrapped from the
    /// bracketed transcript form the model echoes (`[Alex]` → `Alex`).
    static func parseSummary(_ raw: String) -> String {
        if let block = firstBraceBlock(in: raw),
           let data = block.data(using: .utf8),
           let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let s = obj["summary"] as? String,
           !s.trimmingCharacters(in: .whitespaces).isEmpty {
            return unwrapBrackets(clean(s))
        }
        return unwrapBrackets(clean(raw))
    }

    /// Progressive text for the summary UI, from a buffer that is still streaming.
    /// `parseSummary` remains the authoritative parse once generation completes.
    static func partialSummary(from raw: String) -> String? {
        partialValue(from: raw, key: "summary")
    }

    /// The same, for the translation feature's `{"translation":"…"}` output, used
    /// by `ChatService.translateLongStream`.
    static func partialTranslation(from raw: String) -> String? {
        partialValue(from: raw, key: "translation")
    }

    /// Best-effort extraction of a value the model is still writing, for
    /// token-by-token rendering.
    ///
    /// Handles BOTH shapes the finetunes actually produce, which is the whole
    /// point. This used to require the JSON key and return nil without it — but
    /// both shipping checkpoints emit plain prose for summaries, so it returned nil
    /// for every token of every stream. Nothing rendered progressively; the UI got
    /// exactly one update, the final yield. The feature still looked correct
    /// because `parseSummary` falls back to prose at the END, so only the streaming
    /// was dead, and silently. Measured: 0 partials rendered across both models.
    ///
    /// The three cases, in order:
    ///
    ///  1. **The key has arrived** — decode the JSON string value, honouring
    ///     escapes, stopping at the closing quote that may not exist yet.
    ///  2. **The buffer opens with `{`** — JSON is on its way but the key hasn't
    ///     arrived. Return nil rather than rendering `{"task":"summarizat…` as
    ///     prose for a few frames.
    ///  3. **Anything else** — prose. Render it.
    ///
    /// Case 2 is the only heuristic here, and it exists purely to stop that
    /// flicker. A model emitting prose that opens with `{` would be misread, which
    /// is not a shape any of these tasks produce.
    private static func partialValue(from raw: String, key: String) -> String? {
        guard keyRange(in: raw, key: key) != nil else {
            // Not JSON (or not yet). `clean` also suppresses an unclosed <think>
            // block, so a reasoning model shows nothing until it starts answering.
            let lead = raw.drop { $0 == " " || $0 == "\n" || $0 == "\t" || $0 == "\r" }
            if lead.first == "{" { return nil }
            let cleaned = unwrapBrackets(clean(dropUnclosedTag(raw)))
            return cleaned.isEmpty ? nil : cleaned
        }
        return partialJSONValue(from: raw, key: key)
    }

    /// Cut a `[[…` tag the model hasn't closed yet (and a lone trailing `[` that
    /// may become one). `clean` removes whole `[[…]]` tags, so without this the
    /// translation finetune's `[[LANG:English -> Vietnamese]]` preamble renders as
    /// `[[LANG:English -> H` for a few frames and then, once it closes, the text
    /// jumps back to `H`.
    private static func dropUnclosedTag(_ s: String) -> String {
        var s = s
        if let open = s.range(of: "[[", options: .backwards),
           s.range(of: "]]", range: open.upperBound..<s.endIndex) == nil {
            s = String(s[..<open.lowerBound])
        }
        if s.hasSuffix("[") { s.removeLast() }
        return s
    }

    /// Locate `"<key>"` used as a KEY — i.e. followed by `:` — not as a value.
    ///
    /// A plain substring search is wrong here, and the translation task proves it:
    /// `{"task":"translation","translation":"Hẹn gặp lại"}` contains `"translation"`
    /// twice, and the FIRST one is the value of `task`. Matching it extracts the
    /// literal word "translation" instead of the translated text. Summaries dodge
    /// this only by accident — `"summarization"` is not `"summary"` — so the naive
    /// version passed every summary test while being wrong.
    private static func keyRange(in raw: String, key: String) -> Range<String.Index>? {
        let needle = "\"\(key)\""
        var from = raw.startIndex
        while let r = raw.range(of: needle, range: from..<raw.endIndex) {
            var i = r.upperBound
            while i < raw.endIndex, raw[i] == " " || raw[i] == "\n" || raw[i] == "\t" {
                i = raw.index(after: i)
            }
            if i < raw.endIndex, raw[i] == ":" { return r }
            from = r.upperBound
        }
        return nil
    }

    /// Decode a JSON string value mid-stream: `"<key>"` … `:` … `"` … chars.
    private static func partialJSONValue(from raw: String, key: String) -> String? {
        guard let key = keyRange(in: raw, key: key) else { return nil }
        // Advance to the value's opening quote (skip the `:` and any whitespace).
        var i = key.upperBound
        while i < raw.endIndex, raw[i] != "\"" { i = raw.index(after: i) }
        guard i < raw.endIndex else { return nil }   // value quote hasn't arrived yet
        var j = raw.index(after: i)                  // first char of the value
        var out = ""
        loop: while j < raw.endIndex {
            let ch = raw[j]
            if ch == "\\" {                          // escape sequence
                let n = raw.index(after: j)
                guard n < raw.endIndex else { break }  // dangling backslash at buffer end
                switch raw[n] {
                case "n": out.append("\n")
                case "t": out.append("\t")
                case "r": out.append("\r")
                case "\"": out.append("\"")
                case "\\": out.append("\\")
                case "/": out.append("/")
                case "u":
                    // \uXXXX needs four hex digits; if they haven't all streamed in
                    // yet, stop and render what we have so far.
                    guard raw.distance(from: n, to: raw.endIndex) >= 5,
                          let hexEnd = raw.index(n, offsetBy: 5, limitedBy: raw.endIndex) else { break loop }
                    let hex = raw[raw.index(after: n)..<hexEnd]
                    if let code = UInt32(hex, radix: 16), let u = Unicode.Scalar(code) {
                        out.append(Character(u))
                    }
                    j = hexEnd
                    continue
                default: out.append(raw[n])
                }
                j = raw.index(after: n)
            } else if ch == "\"" {
                break                                // closing quote — value complete
            } else {
                out.append(ch)
                j = raw.index(after: j)
            }
        }
        let cleaned = unwrapBrackets(clean(out))
        return cleaned.isEmpty ? nil : cleaned
    }

    /// Strip the square brackets the model copies from the `[Name]:` transcript
    /// format, unwrapping `[X]` to `X` so summaries read "Alex asked…" not
    /// "[Alex] asked…". Only matched single-bracket pairs are unwrapped.
    private static func unwrapBrackets(_ s: String) -> String {
        s.replacingOccurrences(of: #"\[([^\[\]]*)\]"#, with: "$1", options: .regularExpression)
    }

    /// First brace-balanced `{ … }` block in `text`, or nil.
    private static func firstBraceBlock(in text: String) -> String? {
        guard let start = text.firstIndex(of: "{") else { return nil }
        var depth = 0
        var i = start
        while i < text.endIndex {
            let ch = text[i]
            if ch == "{" { depth += 1 }
            else if ch == "}" {
                depth -= 1
                if depth == 0 { return String(text[start...i]) }
            }
            i = text.index(after: i)
        }
        return nil
    }

    // MARK: - 4b. Per-message (long) summarization

    /// A single message this long (rough token estimate) is worth offering the
    /// per-message "Summarize" long-press action for; shorter texts never show it.
    static let longMessageTokenThreshold = 200

    /// Fallback chunk size in CHARACTERS, used only when no model is loaded and a
    /// real token count is unavailable (tests, or chunking before warm-up).
    ///
    /// Characters are a bad proxy and this is why it is the fallback: the ~4
    /// chars/token rule it was chosen against is an English figure. Vietnamese
    /// measures ~2.9, so the same text is ~38% more tokens than this assumes —
    /// under-counting, i.e. wrong in the unsafe direction. Prefer
    /// `chunk(_:maxTokens:countTokens:)`.
    static let longChunkChars = 600

    /// Target chunk size (characters) for the REDUCE pass, where already-compact
    /// partial summaries are folded together. Larger than `longChunkChars` so a
    /// handful of partials fold in ONE call, yet ~500 tokens — small enough that the
    /// fold prompt plus a 256-token summary stays under the model's ~1024 context.
    static let longReduceChars = 2000

    /// Rough token estimate used only to GATE the long-message summarize action —
    /// ~4 characters per token, but never fewer than the word count.
    static func estimatedTokens(_ text: String) -> Int {
        max(text.count / 4, text.split { $0.isWhitespace }.count)
    }

    /// Whether `text` is long enough to be worth summarizing on its own.
    static func isLongMessage(_ text: String) -> Bool {
        estimatedTokens(text) >= longMessageTokenThreshold
    }

    /// Summarize a single free-text block (one long message) in `language`.
    ///
    /// The instruction names BOTH parties — the author and the device owner —
    /// rather than calling the input "the following conversation". A long message
    /// is one person talking, and the old wording left the model to guess who the
    /// other party was; it guessed the author, producing summaries like "Mai shared
    /// her diary with Mai".
    ///
    /// The body keeps the single speaker-tagged line (`[Alex]: …`) so the finetune
    /// still sees the `[Name]:` transcript format it was trained on. `speaker` is
    /// the message author's display name. The model replies with
    /// `{"task":"summarization","summary":"…"}`; parse with `parseSummary`.
    static func summarizeText(_ text: String, language: String, speaker: String) -> String {
        """
        Summarize the message of \(speaker) talk with \(userName) in \(language):

        [\(speaker)]: \(text.trimmingCharacters(in: .whitespacesAndNewlines))
        """
    }

    /// Fold partial summaries into one, for the reduce pass of a CONVERSATION
    /// summary.
    ///
    /// Distinct from `summarizeText` on purpose. That one wraps its input in
    /// `[Speaker]: …`, which is true of a real message and false of a block of
    /// summaries — telling the model that summary prose was spoken by one person is
    /// how "Mai shared her diary with Mai" happens. These are notes about a
    /// conversation, so they are labelled as notes.
    static func combineSummaries(_ parts: [String], language: String) -> String {
        """
        Summarize these notes about a conversation in \(language):

        \(parts.joined(separator: "\n"))
        """
    }

    /// Split a long text into chunks no larger than ~`maxChars`, breaking on
    /// sentence and line boundaries so each chunk stays coherent, then greedily
    /// packing them. A run-on unit longer than the budget is hard-split on word
    /// boundaries. Drives the chunked (map-reduce) long-message summarizer in
    /// `ChatService.summarizeLong`.
    /// Split `text` into pieces that each fit `maxTokens`, measured with the
    /// MODEL'S tokenizer rather than a characters-per-token guess.
    ///
    /// Measuring matters twice over. The guess is language-dependent (~4
    /// chars/token English, ~2.9 Vietnamese), so a character limit silently means
    /// different things in different languages — and it errs toward TOO MANY
    /// tokens, which is the direction that overruns the context.
    ///
    /// Boundaries are chosen in three tiers, coarsest first, because where a chunk
    /// is cut decides how good its summary can be:
    ///
    ///  1. **Sentences** (`.` `!` `?` newline) — a whole thought per piece.
    ///  2. **Clauses** (`,` `;` `:` `—`) when one sentence is itself too long.
    ///     This tier is why it exists at all: comma-spliced prose — a diary entry,
    ///     a message typed in one breath — is a single "sentence" of many hundreds
    ///     of tokens, and without this it fell straight to tier 3 and got cut
    ///     mid-word ("đèn neon rực"). The map model then summarised a fragment and
    ///     its summary ended mid-thought, which reads as a cropped summary with no
    ///     truncation anywhere in the logs.
    ///  3. **Words** — last resort, for a single clause that still doesn't fit.
    ///
    /// `countTokens` is called once per segment, not per candidate chunk, and the
    /// packing sums those counts. Summing is very slightly optimistic (tokenizers
    /// can merge across a join), which the caller's headroom absorbs.
    static func chunk(_ text: String,
                      maxTokens: Int,
                      countTokens: (String) -> Int) -> [String] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, maxTokens > 0 else { return [] }
        if countTokens(trimmed) <= maxTokens { return [trimmed] }

        // Tier 1 -> 2 -> 3: expand any segment that is still too big.
        var segments: [String] = []
        for sentence in splitSentences(trimmed) {
            guard countTokens(sentence) > maxTokens else { segments.append(sentence); continue }
            for clause in splitClauses(sentence) {
                if countTokens(clause) > maxTokens {
                    segments.append(contentsOf: hardSplitByTokens(clause, maxTokens: maxTokens,
                                                                  countTokens: countTokens))
                } else {
                    segments.append(clause)
                }
            }
        }

        // Pack segments greedily up to the budget.
        var chunks: [String] = []
        var current = ""
        var currentTokens = 0
        for seg in segments {
            let n = countTokens(seg)
            if current.isEmpty {
                current = seg; currentTokens = n
            } else if currentTokens + n <= maxTokens {
                current += " " + seg; currentTokens += n
            } else {
                chunks.append(current); current = seg; currentTokens = n
            }
        }
        if !current.isEmpty { chunks.append(current) }
        return chunks
    }

    /// Character-based fallback, for callers with no tokenizer available.
    static func chunk(_ text: String, maxChars: Int = longChunkChars) -> [String] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }
        guard trimmed.count > maxChars else { return [trimmed] }

        var chunks: [String] = []
        var current = ""
        for piece in splitSentences(trimmed) {
            let parts = piece.count > maxChars ? splitClauses(piece) : [piece]
            for part in parts {
                if part.count > maxChars {
                    if !current.isEmpty { chunks.append(current); current = "" }
                    chunks.append(contentsOf: hardSplit(part, maxChars: maxChars))
                } else if current.isEmpty {
                    current = part
                } else if current.count + 1 + part.count <= maxChars {
                    current += " " + part
                } else {
                    chunks.append(current); current = part
                }
            }
        }
        if !current.isEmpty { chunks.append(current) }
        return chunks
    }

    /// Break a too-long sentence at clause boundaries, keeping the punctuation on
    /// the left-hand piece so the text reads unchanged when rejoined.
    private static func splitClauses(_ text: String) -> [String] {
        var out: [String] = []
        var current = ""
        for ch in text {
            current.append(ch)
            if ch == "," || ch == ";" || ch == ":" || ch == "—" || ch == "–" {
                let t = current.trimmingCharacters(in: .whitespacesAndNewlines)
                if !t.isEmpty { out.append(t) }
                current = ""
            }
        }
        let t = current.trimmingCharacters(in: .whitespacesAndNewlines)
        if !t.isEmpty { out.append(t) }
        return out.isEmpty ? [text] : out
    }

    /// Word-level split against a TOKEN budget — the last resort, when a single
    /// clause still doesn't fit.
    private static func hardSplitByTokens(_ text: String,
                                          maxTokens: Int,
                                          countTokens: (String) -> Int) -> [String] {
        var out: [String] = []
        var current = ""
        for word in text.split(separator: " ") {
            let candidate = current.isEmpty ? String(word) : current + " " + word
            if current.isEmpty || countTokens(candidate) <= maxTokens {
                current = candidate
            } else {
                out.append(current); current = String(word)
            }
        }
        if !current.isEmpty { out.append(current) }
        return out
    }

    /// Break text into sentence-ish units on `.`/`!`/`?`/newline, keeping each
    /// terminator with its sentence and dropping empty runs.
    private static func splitSentences(_ text: String) -> [String] {
        var out: [String] = []
        var current = ""
        for ch in text {
            current.append(ch)
            if ch == "." || ch == "!" || ch == "?" || ch == "\n" {
                let t = current.trimmingCharacters(in: .whitespacesAndNewlines)
                if !t.isEmpty { out.append(t) }
                current = ""
            }
        }
        let t = current.trimmingCharacters(in: .whitespacesAndNewlines)
        if !t.isEmpty { out.append(t) }
        return out
    }

    /// Hard-split an over-long unit (no usable sentence breaks) on word boundaries
    /// so no chunk exceeds `maxChars`.
    private static func hardSplit(_ text: String, maxChars: Int) -> [String] {
        var out: [String] = []
        var current = ""
        for word in text.split(separator: " ") {
            if current.isEmpty {
                current = String(word)
            } else if current.count + 1 + word.count <= maxChars {
                current += " " + word
            } else {
                out.append(current); current = String(word)
            }
        }
        if !current.isEmpty { out.append(current) }
        return out
    }

    // MARK: - 5. Event extraction

    /// Extraction prompt matching the finetune's `event_creation` task format. The
    /// `<start_of_turn>user … <end_of_turn><start_of_turn>model` wrapper is applied
    /// by `ChatLM`/`SLMEngine`, so we pass ONLY the instruction body as `user`.
    /// `language` is the user's native language, and only `event_name` and
    /// `location` follow it — those are the two fields rendered into the UI as
    /// prose, so an event detected in a Spanish thread should still read in the
    /// language the user chose in Settings. `intent`, `date` and `time` are
    /// deliberately excluded: `intent` is a machine token the parser matches on,
    /// and translating a date or time is how "3:30" becomes prose that no longer
    /// parses.
    static func extractEvent(from text: String, language: TranslateLanguage) -> String {
        """
        Extract event information from the following conversation.
        Return a JSON object with exactly these fields:
        - intent
        - event_name
        - date
        - time
        - location
        If a field is missing, return an empty string.
        Write event_name and location in \(language.name). Keep date and time exactly as written.

        Conversation:
        \(text)
        """
    }

    // MARK: - 5b. Event heuristic

    /// Cheap heuristic gate so we don't spend a task-model call on every trivial
    /// message. Returns true only when the text plausibly mentions a time or an
    /// event, in which case the caller runs `extractEvent`.
    static func mightMentionEvent(_ text: String) -> Bool {
        let lower = text.lowercased()
        // A clock time like "2h", "2:30", "3 pm", "8 giờ".
        if lower.range(of: #"\d{1,2}(:\d{2})?\s*(am|pm|h|giờ)"#,
                       options: .regularExpression) != nil {
            return true
        }
        let keywords = ["họp", "hẹn", "meeting", "lịch", "lúc", "thứ", "ngày",
                        "mai", "tomorrow", "tối", "sáng", "chiều",
                        "cuối tuần", "weekend", "@"]
        return keywords.contains { lower.contains($0) }
    }

    // MARK: - 6. Magic Compose (rewrite / smart reply)

    /// One chip in the Magic Compose menu. `instruction` is the exact string
    /// handed to the finetune as `{COMPOSE_INSTRUCTION}` — it is a MODEL
    /// CONTRACT, not UI copy, and rewording it changes output quality with no
    /// compile error. `label` is the UI copy and is free to change.
    ///
    /// The two catalogs below are the ONLY instructions the app ever sends. There
    /// is deliberately no free-typed variant: every string here is one the
    /// finetune was trained on, and an arbitrary user instruction is an untrained
    /// prompt whose failure mode is plausible-looking wrong output rather than an
    /// error. Adding one is a training question, not a UI question.
    struct ComposeInstruction: Identifiable, Equatable, Hashable {

        /// Which of the two prompt shapes this chip belongs to.
        ///
        /// The split is not cosmetic: `.rewrite` chips act on a draft the user
        /// already typed and are meaningless without one ("Make it shorter" with
        /// nothing to shorten), while `.smartReply` chips generate from the
        /// conversation alone and would fight a draft that is already there.
        /// `composeChips(hasDraft:)` picks the set; `magicCompose` decides
        /// whether to append the `Draft to rewrite:` block.
        enum Mode { case rewrite, smartReply }

        /// Stable key — persisted nowhere, but used as the `Identifiable` id and
        /// to mark the active chip, so it must not collide across the two sets.
        let id: String
        let label: String
        /// SF Symbol name. Deliberately not the emoji of the design doc: the
        /// codebase already avoids emoji in chrome because they don't render
        /// reliably across devices and the simulator (see `Avatar.initials`).
        let icon: String
        let instruction: String
        let mode: Mode
    }

    /// Mode B — the user has a draft, so every chip REWRITES it in place.
    static let rewriteChips: [ComposeInstruction] = [
        .init(id: "fix",          label: "Fix Errors",  icon: "checkmark.circle",
              instruction: "Fix spelling and diacritics",       mode: .rewrite),
        .init(id: "professional", label: "Professional", icon: "briefcase",
              instruction: "Make it more professional",         mode: .rewrite),
        .init(id: "friendly",     label: "Friendly",     icon: "face.smiling",
              instruction: "Make it friendlier",                mode: .rewrite),
        .init(id: "shorten",      label: "Shorten",      icon: "scissors",
              instruction: "Make it shorter",                   mode: .rewrite),
        .init(id: "elaborate",    label: "Elaborate",    icon: "text.append",
              instruction: "Make it longer and more detailed",  mode: .rewrite),
        .init(id: "soften",       label: "Soften",       icon: "leaf",
              instruction: "Soften it",                         mode: .rewrite),
        .init(id: "casual",       label: "Casual",       icon: "bubble.left.and.bubble.right",
              instruction: "Make it more casual",               mode: .rewrite),
    ]

    /// Mode A — the input box is empty, so every chip GENERATES a reply from the
    /// conversation. Same prompt template, minus the `Draft to rewrite:` block.
    static let smartReplyChips: [ComposeInstruction] = [
        .init(id: "agree",          label: "Agree",          icon: "hand.thumbsup",
              instruction: "Agree / accept",           mode: .smartReply),
        .init(id: "decline",        label: "Decline",        icon: "hand.thumbsdown",
              instruction: "Politely decline",         mode: .smartReply),
        .init(id: "reply-pro",      label: "Professional",   icon: "briefcase",
              instruction: "Reply professionally",     mode: .smartReply),
        .init(id: "busy",           label: "Busy",           icon: "hourglass",
              instruction: "Say I'm busy",             mode: .smartReply),
        .init(id: "get-back-later", label: "Get Back Later", icon: "clock",
              instruction: "I'll get back to you",     mode: .smartReply),
    ]

    /// The chip set for the current input-box state. Whether a draft is present is
    /// the ONLY thing that decides it — see `ComposeInstruction.Mode`.
    static func composeChips(hasDraft: Bool) -> [ComposeInstruction] {
        hasDraft ? rewriteChips : smartReplyChips
    }

    /// The language ONE Magic Compose call should run in — both the `{LANG}` in
    /// the prompt and the LoRA pack the generation runs under.
    ///
    /// Not simply `conversationLanguage`, and the difference only shows up in the
    /// rewrite mode. A rewrite acts on words the user has ALREADY written, and
    /// those words are not always in the language of the thread — drafting an
    /// English line in a Vietnamese conversation is an ordinary thing to do. Ask
    /// for that under the thread's language and "Make it shorter" comes back as a
    /// Vietnamese translation of the draft: a confident, fluent answer to a
    /// question the user did not ask, generated under the wrong pack.
    ///
    /// So the draft wins when it is identifiable. `sourceLanguage` returns nil for
    /// text too short or too ambiguous to call — "ok", "haha", a bare emoji — and
    /// those fall through to the conversation rather than guessing, which is the
    /// same place a short message lands in `conversationLanguage` itself.
    ///
    /// Smart-reply mode passes an empty draft, so it resolves through exactly the
    /// path auto-suggest takes. That is deliberate: the two features answer the
    /// same question ("what should I say next?") and disagreeing about the
    /// language would be visible to the user as one chip row contradicting the
    /// other.
    static func composeLanguage(history: [any SLMMessage],
                                draft: String,
                                native: TranslateLanguage) -> TranslateLanguage {
        if let drafted = sourceLanguage(of: draft) { return drafted }
        return conversationLanguage(history, native: native)
    }

    /// Magic Compose prompt, in the finetune's `magic_compose` format. One
    /// template serves both modes: the trailing `Draft to rewrite:` block is
    /// appended only when there is a draft, which is exactly what turns a rewrite
    /// into a from-scratch smart reply.
    ///
    /// `language` is the resolved conversation language and is passed in rather
    /// than computed here for the same reason as `suggestReplies`: the caller uses
    /// the SAME value to pick the LoRA pack the generation runs under, so the
    /// prompt and the adapter cannot disagree.
    ///
    /// The conversation is rendered with `renderHandles` — the finetune's
    /// `<Handle>: text` form, identical to the auto-suggest task — over the same
    /// short `suggestWindow`, because this is likewise a reaction to the latest
    /// message rather than a summary of the thread.
    ///
    /// The model replies with `[[REPLY]] …`, or `[[REFUSE:HARMFUL_INSTRUCTION]]`
    /// when the instruction asks for something it won't write — parse it with
    /// `parseCompose`.
    static func magicCompose(history: [any SLMMessage],
                             instruction: ComposeInstruction,
                             language: TranslateLanguage,
                             draft: String = "",
                             friendName: String = friendName,
                             user: String = userName,
                             window: Int = suggestWindow,
                             stride: Int = historyWindowStride) -> String {
        // Mirrors the reference builder's `user.strip() or "me"` fallback: an
        // empty handle would otherwise render as "for  in the conversation".
        let who = user.trimmingCharacters(in: .whitespacesAndNewlines)
        let activeUser = who.isEmpty ? "me" : who
        let conversation = renderHandles(history, friendName: friendName,
                                         window: window, stride: stride)

        var parts = [
            "Write a reply in \(language.name) for \(activeUser) in the conversation below, "
          + "following this instruction: \(instruction.instruction.trimmingCharacters(in: .whitespacesAndNewlines))",
            "Conversation:\n\(conversation.trimmingCharacters(in: .whitespacesAndNewlines))",
        ]

        let trimmedDraft = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmedDraft.isEmpty {
            parts.append("Draft to rewrite:\n\(trimmedDraft)")
        }
        return parts.joined(separator: "\n\n")
    }

    /// What one Magic Compose generation produced.
    enum ComposeOutcome: Equatable {
        /// A usable rewrite / reply, ready to preview.
        case text(String)
        /// The model declined the instruction. `reason` is the tag's payload
        /// (`HARMFUL_INSTRUCTION`), or nil for a bare `[[REFUSE]]`.
        case refused(reason: String?)
        /// The generation came back blank or was nothing but tags. Distinct from
        /// `.refused` on purpose: this one is worth retrying, that one is not.
        case empty
    }

    /// Parse a Magic Compose generation.
    ///
    /// Tag detection runs on the raw output with only `<think>` removed, BEFORE
    /// `clean` strips every `[[…]]` marker — the refusal signal IS a marker, so
    /// cleaning first would erase the one thing this has to detect. Reasoning is
    /// dropped first so a model musing about `[[REFUSE:…]]` in its scratchpad
    /// isn't read as having actually refused.
    ///
    /// A generation with no tag at all falls through to `.text`: the small models
    /// drop the `[[REPLY]]` prefix often enough that treating an untagged rewrite
    /// as a failure would throw away good output.
    static func parseCompose(_ raw: String) -> ComposeOutcome {
        let body = stripThinking(raw)

        if let r = body.range(of: #"\[\[REFUSE(?::([A-Z_]+))?\]\]"#,
                              options: .regularExpression) {
            let reason = body[r]
                .trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
                .split(separator: ":").dropFirst().first.map(String.init)
            return .refused(reason: reason)
        }

        // Drop everything up to and including the [[REPLY]] marker when it's
        // there, so a model that narrates ahead of the tag doesn't leak that
        // preamble into the input box. `clean` handles the untagged case and any
        // stray markers either way.
        var text = body
        if let r = text.range(of: #"\[\[REPLY\]\]"#, options: .regularExpression) {
            text = String(text[r.upperBound...])
        }
        let cleaned = clean(text)
        return cleaned.isEmpty ? .empty : .text(cleaned)
    }

}
