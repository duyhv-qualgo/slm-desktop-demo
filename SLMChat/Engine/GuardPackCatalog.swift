import Foundation

/// One language's guardrail assets: a classifier and the tokenizer it was
/// trained with.
///
/// The tokenizer travels WITH the model rather than being shared, because a
/// classifier is only meaningful over the vocabulary it was trained on — pairing
/// a model with someone else's tokenizer produces token ids that mean nothing to
/// it, and the result is confident nonsense rather than an error.
struct GuardPack {
    let language: TranslateLanguage
    let modelResource: String
    let tokenizerResource: String
}

/// Which guardrail pack screens which language.
///
/// Mirrors `LoRACatalog`: assets are per-language, resolved by lookup rather
/// than hardcoded at the call site, so adding a language is a row here.
///
/// A language absent from this table is NOT screened — `GuardEngine` skips it
/// rather than handing the text to a classifier that can't read it. Running the
/// Vietnamese model over, say, Thai would not fail; softmax always sums to 1 and
/// argmax always returns something, so it would return a confident verdict about
/// text it never understood. Skipping is the honest outcome.
enum GuardPackCatalog {

    /// Release `2026_08_25_v16_multilingual_label_smooth`: eight genuinely
    /// distinct checkpoints, one per language, each with the tokenizer it was
    /// trained against (vocab 12000, seq 128).
    ///
    /// The model files are named `v16ls`, the tokenizers `v16`, and the mismatch
    /// is deliberate. Label smoothing retrained the CLASSIFIERS on the same data,
    /// config and architecture; the vocabularies did not move, and all eight
    /// tokenizer files are byte-identical to the ones the previous release
    /// shipped. Renaming them would have implied a change that did not happen —
    /// and a pack that pairs a v16ls model with a v16 tokenizer is not a
    /// mismatch, it is the accurate description of what was rebuilt.
    ///
    /// The previous release shipped ONE bilingual checkpoint under two names —
    /// `vi` and `en` were byte-identical (same SHA-256), so a verdict could never
    /// tell you which pack ran and tests asserting on pack selection had to
    /// observe the load rather than the output. That is no longer true: all eight
    /// files hash differently, and a test may now assert that the right pack
    /// produced a verdict.
    ///
    /// German is here, and it is the one language screened WITHOUT the rest of
    /// the app supporting it. `LoRACatalog` has no German adapter, so
    /// `isSupported(forBaseModel:)` is false for it and Settings never offers it
    /// as a native language — no translation, no reply suggestions.
    ///
    /// Screening does not need any of that. The guardrail's only requirement is
    /// a pack for the language a message is written in, and refusing to route
    /// German did not make it unscreened: it made it screened by the ENGLISH
    /// pack, because `identify()` skipped the hypothesis it couldn't spell and
    /// took the next one. Since v16 the packs are genuinely different models, so
    /// that was a real misclassification rather than the harmless no-op it was
    /// when `vi` and `en` were the same file.
    static let packs: [GuardPack] = [
        GuardPack(language: .vietnamese, modelResource: "guardrail-v16ls-vi",
                  tokenizerResource: "guardrail-v16-tokenizer-vi"),
        GuardPack(language: .english,    modelResource: "guardrail-v16ls-en",
                  tokenizerResource: "guardrail-v16-tokenizer-en"),
        GuardPack(language: .arabic,     modelResource: "guardrail-v16ls-ar",
                  tokenizerResource: "guardrail-v16-tokenizer-ar"),
        GuardPack(language: .spanish,    modelResource: "guardrail-v16ls-es",
                  tokenizerResource: "guardrail-v16-tokenizer-es"),
        GuardPack(language: .french,     modelResource: "guardrail-v16ls-fr",
                  tokenizerResource: "guardrail-v16-tokenizer-fr"),
        GuardPack(language: .russian,    modelResource: "guardrail-v16ls-ru",
                  tokenizerResource: "guardrail-v16-tokenizer-ru"),
        GuardPack(language: .chinese,    modelResource: "guardrail-v16ls-zh",
                  tokenizerResource: "guardrail-v16-tokenizer-zh"),
        GuardPack(language: .german,     modelResource: "guardrail-v16ls-de",
                  tokenizerResource: "guardrail-v16-tokenizer-de"),
    ]

    /// The pack that screens `language`, or nil when none does.
    static func pack(for language: TranslateLanguage) -> GuardPack? {
        packs.first { $0.language == language }
    }

    /// Languages the guardrail can screen — the Settings sheet reports these so
    /// "which languages are actually covered" is visible rather than assumed.
    static var supportedLanguages: [TranslateLanguage] { packs.map(\.language) }
}
