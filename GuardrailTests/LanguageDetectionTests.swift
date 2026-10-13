import XCTest
@testable import SLMChat

/// Pins the language a conversation resolves to, which is what auto-reply
/// suggestions are generated in.
///
/// This is worth a test for the same reason the guardrail is: it fails *silently*.
/// A suggestion chip in the wrong language is still a well-formed chip — the JSON
/// parses, the UI renders three buttons, nothing logs an error. The only symptom
/// is that a Spanish thread offers English replies, which no other test would
/// notice.
///
/// It also pins `Prompts.identificationFloor`, a tuned constant. The floor sits
/// below 0.5 deliberately (see its doc comment); a future "that looks too low,
/// let's make it 0.5" would break Russian short messages and nothing else.
final class LanguageDetectionTests: XCTestCase {

    private func incoming(_ text: String) -> ChatMessage {
        ChatMessage(sender: .friend, text: text)
    }

    /// Every language the catalog serves, identified from a single chat-length
    /// incoming message. `native` is deliberately the WRONG language throughout,
    /// so a result that merely echoes the fallback can't pass.
    func testIdentifiesEachSupportedLanguage() {
        let cases: [(TranslateLanguage, String)] = [
            (.vietnamese, "Tôi rất thích cái này, mai gặp nhé"),
            (.english,    "See you at the office tomorrow morning"),
            (.spanish,    "Nos vemos mañana en la oficina, ¿te parece bien?"),
            (.french,     "On se voit demain matin au bureau, ça te va ?"),
            (.russian,    "Здравствуйте, как дела? Увидимся завтра в офисе"),
            (.arabic,     "مرحبا كيف حالك اليوم؟ سنلتقي غدا في المكتب"),
            (.chinese,    "你好，我们明天在办公室见面吧"),
        ]
        for (expected, text) in cases {
            let native: TranslateLanguage = expected == .chinese ? .english : .chinese
            XCTAssertEqual(Prompts.conversationLanguage([incoming(text)], native: native),
                           expected, "failed on: \(text)")
        }
    }

    /// Vietnamese typed without diacritics — ordinary in chat.
    func testDiacriticFreeVietnamese() {
        for text in ["em khong biet dau, de anh hoi lai",
                     "Ok anh, 4 gio chieu minh gap nhe"] {
            XCTAssertEqual(Prompts.conversationLanguage([incoming(text)], native: .english),
                           .vietnamese, "failed on: \(text)")
        }
    }

    /// Latin languages that share Vietnamese's tone marks must not be captured by
    /// `vietnameseRatio`.
    ///
    /// This is a regression test for a bug that shipped in the first cut of
    /// `conversationLanguage`: `viDiacritics` contains the plain acute and grave
    /// vowels, so "¡Dale, aquí te espero! Me hace mucha ilusión que vayamos."
    /// scores 1.0 Vietnamese on `aquí` and `ilusión` alone. Running the heuristic
    /// ahead of `NLLanguageRecognizer` made a live Spanish thread suggest replies
    /// in Vietnamese — the very failure the function was written to fix. Any
    /// reordering that puts the heuristic first fails here.
    func testAccentedSpanishAndFrenchAreNotMistakenForVietnamese() {
        let cases: [(TranslateLanguage, String)] = [
            (.spanish, "¡Dale, aquí te espero! Me hace mucha ilusión que vayamos."),
            (.spanish, "¡Qué buen plan, me apunto totalmente! Dime a qué hora nos vemos."),
            (.spanish, "Hola, ¿cómo estás?"),
            (.french,  "J'ai déjà réservé une table pour huit heures"),
        ]
        for (expected, text) in cases {
            XCTAssertEqual(Prompts.conversationLanguage([incoming(text)], native: .vietnamese),
                           expected, "failed on: \(text)")
        }
    }

    /// Russian short messages are the reason the floor is 0.40: "Привет" scores
    /// Bulgarian 0.53 / Russian 0.42, so the mapping has to skip an unsupported
    /// language ahead of a supported one AND accept a sub-0.5 score.
    func testShortRussianSurvivesTheConfidenceFloor() {
        XCTAssertEqual(
            Prompts.conversationLanguage([incoming("Привет")], native: .english),
            .russian)
    }

    /// A language with no pack must NOT be claimed. Unconstrained recognition is
    /// what makes this possible — constraining to `TranslateLanguage` would force
    /// Italian onto whichever of the eight it least dislikes.
    ///
    /// German used to be one of these examples and is now the counter-example
    /// below: it gained a case when the v16 guardrail pack arrived. That move is
    /// the whole point of this pair — the boundary is "does a pack exist", and it
    /// shifts as packs ship.
    func testUnsupportedLanguageFallsBackToNative() {
        for text in ["Ciao, ci vediamo domani in ufficio",
                     "Olá, vejo-te amanhã no escritório",
                     "안녕하세요, 내일 사무실에서 봐요"] {
            XCTAssertEqual(Prompts.conversationLanguage([incoming(text)], native: .vietnamese),
                           .vietnamese, "wrongly claimed a language for: \(text)")
        }
    }

    /// German IS claimed, because it has a guardrail pack.
    ///
    /// Before `TranslateLanguage.german` existed this text resolved to the native
    /// language and the message was screened by the wrong classifier — the danger
    /// is not that German went unscreened, it is that it went screened by a model
    /// that never saw the language. Detection has to reach the pack for the pack
    /// to be worth shipping.
    func testGermanIsDetectedNowThatItHasAPack() {
        XCTAssertEqual(
            Prompts.conversationLanguage([incoming("Guten Morgen, wir sehen uns morgen im Büro")],
                                         native: .vietnamese),
            .german)
    }

    /// Text that carries no language at all resolves to native rather than to
    /// whatever a two-letter token scored highest.
    func testUndeterminedTextFallsBackToNative() {
        XCTAssertEqual(Prompts.conversationLanguage([incoming("ok")], native: .french), .french)
        XCTAssertEqual(Prompts.conversationLanguage([], native: .russian), .russian)
    }

    /// A short, unidentifiable latest message widens to the surrounding window
    /// instead of giving up — the reply still lands in the thread's language.
    func testShortMessageWidensToTheConversationWindow() {
        let history = [
            incoming("Nos vemos mañana en la oficina, ¿te parece bien?"),
            ChatMessage(sender: .you, text: "Vale"),
            incoming("ok"),
        ]
        XCTAssertEqual(Prompts.conversationLanguage(history, native: .english), .spanish)
    }

    /// The language follows the conversation, not the Settings choice: a Vietnamese
    /// user in a Russian thread gets Russian suggestions.
    func testSuggestPromptAsksForTheConversationLanguage() {
        let history = [incoming("Здравствуйте, как дела? Увидимся завтра в офисе")]
        let language = Prompts.conversationLanguage(history, native: .vietnamese)
        let prompt = Prompts.suggestReplies(history: history, language: language)
        XCTAssertTrue(prompt.contains("in Russian"), prompt)
        XCTAssertFalse(prompt.contains("in Vietnamese"), prompt)
    }

    /// A switch mid-thread is respected — the newest incoming message wins over
    /// the older ones, because that's the message being replied to.
    func testMidThreadLanguageSwitchIsRespected() {
        let history = [
            incoming("See you at the office tomorrow morning"),
            ChatMessage(sender: .you, text: "Sounds good to me"),
            incoming("On se voit demain matin au bureau, ça te va ?"),
        ]
        XCTAssertEqual(Prompts.conversationLanguage(history, native: .english), .french)
    }

    /// Translation targets the user's NATIVE language for anything foreign, and
    /// swings out to English only for text already in the native language.
    ///
    /// Before real language ID this was structurally impossible for a third
    /// language: `detectLanguage` answered only Vietnamese or English, so Spanish
    /// resolved to one of those and the direction was decided from a coin flip.
    func testTranslateTargetsTheNativeLanguage() {
        // Foreign → native, whatever the foreign language is.
        for text in ["Nos vemos mañana en la oficina",
                     "Здравствуйте, как дела?",
                     "See you at the office tomorrow morning"] {
            XCTAssertEqual(Prompts.autoTarget(for: text, native: .vietnamese), .vietnamese,
                           "failed on: \(text)")
        }
        // Native → out to English.
        XCTAssertEqual(Prompts.autoTarget(for: "Tôi rất thích cái này", native: .vietnamese),
                       .english)
        // And the same rule with a non-Vietnamese native language.
        XCTAssertEqual(Prompts.autoTarget(for: "Tôi rất thích cái này", native: .spanish),
                       .spanish)
        XCTAssertEqual(Prompts.autoTarget(for: "Nos vemos mañana en la oficina", native: .spanish),
                       .english)
    }

    /// The prompt must name the language it is actually translating FROM. The old
    /// `target.opposite` default announces a Spanish message as English.
    func testTranslatePromptNamesTheRealSourceLanguage() {
        let text = "Nos vemos mañana en la oficina"
        let prompt = Prompts.translate(text,
                                       from: Prompts.sourceLanguage(of: text),
                                       to: .vietnamese)
        XCTAssertTrue(prompt.contains("Spanish sentence"), prompt)
        XCTAssertFalse(prompt.contains("English sentence"), prompt)
    }

    /// Summaries and event extraction stay on the user's own language regardless of
    /// what the conversation is in — their output is read, not sent.
    func testSummaryAndEventTargetNativeNotConversation() {
        let spanish = [incoming("Nos vemos mañana en la oficina a las cuatro")]
        XCTAssertTrue(Prompts.summarize(spanish, native: .vietnamese).contains("in Vietnamese"))
        let event = Prompts.extractEvent(from: "Nos vemos mañana a las cuatro", language: .vietnamese)
        XCTAssertTrue(event.contains("in Vietnamese"), event)
        // Dates must survive untranslated or the parser loses them.
        XCTAssertTrue(event.contains("Keep date and time exactly as written"), event)
    }

    /// What the Translate control keys off. `MessageBubble.isNativeLanguage` hides
    /// the button when a bubble is already in the user's language, and it reads
    /// `ChatMessage.detectedLanguage` to decide.
    ///
    /// Regression test: that predicate used to read `vietnameseRatio` alone, which
    /// abstains (nil) on non-Latin scripts — and nil is treated as "native, hide
    /// the button". So Translate disappeared on exactly the Chinese, Russian and
    /// Arabic messages that needed it most. The identity check has to answer for
    /// those, and it has to answer nil for text carrying no language so a
    /// half-streamed bubble doesn't flash the control.
    func testDetectedLanguageDrivesTheTranslateControl() {
        XCTAssertEqual(incoming("你好，我们明天在办公室见面吧").detectedLanguage, .chinese)
        XCTAssertEqual(incoming("Здравствуйте, как дела?").detectedLanguage, .russian)
        XCTAssertEqual(incoming("مرحبا كيف حالك اليوم").detectedLanguage, .arabic)
        XCTAssertEqual(incoming("Tôi rất thích cái này").detectedLanguage, .vietnamese)
        XCTAssertNil(incoming("ok").detectedLanguage)
        XCTAssertNil(incoming("").detectedLanguage)
    }

    /// The cache must follow `text`, which is rewritten token-by-token while a
    /// reply streams — a stale label would freeze the control in whatever state
    /// the first token implied.
    func testDetectedLanguageCacheInvalidatesOnTextChange() {
        let m = incoming("See you at the office tomorrow morning")
        XCTAssertEqual(m.detectedLanguage, .english)
        m.text = "你好，我们明天在办公室见面吧"
        XCTAssertEqual(m.detectedLanguage, .chinese)
    }

    /// Non-Latin scripts must abstain from the Vietnamese/English word heuristic.
    /// Without this, `isOrthographicallyNonVietnamese`'s "longer than 7 letters"
    /// rule votes English for ordinary Russian words and for a whole Chinese
    /// sentence (Han text has no spaces, so it tokenizes as one long word), and
    /// the Translate action offers to translate Russian into Russian.
    func testNonLatinScriptsAbstainFromTheVietnameseHeuristic() {
        XCTAssertNil(Prompts.vietnameseRatio("Здравствуйте, как дела?"))
        XCTAssertNil(Prompts.vietnameseRatio("你好，我们明天在办公室见面吧"))
        XCTAssertNil(Prompts.vietnameseRatio("مرحبا كيف حالك اليوم"))
        // Latin-script behaviour is unchanged: still classifiable, still on the
        // right side of the 0.5 threshold every caller compares against.
        //
        // Not asserted as exactly 0.0 for English — "the" is in `viStrongWords`
        // on purpose (it is also Vietnamese "thế"), so an English sentence
        // containing it scores a nonzero minority. That is the heuristic working
        // as designed, and pinning the exact fraction would just make this test
        // break every time a word moves between the lists.
        XCTAssertEqual(Prompts.vietnameseRatio("Tôi rất thích cái này") ?? 0, 1.0, accuracy: 0.001)
        XCTAssertLessThan(Prompts.vietnameseRatio("See you at the office") ?? 1, 0.5)
    }

    // MARK: - Accented Latin languages the app has no pack for

    /// The regression that motivated replacing `NLLanguageRecognizer` with
    /// `LangDetect`, and the reason `identify`'s `vietnameseRatio` stage is gone.
    ///
    /// `viDiacritics` contains the plain acute and grave vowels, so the heuristic
    /// scored ordinary Portuguese, Romanian, Czech, Catalan and Indonesian as
    /// majority-Vietnamese. It only ran when the recognizer had abstained, which
    /// is exactly what the recognizer does on a short message in a language we
    /// never ask it about — so these five came back `.vietnamese` and were
    /// screened by the VIETNAMESE guardrail pack. That is not a missed detection:
    /// the classifier returns a confident verdict about text it cannot read, and
    /// nothing logs.
    ///
    /// Measured ratios at the time, all >= the 0.5 the stage compared against:
    /// pt 0.75, ro 1.00, cs 0.75, id 0.50, ca 0.50.
    func testAccentedLatinLanguagesWithNoPackAreNotClaimedAsVietnamese() {
        for text in ["Olá, vejo-te amanhã no escritório",
                     "Bună, ne vedem mâine la birou",
                     "Ahoj, uvidíme se zítra v kanceláři",
                     "Halo, sampai jumpa besok di kantor",
                     "Hola, ens veiem demà a l\'oficina",
                     "Ciao, ci vediamo domani in ufficio"] {
            XCTAssertNil(Prompts.sourceLanguage(of: text),
                         "wrongly claimed a language for: \(text)")
            XCTAssertEqual(Prompts.conversationLanguage([incoming(text)], native: .english),
                           .english, "did not fall back to native for: \(text)")
        }
    }

    /// The heuristic itself still scores these as Vietnamese — that is WHY it must
    /// not sit on the identification path.
    ///
    /// Kept as an explicit pin rather than deleted with the stage: `vietnameseRatio`
    /// is still used by the Translate control, where the question really is
    /// "Vietnamese or English" about text already known to be one of the two. If
    /// someone reintroduces it as a fallback in `identify`, this documents what
    /// they would be buying.
    func testVietnameseRatioStillMisreadsThemWhichIsWhyItIsNotOnThePath() {
        XCTAssertGreaterThanOrEqual(Prompts.vietnameseRatio("Olá, vejo-te amanhã no escritório") ?? 0, 0.5)
        XCTAssertGreaterThanOrEqual(Prompts.vietnameseRatio("Bună, ne vedem mâine la birou") ?? 0, 0.5)
    }

    /// Romanised text must not be routed to a native-script pack.
    ///
    /// The detector emits `zh-Latn`, `ru-Latn` and `ar-Latn` for pinyin and
    /// romanised Cyrillic/Arabic. `TranslateLanguage(detectorCode:)` deliberately
    /// leaves them unmapped: the guardrail and LoRA packs were trained on native
    /// script, so claiming them would screen romanised text with a model that
    /// never saw the form it is written in.
    func testRomanisedLabelsAreNotMappedToNativeScriptPacks() {
        XCTAssertNil(TranslateLanguage(detectorCode: "zh-Latn"))
        XCTAssertNil(TranslateLanguage(detectorCode: "ru-Latn"))
        XCTAssertNil(TranslateLanguage(detectorCode: "ar-Latn"))
        XCTAssertNil(TranslateLanguage(detectorCode: "unknown"))
        // The eight that DO map are the eight GuardPackCatalog ships.
        let mapped = ["vi", "en", "ar", "es", "fr", "ru", "zh", "de"]
            .compactMap { TranslateLanguage(detectorCode: $0) }
        XCTAssertEqual(Set(mapped), Set(GuardPackCatalog.supportedLanguages))
    }

    /// A missing weights blob must degrade to "can't tell", never crash — the app
    /// ships the .bin via bootstrap.sh, so a fresh checkout genuinely lacks it.
    func testDetectorReportsWhetherItLoaded() {
        if !LangDetect.shared.isLoaded {
            XCTAssertNil(Prompts.sourceLanguage(of: "Tôi rất thích cái này"),
                         "an unloaded detector must abstain, not guess")
            XCTAssertEqual(Prompts.conversationLanguage([incoming("Tôi rất thích cái này")],
                                                        native: .english), .english)
        }
    }
}
