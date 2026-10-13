import XCTest
@testable import SLMChat

/// End-to-end check of the on-device guardrail: text → tokenizer → BERT-Tiny
/// (TFLite) → `SafetyResult`, compared against `reference_verdicts.json`.
///
/// Where `GuardTokenizerTests` pins one stage, this pins the whole path — it
/// would catch a wrong tensor layout, a mixed-up output head, or a changed
/// argmax/threshold in `GuardEngine.decide`, none of which the tokenizer test
/// can see.
final class GuardEngineTests: XCTestCase {

    private struct Verdict: Decodable {
        /// Which pack must classify this text. Fixtures are per-pack since v16:
        /// the packs are genuinely monolingual, so a text is only meaningful
        /// against the one it was generated for. Under the previous release
        /// `vi` and `en` were byte-identical and this field would have been
        /// redundant.
        let lang: String
        let text: String
        let classID: Int
        let unsafe: Bool
        let pUnsafe: Float

        enum CodingKeys: String, CodingKey {
            case lang, text, unsafe
            case classID = "class_id"
            case pUnsafe = "p_unsafe"
        }

        var language: TranslateLanguage? {
            switch lang {
            case "vi": return .vietnamese
            case "en": return .english
            default:   return nil
            }
        }
    }

    /// Confidences come from a float model through a different runtime than the
    /// one that produced the fixtures, so they are compared with tolerance. The
    /// CLASS is categorical and must match exactly — that is the contract; the
    /// confidence is a sanity bound on top.
    private static let confidenceTolerance: Float = 0.05

    private func loadVerdicts() throws -> [Verdict] {
        let bundle = Bundle(for: Self.self)
        let url = try XCTUnwrap(bundle.url(forResource: "reference_verdicts",
                                           withExtension: "json"),
                                "reference_verdicts.json missing from the test bundle")
        return try JSONDecoder().decode([Verdict].self, from: Data(contentsOf: url))
    }

    /// Loaded WITHOUT rules, and every assertion below goes through
    /// `classifyModelOnly`. These fixtures are reference outputs of the original
    /// Python model, so the regex pre-filter must not stand in front of them —
    /// two of the twelve texts do match a pattern, and a rule verdict carries the
    /// class's ceiling severity and confidence 1, not the model's numbers.
    /// `GuardRulesTests` covers the pattern half.
    private func makeEngine(language: TranslateLanguage = .vietnamese) async throws -> GuardEngine {
        let engine = GuardEngine()
        let res = await engine.warmUp(language: language, region: nil)
        // The app treats a failed guardrail load as non-fatal (fails open to
        // .clean). A test must NOT: silently passing on a dead model is exactly
        // the regression this suite exists to catch.
        XCTAssertTrue(res.ok, "guardrail failed to load: \(res.error ?? "unknown")")
        return engine
    }

    func testClassifyMatchesReferenceVerdicts() async throws {
        let engine = try await makeEngine()
        let verdicts = try loadVerdicts()
        XCTAssertFalse(verdicts.isEmpty)
        // Both packs must be represented, or a regression in one of them passes
        // unnoticed because the fixtures only ever exercised the other.
        XCTAssertTrue(Set(verdicts.map(\.lang)).isSuperset(of: ["vi", "en"]),
                      "fixtures must cover both packs — regenerate with "
                    + "scripts/build_guard_fixtures.py")

        for expected in verdicts {
            let language = try XCTUnwrap(expected.language,
                                         "unknown fixture lang \(expected.lang)")
            let got = await engine.classifyModelOnly(expected.text, language: language)
            let what = "\(expected.lang) \(expected.text.debugDescription)"

            XCTAssertEqual(got.label.rawValue, expected.classID,
                           "class differs for \(what) — got \(got.label.displayName)")
            XCTAssertEqual(got.isUnsafe, expected.unsafe, "unsafe flag differs for \(what)")
            XCTAssertEqual(got.confidence, expected.pUnsafe,
                           accuracy: Self.confidenceTolerance,
                           "p_unsafe drifted for \(what)")
            // v16 is single-head, so a FLAGGED model verdict has no severity to
            // report. A cleared one does: `.safe` there is a real finding — the
            // model assessed the text and put it in class 0 — and the distinction
            // is what keeps "screened, harmless" from looking like "unassessed".
            //
            // Asserted per fixture rather than once, because the way this would
            // regress is a well-meaning "default it to .safe" that only shows up
            // on the flagged path.
            if expected.unsafe {
                XCTAssertNil(got.severity,
                             "a flagged model verdict must not invent a severity — \(what)")
            } else {
                XCTAssertEqual(got.severity, .safe,
                               "a cleared model verdict should record that it WAS "
                             + "assessed — \(what)")
            }
        }
    }

    /// `classify` is called on every message, so an empty or whitespace-only one
    /// must return cleanly rather than trip the fixed-shape tensor copy.
    func testClassifyHandlesEmptyInput() async throws {
        let engine = try await makeEngine()
        for text in ["", "   ", "\n"] {
            let result = await engine.classifyModelOnly(text, language: .vietnamese)
            XCTAssertFalse(result.isUnsafe, "empty input must not be flagged")
        }
    }
}
