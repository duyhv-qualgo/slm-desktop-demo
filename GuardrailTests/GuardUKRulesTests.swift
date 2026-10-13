import XCTest
@testable import SLMChat

/// Checks the UK rule set (`guardrails/regions/uk.yaml`), written against the
/// Online Safety Act 2023 and Ofcom's Illegal Content Judgements Guidance.
///
/// The Vietnamese set arrived precision-tuned from upstream. This one did not —
/// it was authored here, in English, against a language the app's users actually
/// write in, which makes FALSE POSITIVES the dominant risk rather than a
/// footnote. Roughly half the tests below exist to pin things that must NOT be
/// flagged; treat a failure there as more serious than a missed detection, since
/// a guardrail that blurs ordinary conversation is one users learn to ignore.
final class GuardUKRulesTests: XCTestCase {

    private func ukRules() throws -> GuardRules {
        let url = try XCTUnwrap(ResourceLookup.url(GuardRegion.uk.rulesResource(for: .english),
                                                   ext: "json"),
                                "guardrail-rules-uk-en.json missing from the bundle")
        return try XCTUnwrap(GuardRules(jsonURL: url), "UK GuardRules failed to load")
    }

    // MARK: - Loading

    /// Every UK pattern must survive ICU. `GuardRules.init` SKIPS one that
    /// doesn't, so the failure mode is lost coverage with no error anywhere —
    /// and the build script only ever proved they compile under Python's `re`.
    func testAllUKPatternsCompile() throws {
        let rules = try ukRules()
        XCTAssertGreaterThanOrEqual(rules.count, 120,
                                    "expected the full UK set — a lower count means "
                                  + "patterns failed to compile under ICU and were skipped")
    }

    /// Every category the generator emits, for every region, must have an entry
    /// in `GuardRules.labels`.
    ///
    /// An unmapped category is logged and dropped, so adding `scam_fraud` to the
    /// YAML without adding it to the Swift map would ship a file with 31 fraud
    /// patterns in it and screen for none of them. Nothing else notices.
    func testEveryBuiltCategoryIsMapped() throws {
        struct Doc: Decodable {
            struct Category: Decodable { let category: String; let patterns: [Pattern] }
            struct Pattern: Decodable { let re: String; let severity: Int }
            let categories: [Category]
        }

        for region in GuardRegion.allCases {
            for language in region.ruleLanguages {
            let url = try XCTUnwrap(ResourceLookup.url(region.rulesResource(for: language),
                                                       ext: "json"))
            let doc = try JSONDecoder().decode(Doc.self, from: Data(contentsOf: url))
            let rules = try XCTUnwrap(GuardRules(jsonURL: url))

            let declared = doc.categories.reduce(0) { $0 + $1.patterns.count }
            XCTAssertEqual(rules.count, declared,
                           "\(region.displayName)/\(language.name): file declares "
                         + "\(declared) patterns but \(rules.count) loaded — an unmapped "
                         + "category in GuardRules.labels, or a pattern ICU rejected")
            }
        }
    }

    // MARK: - The policy guarantee

    /// Political speech is protected in the UK. Foreign interference is a
    /// priority offence, but no pattern can separate it from ordinary criticism
    /// of a government, so the UK set carries none — deliberately.
    ///
    /// This is the mirror of `testPoliticalRulesDoNotLeakOutOfVietnam`: that one
    /// stops Vietnam's rules spreading, this one stops anyone writing new ones.
    func testUKHasNoPoliticalRules() throws {
        let rules = try ukRules()
        for text in [
            "The government are liars and this party should be voted out",
            "The Prime Minister is a disgrace and should resign immediately",
            "Abolish the monarchy",
            "This policy is authoritarian and I'll be protesting on Saturday",
        ] {
            XCTAssertNil(rules.firstMatch(in: text),
                         "UK must not flag protected political speech: \(text.debugDescription)")
        }
    }

    // MARK: - Detection, by priority harm

    /// One representative message per priority harm the rules claim to cover.
    /// Deliberately thin — the point is that each category is wired end to end,
    /// not that these specific strings are the interesting cases.
    func testUKCatchesPriorityHarms() throws {
        let rules = try ukRules()
        let cases: [(String, SafetyLabel)] = [
            ("I'm going to kill you when I see you",                       .violence),
            ("you should just kill yourself, nobody likes you",            .suicide),
            ("pay me £500 or I'll send your nudes to all your contacts",   .sexual),
            ("gas all the jews",                                           .hate),
            ("I've been watching you every day this week",                 .harassment),
            ("selling weed, £30 a gram, delivery tonight",                 .illegal),
            ("zombie knife for sale £80, DM me",                           .illegal),
            ("HMRC: you are due a tax refund of £284, claim here https://t.co/x", .scam),
            ("Move your money to a safe account straight away",            .scam),
            ("just use your bank account and you keep £300 commission",    .scam),
            ("send me 0.5 btc and I'll double it",                         .scam),
            ("Mum I lost my phone, this is my new number, can you transfer me some money", .scam),
        ]
        for (text, expected) in cases {
            let hit = rules.firstMatch(in: text)
            XCTAssertEqual(hit?.label, expected,
                           "\(text.debugDescription) → \(hit?.label.displayName ?? "no match")")
        }
    }

    // MARK: - False positives

    /// Ordinary British chat that the patterns come close to.
    ///
    /// Every line here was written against a specific rule that could plausibly
    /// fire on it — drug nouns without supply context, delivery complaints
    /// without a payment link, threats without a target. If one of these starts
    /// matching, the rule that caught it is too loose, and the fix belongs in the
    /// YAML rather than in this list.
    func testBenignUKEnglishIsNotFlagged() throws {
        let rules = try ukRules()
        let benign = [
            // Drug nouns with no supply context.
            "grabbed a coke and a meal deal from Tesco",
            "the grass in the garden needs cutting",
            "my gear is still in the car from football",
            // Delivery and money talk that is not a scam.
            "Royal Mail are on strike again so the parcel is late",
            "my parcel is on hold at the depot, so annoying",
            "can you transfer me a tenner for the taxi",
            "I got a new number, will text you from it",
            // Violence vocabulary with no target.
            "this heat is killing me",
            "I could murder a curry right now",
            "he got absolutely battered in the second half",
            // First-person distress — see testFirstPersonDistressIsNotFlagged.
            "I'm so stressed I could scream",
            // Harassment vocabulary in ordinary use.
            "I've been thinking about you all day",
            "can you send me your address for the invite",
            "see you at work tomorrow",
            // Sexual/relationship talk between adults that is not abuse.
            "you looked gorgeous last night",
            // Hate-adjacent but discussing, not stirring.
            "we did the Holocaust in history this term",
            "the knife crime documentary was really grim",
        ]
        for text in benign {
            let hit = rules.firstMatch(in: text)
            XCTAssertNil(hit, "false positive on \(text.debugDescription) — matched "
                            + "\(hit?.label.displayName ?? "") via \(hit?.pattern ?? "")")
        }
    }

    /// The safety property that matters most in this whole file.
    ///
    /// Ofcom's children's codes want a service to SURFACE crisis support, not
    /// suppress the request for it. Someone saying they want to die must reach
    /// the person they are talking to — flagging that as self-harm content would
    /// blur a cry for help behind a warning. Every `self_harm` pattern is
    /// therefore anchored on a second-person target or on instructional
    /// vocabulary, and this test is what stops the next one being added carelessly.
    func testFirstPersonDistressIsNotFlagged() throws {
        let rules = try ukRules()
        for text in [
            "I want to kill myself",
            "I've been thinking about killing myself",
            "I don't want to be here anymore",
            "I hate myself and I want to die",
            "sometimes I think about ending it all",
            "this queue is making me want to kill myself",
            "I self harmed again last night and I feel awful",
        ] {
            let hit = rules.firstMatch(in: text)
            XCTAssertNil(hit, "distress must not be flagged: \(text.debugDescription) "
                            + "matched \(hit?.pattern ?? "")")
        }
    }

    // MARK: - Per-pattern severity

    /// Severity comes from the pattern, not the category ceiling.
    ///
    /// The name of a banned weapon is evidence that labels; an offer to sell one
    /// is proof that blurs. Both are `illegal`, and before per-pattern
    /// severity both landed on Critical — which meant a message about a news
    /// story got hidden.
    func testWeaponNameLabelsButSaleBlurs() throws {
        let rules = try ukRules()

        let mention = try XCTUnwrap(rules.firstMatch(in: "they're banning zombie knives finally"))
        XCTAssertEqual(mention.label, .illegal)
        XCTAssertEqual(mention.severity, .caution, "a bare mention must not blur")

        let sale = try XCTUnwrap(rules.firstMatch(in: "zombie knife for sale £80, DM me"))
        XCTAssertEqual(sale.severity, .critical, "an offer to sell must blur")
    }

    /// A slur with common reclaimed in-group use labels rather than blurs, while
    /// one with no such use does not soften. Ofcom's ICJG requires context to be
    /// weighed and a regex cannot weigh it, so it defers instead of guessing.
    func testReclaimedSlurLabelsWithoutBlurring() throws {
        let rules = try ukRules()

        let reclaimed = try XCTUnwrap(rules.firstMatch(in: "my nigga where you been"))
        XCTAssertEqual(reclaimed.label, .hate)
        XCTAssertEqual(reclaimed.severity, .caution)

        let slur = try XCTUnwrap(rules.firstMatch(in: "go home you paki"))
        XCTAssertEqual(slur.severity, .critical)
    }

    /// Severity, not file position, picks the winner.
    ///
    /// `firstMatch` used to return the first hit in category order, so a
    /// `caution` rule in an earlier category could mask a `critical` one later.
    /// Here the violence `caution` ("watch your back") sits ahead of the
    /// scam/fraud `critical`, and the serious verdict must still come back.
    ///
    /// This used to use "I know where you live", which was removed from the set
    /// as too strict — a punchline between friends far more often than a threat.
    /// Any violence `caution` demonstrates the ordering; the point is the
    /// category position, not the words.
    func testMostSevereMatchWins() throws {
        let rules = try ukRules()
        let text = "Watch your back. Move your money to a safe account now."

        let hit = try XCTUnwrap(rules.firstMatch(in: text))
        XCTAssertEqual(hit.severity, .critical)
        XCTAssertEqual(hit.label, .scam,
                       "an earlier low-severity category must not mask a later critical one")
    }

    /// The UK patterns must not be able to stall the guard queue.
    ///
    /// This set introduced a construct the Vietnamese one never used: bounded
    /// gaps like `[\s\S]{0,140}` joining two halves of a scam script. Those
    /// backtrack in a way single-token patterns do not, and `scanLimit` lets a
    /// pasted wall of text be 20k characters. The worst input is one that nearly
    /// matches over and over — repeated bait words with the second half never
    /// arriving.
    func testUKScanStaysBoundedOnAdversarialInput() throws {
        let rules = try ukRules()
        let bait = String(repeating: "your parcel delivery account safe transfer money ",
                          count: 1_000)   // ~49k chars, past scanLimit

        let start = Date()
        _ = rules.firstMatch(in: bait)
        let elapsed = Date().timeIntervalSince(start)
        XCTAssertLessThan(elapsed, 2.0,
                          "scan took \(elapsed)s — a bounded-gap pattern is backtracking")
    }

    // MARK: - Engine integration

    /// The verdict the app actually shows, through the whole engine rather than
    /// the rule object — this is the path that would break if `GuardEngine`
    /// went back to applying the category ceiling.
    func testEngineCarriesPatternSeverity() async throws {
        let engine = GuardEngine()
        let res = await engine.warmUp(language: .english, region: .uk)
        XCTAssertTrue(res.ok, "warm-up failed: \(res.error ?? "unknown")")
        XCTAssertGreaterThan(engine.ruleCount, 0)
        XCTAssertTrue(engine.rulesAuthored, "the UK set is authored")

        let mention = await engine.classify("they're banning zombie knives finally",
                                            language: .english)
        XCTAssertEqual(mention.severity, .caution)

        let scam = await engine.classify("Move your money to a safe account straight away",
                                         language: .english)
        XCTAssertEqual(scam.label, .scam)
        XCTAssertEqual(scam.severity, .critical)
    }
}
