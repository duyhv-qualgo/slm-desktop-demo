import XCTest
@testable import SLMChat

/// Checks the regex pre-filter: `Resources/guardrail-rules-<region>.json` →
/// `GuardRules`, for every region in `GuardRegion`.
///
/// The patterns are an upstream artifact written for Python's `re`, and the app
/// runs them through ICU (`NSRegularExpression`). The two engines agree on this
/// set today — that was verified pattern-by-pattern against Python before the
/// port — but nothing in the build would notice if a re-generated file
/// introduced a construct ICU rejects. A pattern that fails to compile is
/// SKIPPED at load by design, so the failure mode is silent lost coverage, which
/// is exactly what `testAllPatternsCompile` exists to catch.
///
/// `reference_regex.json` pins the expected category per text, generated from
/// the same Python evaluation.
final class GuardRulesTests: XCTestCase {

    private struct Case: Decodable {
        let text: String
        /// Upstream category name, or nil when nothing should match.
        let category: String?
    }

    /// Every pattern in the shipped file. Kept as a hard number so a truncated or
    /// partially-parsed file fails loudly rather than quietly screening less.
    private static let expectedPatternCount = 204

    /// A region's rule set for one of its languages.
    ///
    /// Rule sets are keyed by (region, language) — see `GuardRegion.ruleLanguages`.
    /// Defaulting to the region's FIRST declared language gives each region the
    /// set it was primarily authored in, and a region with none (Germany, which
    /// has no `TranslateLanguage` case) resolves to the universal fallback, which
    /// is exactly what the app would load for it.
    private func makeRules(_ region: GuardRegion = .vietnam,
                           language: TranslateLanguage? = nil) throws -> GuardRules {
        let name: String
        if let language, region.ruleLanguages.contains(language) {
            name = region.rulesResource(for: language)
        } else if let primary = region.ruleLanguages.first {
            name = region.rulesResource(for: primary)
        } else {
            name = GuardRegion.universalRulesResource
        }
        let url = try XCTUnwrap(ResourceLookup.url(name, ext: "json"),
                                "\(name).json missing from the bundle")
        return try XCTUnwrap(GuardRules(jsonURL: url), "GuardRules failed to load")
    }

    private func loadCases() throws -> [Case] {
        let url = try XCTUnwrap(Bundle(for: Self.self).url(forResource: "reference_regex",
                                                           withExtension: "json"),
                                "reference_regex.json missing from the test bundle")
        return try JSONDecoder().decode([Case].self, from: Data(contentsOf: url))
    }

    /// Guards the silent-skip path in `GuardRules.init`.
    func testAllPatternsCompile() throws {
        let rules = try makeRules()
        XCTAssertEqual(rules.count, Self.expectedPatternCount,
                       "expected \(Self.expectedPatternCount) usable patterns — a lower "
                     + "number means some failed to compile under ICU and were skipped")
    }

    /// Category-for-category agreement with the Python reference evaluation.
    func testMatchesReferenceCategories() throws {
        let rules = try makeRules()
        let cases = try loadCases()
        XCTAssertFalse(cases.isEmpty)

        for c in cases {
            let hit = rules.firstMatch(in: c.text)
            switch c.category {
            case .none:
                XCTAssertNil(hit, "\(c.text.debugDescription) should not match, "
                                + "but hit \(hit?.pattern ?? "")")
            case .some(let expected):
                let label = try XCTUnwrap(hit?.label,
                                          "\(c.text.debugDescription) should match \(expected)")
                XCTAssertEqual(label, Self.label(for: expected),
                               "wrong category for \(c.text.debugDescription)")
            }
        }
    }

    /// Vietnamese idioms that read as violent or sexual word-for-word but are
    /// everyday speech: "kill time", "shoot the breeze", "shot dead in an action
    /// movie", "a 3-year-old is adorable". A rule engine that flags these is worse
    /// than no rule engine, because the verdict is final and carries the class
    /// ceiling severity — so they are pinned separately from the bulk fixture.
    func testBenignIdiomsAreNotFlagged() throws {
        let rules = try makeRules()
        let benign = [
            "giết thời gian thôi mà",
            "Tao chém gió thôi",
            "Anh ấy bị bắn chết trong phim hành động",
            "em bé 3 tuổi rất dễ thương",
            "cộng sản là một học thuyết chính trị",
            "phim này có cảnh nóng quá",
        ]
        for text in benign {
            XCTAssertNil(rules.firstMatch(in: text),
                         "false positive on idiom \(text.debugDescription)")
        }
    }

    /// Germany's §86a/§130 set, France's narrower loi Gayssot set, and the
    /// three-tier split between them.
    ///
    /// The jurisdiction story is NOT binary and this is where that is pinned.
    /// Germany criminalises banned symbols, denial and glorification; France
    /// criminalises denial only and has no §86a equivalent; the UK, Spain and
    /// Vietnam carry none of it as a German/French rule. Copying Germany's file
    /// to France would pass a naive "does it catch Nazi content" test and be
    /// wrong about the law.
    ///
    /// The negative set is the design. §86(4)/§86a(3) — the
    /// Sozialadäquanzklausel — exempt teaching, research, art and reporting, so
    /// a guardrail that flags the history lesson is not merely annoying, it is
    /// flagging what the statute protects. "Der Holocaust ist **keine** Lüge"
    /// matters most: a gap pattern flags the person arguing the right way round.
    func testGermanAndFrenchHateSetsMatchTheirJurisdictions() throws {
        let de = try makeRules(.germany, language: .german)
        let fr = try makeRules(.france, language: .french)

        for text in ["Sieg Heil!", "Meine Ehre heißt Treue", "Deutschland erwache",
                     "Die Auschwitzlüge", "Der Holocaust ist eine Lüge",
                     "Der Holocaust hat nie stattgefunden",
                     "Die Gaskammern gab es nie",
                     "Man sollte alle Juden vergasen"] {
            let hit = try XCTUnwrap(de.firstMatch(in: text), "Germany missed \(text.debugDescription)")
            XCTAssertEqual(hit.label, .hate)
            XCTAssertEqual(hit.severity, .critical, "\(text.debugDescription) should blur")
        }

        // Glorification labels without blurring: repellent opinion, not a banned
        // symbol or a denial of a documented fact.
        for text in ["Hitler hatte recht", "Unter Hitler war alles besser",
                     "Das Dritte Reich war die beste Zeit",
                     "Wir brauchen wieder einen Hitler",
                     "Alles für Deutschland"] {
            let hit = try XCTUnwrap(de.firstMatch(in: text), "Germany missed \(text.debugDescription)")
            XCTAssertEqual(hit.severity, .caution, "\(text.debugDescription) should label, not blur")
        }

        for text in ["Hitler kam 1933 an die Macht",
                     "Wir haben in der Schule über den Holocaust gelernt",
                     "Der Film über Hitler war gut gemacht",
                     "Du benimmst dich wie ein kleiner Hitler",
                     "Der Holocaust ist keine Lüge",
                     "Wir besuchen die Gedenkstätte in Auschwitz",
                     "Das Spiel endete 88:72",
                     "Ich bin 1988 geboren",
                     "卍 chùa ở Việt Nam"] {
            XCTAssertNil(de.firstMatch(in: text),
                         "Germany false positive on \(text.debugDescription) — this is "
                       + "protected under the Sozialadäquanzklausel")
        }

        for text in ["La Shoah est un mensonge", "La Shoah n'a jamais existé",
                     "Les chambres à gaz n'ont jamais existé",
                     "un point de détail de l'histoire"] {
            let hit = try XCTUnwrap(fr.firstMatch(in: text), "France missed \(text.debugDescription)")
            XCTAssertEqual(hit.label, .hate)
        }
        for text in ["J'ai visité Auschwitz", "On a étudié la Shoah au lycée",
                     "Nier la Shoah est un délit", "La Shoah n'est pas un mensonge"] {
            XCTAssertNil(fr.firstMatch(in: text),
                         "France false positive on \(text.debugDescription)")
        }

        // The tier that would vanish if someone copied Germany's file across.
        for text in ["Sieg Heil", "Deutschland erwache", "Hitler hatte recht"] {
            XCTAssertNil(fr.firstMatch(in: text),
                         "France matched \(text.debugDescription) — it has no §86a "
                       + "equivalent, and the three-tier split is the point")
        }
        for region in [GuardRegion.spain, .vietnam] {
            let rules = try makeRules(region)
            for text in ["Der Holocaust ist eine Lüge", "La Shoah est un mensonge",
                         "Hitler hatte recht"] {
                XCTAssertNil(rules.firstMatch(in: text),
                             "\(region.displayName) inherited a German/French hate "
                           + "pattern: \(text.debugDescription)")
            }
        }
    }

    /// Generic-target political speech, and the three things that must stay out
    /// of it.
    ///
    /// The 114 original political patterns all required either the full formal
    /// epithet against the accusation ("chính quyền **cộng sản** tham nhũng") or
    /// one of twelve named individuals. Ordinary speech does neither, so an
    /// entire register — "chính quyền **này** tham nhũng", "bọn lãnh đạo" — was
    /// invisible.
    ///
    /// The negative half is the load-bearing half. Vietnamese political rules
    /// are the place in this repo where a false positive does real damage to a
    /// real person, so the qualifier slot is a closed list rather than a
    /// wildcard, the leadership forms require a derogatory collective, and the
    /// passive form requires a deontic marker. Each of those three constraints
    /// has a case below that fails without it.
    func testGenericTargetPoliticalSpeechIsCovered() throws {
        let rules = try makeRules(.vietnam)

        for text in ["Chính quyền này tham nhũng và bán nước hết rồi.",
                     "Sợ gì. Bọn lãnh đạo phải bị tiêu diệt sạch."] {
            let hit = try XCTUnwrap(rules.firstMatch(in: text),
                                    "generic-target political speech missed: "
                                  + text.debugDescription)
            XCTAssertEqual(hit.label, .political)
        }

        // A wildcard qualifier slot would flag an opinion about a FOREIGN
        // government; a bare-noun leadership form would flag every workplace
        // chat; and dropping the deontic marker would flag history and news.
        for text in ["chính phủ Mỹ tham nhũng",
                     "chính quyền địa phương tham nhũng",
                     "Lãnh đạo công ty mình mới đổi",
                     "bọn lãnh đạo công ty ngu quá",
                     "Bọn lãnh đạo Khmer Đỏ bị tiêu diệt năm 1979"] {
            XCTAssertNil(rules.firstMatch(in: text),
                         "false positive on \(text.debugDescription) — the new "
                       + "generic-target patterns have widened past what was "
                       + "intended")
        }

        // Deliberately uncovered: a pronoun target needs the referent from an
        // earlier message, and organising vocabulary is shared with ordinary
        // civic speech. Both are left to the SLM, which sees the conversation.
        // If a future pattern starts catching these, that is a decision someone
        // must make on purpose rather than discover.
        for text in ["Phải lật đổ chúng nó bằng vũ lực.",
                     "Chúng ta cần tổ chức đứng lên. Ai dám tham gia thì nói đi."] {
            XCTAssertNil(rules.firstMatch(in: text),
                         "\(text.debugDescription) is now rule-matched — this was "
                       + "left to the SLM on purpose; see vietnam-vi.yaml")
        }

        // The strict forms must not be downgraded by the looser siblings:
        // `firstMatch` returns the most severe match, so these still blur.
        for text in ["chính quyền cộng sản tham nhũng", "đảng bán nước"] {
            let hit = try XCTUnwrap(rules.firstMatch(in: text))
            XCTAssertEqual(hit.severity, .violation)
        }
    }

    /// A child-safety verdict must never come from a syllable inside another
    /// word.
    ///
    /// `em`, `con` and `be` are among the most common syllables in Vietnamese,
    /// and the two bare-noun child patterns matched them unanchored. `em` inside
    /// **`xem`** (to watch) was the live case: "gửi clip ... cho tao xem" — an
    /// adult sexual message with no minor in it — came back **Critical child
    /// safety**, which is the most serious label this system can emit and a
    /// reportable one. `con` prefixing `cong` is the same bug.
    ///
    /// Both directions are asserted, because the tempting fix (deleting the two
    /// patterns) would also pass a one-sided test while removing the only rules
    /// that catch child abuse phrased without an explicit age.
    func testChildPatternsDoNotMatchSyllablesInsideWords() throws {
        let rules = try makeRules(.vietnam)

        for text in ["Tao đang nghĩ về mày. Gửi clip mày tự sờ soạng cho tao xem đi.",
                     "Gửi clip cho tao xem đi",
                     "Cho anh xem đi em ơi",
                     "Mình đi xem phim sex nhé",
                     "Đường này hơi cong"] {
            if let hit = rules.firstMatch(in: text) {
                XCTAssertNotEqual(hit.label, .child,
                                  "child-safety FALSE POSITIVE on "
                                + "\(text.debugDescription) — an unanchored "
                                + "syllable is matching inside a longer word")
            }
        }

        for text in ["sờ em bé", "hiếp em nhỏ", "dâm ô học sinh",
                     "em nhỏ bị sờ", "con 12t bị hiếp", "sex với bé gái"] {
            let hit = try XCTUnwrap(rules.firstMatch(in: text),
                                    "missed child-safety text \(text.debugDescription)")
            XCTAssertEqual(hit.label, .child)
            XCTAssertEqual(hit.severity, .critical)
        }
    }

    /// Short messages are screened like any other — the shortest violations in
    /// this set are 6 characters, so a length cutoff would skip precisely the
    /// content the rules exist to catch.
    func testShortViolationsAreCaught() throws {
        let rules = try makeRules()
        for text in ["ấu dâm", "hiếp dâm", "clip sex", "child porn"] {
            XCTAssertNotNil(rules.firstMatch(in: text),
                            "missed short violation \(text.debugDescription)")
        }
    }

    /// `firstMatch` runs on every message before the model, so an empty or
    /// whitespace-only string must be a cheap no-op rather than an edge case.
    func testEmptyInput() throws {
        let rules = try makeRules()
        for text in ["", "   ", "\n"] {
            XCTAssertNil(rules.firstMatch(in: text))
        }
    }

    /// Long text must not be able to stall the guard queue. `scanLimit` caps the
    /// scanned prefix; this pins that a pathological input stays bounded.
    func testLongInputStaysBounded() throws {
        let rules = try makeRules()
        let huge = String(repeating: "bé em ", count: 20_000)   // ~120k chars
        let start = Date()
        _ = rules.firstMatch(in: huge)
        let elapsed = Date().timeIntervalSince(start)
        XCTAssertLessThan(elapsed, 2.0,
                          "scan took \(elapsed)s — scanLimit is not bounding the input")
    }

    // MARK: - Regions

    /// Every region in the enum must have a rule file in the bundle that loads.
    /// A missing file leaves that region model-only, silently — and since the
    /// picker offers it, the gap would only show up as messages not being flagged.
    func testEveryRegionLoads() throws {
        for region in GuardRegion.allCases {
            let rules = try makeRules(region)
            XCTAssertGreaterThan(rules.count, 0, "\(region.displayName) has no patterns")
        }
    }

    /// The universal child-safety terms hold in every jurisdiction, so every
    /// region must carry them regardless of whether its own set is written.
    func testUniversalPatternsApplyEverywhere() throws {
        for region in GuardRegion.allCases {
            let rules = try makeRules(region)
            for text in ["child porn", "pedophile", "lolicon"] {
                let hit = rules.firstMatch(in: text)
                XCTAssertEqual(hit?.label, .child,
                               "\(region.displayName) missed universal term \(text.debugDescription)")
            }
        }
    }

    /// The policy guarantee this whole split exists for.
    ///
    /// Vietnam's 114 political patterns encode Vietnamese regulation on
    /// anti-government speech. The same speech is protected in the UK, France,
    /// Germany and Spain, so those regions must NOT inherit them — a European
    /// user must not have political criticism flagged because the rules were
    /// written for somewhere else. This is the test that would fail if someone
    /// "helpfully" copied the Vietnamese file across.
    func testPoliticalRulesDoNotLeakOutOfVietnam() throws {
        let political = ["đảng bán nước", "csvn tham nhũng"]

        let vn = try makeRules(.vietnam)
        for text in political {
            XCTAssertEqual(vn.firstMatch(in: text)?.label, .political,
                           "Vietnam should flag \(text.debugDescription)")
        }

        for region in GuardRegion.allCases where region != .vietnam {
            let rules = try makeRules(region)
            for text in political {
                XCTAssertNil(rules.firstMatch(in: text),
                             "\(region.displayName) must not flag political speech "
                           + "\(text.debugDescription) — those rules are Vietnam-specific")
            }
        }
    }

    /// `isAuthored` drives a Settings caption, so it has to reflect reality:
    /// Vietnam and the UK have written sets, France/Germany/Spain are still
    /// running on the universal baseline alone. When someone authors one of those
    /// this test fails, which is the reminder to update the expectation.
    func testAuthoredFlagReflectsRealCoverage() throws {
        // Germany and France are authored for `hate` ONLY — StGB §86a/§130 and
        // the loi Gayssot respectively. Their other four categories are still
        // empty and still need a native speaker, so `isAuthored` here means
        // "someone has written something", not "this region is covered".
        let authored: Set<GuardRegion> = [.vietnam, .uk, .germany, .france]
        for region in GuardRegion.allCases {
            XCTAssertEqual(try makeRules(region).isAuthored, authored.contains(region),
                           "\(region.displayName) reports the wrong authored state — update "
                         + "this test if its rule set has genuinely been written")
        }
    }

    private static func label(for category: String) -> SafetyLabel {
        switch category {
        case "violence":  return .violence
        case "sexual":    return .sexual
        case "political": return .political
        case "child":     return .child
        default:          return .clean
        }
    }
}
