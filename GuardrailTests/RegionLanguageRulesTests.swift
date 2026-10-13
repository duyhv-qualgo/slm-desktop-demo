import XCTest
@testable import SLMChat

/// Checks that rule sets are keyed by BOTH region and language.
///
/// Keying by region alone silently meant "region, in whichever language someone
/// happened to author it", and the gap was measured rather than theorised. For a
/// reader in London:
///
///     "Move your money to a safe account straight away."   -> Scam / Critical
///     "Chuyển tiền sang tài khoản an toàn ngay lập tức."    -> Clean, 0.915
///
/// The same scam, the same reader, opposite outcomes — decided entirely by the
/// language the rule file happened to be written in. `uk-en.yaml` is English and
/// cannot match Vietnamese, and the classifier misses the UK-specific scam shapes
/// (safe account, OTP theft) because they are not in its training distribution.
@MainActor
final class RegionLanguageRulesTests: XCTestCase {

    private func engine(_ region: GuardRegion) async throws -> GuardEngine {
        let e = GuardEngine()
        let res = await e.warmUp(language: .vietnamese, region: region)
        XCTAssertTrue(res.ok, "warm-up failed: \(res.error ?? "unknown")")
        return e
    }

    // MARK: - Catalog integrity

    /// Every declared (region, language) pair must resolve to a file in the
    /// bundle. A missing one is not an error at load — the region simply falls
    /// back to the universal baseline — so the failure mode is a region quietly
    /// screening far less than its author believes.
    func testEveryDeclaredPairHasAFile() throws {
        for region in GuardRegion.allCases {
            for language in region.ruleLanguages {
                let name = region.rulesResource(for: language)
                XCTAssertNotNil(ResourceLookup.url(name, ext: "json"),
                                "\(name).json declared in ruleLanguages but not in the bundle")
            }
        }
        XCTAssertNotNil(ResourceLookup.url(GuardRegion.universalRulesResource, ext: "json"),
                        "the universal fallback must always exist")
    }

    /// The UK serves a large Vietnamese-speaking population, which is the whole
    /// reason this key exists. If `uk-vi` is ever dropped, this says so.
    func testUKShipsBothEnglishAndVietnameseRules() {
        XCTAssertTrue(GuardRegion.uk.ruleLanguages.contains(.english))
        XCTAssertTrue(GuardRegion.uk.ruleLanguages.contains(.vietnamese),
                      "the UK must ship Vietnamese patterns — without them a UK "
                    + "reader with Vietnamese contacts has no scam rule coverage")
    }

    // MARK: - The gap this closed

    /// Vietnamese-language scams must be caught for a UK reader. Each of these
    /// was measured returning Clean before `uk-vi.yaml` existed.
    func testUKCatchesVietnameseLanguageScams() async throws {
        let e = try await engine(.uk)
        let scams = [
            "Chuyển tiền gấp cho tôi, tài khoản mới.",
            "Chuyển tiền sang tài khoản an toàn ngay lập tức",
            "Đọc cho tôi mã OTP 6 số vừa gửi",
            "Mẹ ơi con mất điện thoại, đây là số mới của con, chuyển tiền cho con nhé",
        ]
        for text in scams {
            let verdict = await e.classify(text, language: .vietnamese)
            XCTAssertEqual(verdict.label, .scam,
                           "\(text.debugDescription) → \(verdict.label.displayName)")
        }
    }

    /// The English set must keep working — adding a second language is additive.
    func testUKStillCatchesEnglishScams() async throws {
        let e = try await engine(.uk)
        for text in ["Move your money to a safe account straight away.",
                     "Read me the 6 digit code we just sent you."] {
            let verdict = await e.classify(text, language: .english)
            XCTAssertEqual(verdict.label, .scam, text.debugDescription)
        }
    }

    /// **Rule coverage must not depend on language detection.**
    ///
    /// Selecting one set by the detected language was the obvious reading of this
    /// design and it is wrong: `NLLanguageRecognizer` abstains on short messages,
    /// which are exactly the ones rules are best at, and a wrong guess would
    /// silently route text to a set with no patterns for it. All of a region's
    /// sets are scanned, which is free because a pattern only matches its own
    /// language anyway.
    func testRulesFireRegardlessOfTheDetectedLanguage() async throws {
        let e = try await engine(.uk)

        // English text, deliberately routed as Vietnamese.
        let misrouted = await e.classify("Move your money to a safe account straight away.",
                                         language: .vietnamese)
        XCTAssertEqual(misrouted.label, .scam,
                       "a misdetected language must not cost rule coverage")

        // Vietnamese text, deliberately routed as English.
        let other = await e.classify("Đọc cho tôi mã OTP 6 số vừa gửi", language: .english)
        XCTAssertEqual(other.label, .scam,
                       "a misdetected language must not cost rule coverage")
    }

    // MARK: - Isolation still holds

    /// Adding `uk-vi` must not import Vietnam's political patterns. This is the
    /// most likely way the isolation guarantee gets broken from here — someone
    /// authoring UK Vietnamese patterns copies from `vietnam-vi.yaml`.
    func testUKVietnameseRulesCarryNoPoliticalPatterns() async throws {
        let uk = try await engine(.uk)
        for text in ["đảng bán nước", "csvn tham nhũng"] {
            let verdict = await uk.classify(text, language: .vietnamese)
            XCTAssertFalse(verdict.isUnsafe,
                           "UK flagged protected political speech \(text.debugDescription) "
                         + "as \(verdict.label.displayName) — either uk-vi.yaml gained "
                         + "political patterns or actionableClasses stopped masking them")
        }

        // Vietnam still flags it, so the test above isn't passing because the
        // strings stopped matching anything at all.
        let vn = try await engine(.vietnam)
        let verdict = await vn.classify("đảng bán nước", language: .vietnamese)
        XCTAssertEqual(verdict.label, .political,
                       "Vietnam must still flag its own political patterns")
    }

    /// A region with no set for a language still gets the universal baseline,
    /// rather than nothing.
    func testUnauthoredPairFallsBackToUniversal() async throws {
        let e = try await engine(.germany)     // no TranslateLanguage case for German
        let verdict = await e.classify("child porn", language: .english)
        XCTAssertEqual(verdict.label, .child,
                       "the universal fallback must still screen a region with no "
                     + "authored language sets")
    }

    /// `ruleCount` drives a Settings row, so it has to total every language of the
    /// region rather than reporting one file's worth.
    func testRuleCountTotalsEveryLanguage() async throws {
        let uk = try await engine(.uk)
        let vn = try await engine(.vietnam)
        XCTAssertGreaterThan(uk.ruleCount, 170,
                             "UK should total English + Vietnamese patterns, got \(uk.ruleCount)")
        XCTAssertEqual(vn.ruleCount, 204)
    }

    // MARK: - Child safety across languages

    /// **The worst gap in the rule set, and the one this file exists to prevent.**
    ///
    /// A UK reader's only child-safety coverage in Vietnamese used to be the three
    /// UNIVERSAL patterns, all English (`\bpedophil`, `\bchild\s+porn`, `lolicon`),
    /// which cannot match Vietnamese at all. Measured before the port:
    ///
    ///     "ấu dâm"           -> Clean 0.691    (UK)   vs  Critical by rule (Vietnam)
    ///     "clip ấu dâm 5GB"  -> Clean 0.538    (UK)   vs  Critical by rule (Vietnam)
    ///
    /// "ấu dâm" is the standard Vietnamese term for the offence, and CSEA is an OSA
    /// Schedule 7 priority offence. The 29 Vietnamese patterns were ported from
    /// `vietnam-vi.yaml` — the one legitimate exception to "never copy from that
    /// file", because the rule guards against importing POLICY DIVERGENCE and CSEA
    /// has none between the two jurisdictions.
    func testUKCatchesVietnameseChildSafetyTerms() async throws {
        let uk = try await engine(.uk)
        for text in ["ấu dâm", "clip ấu dâm 5GB", "có clip bé 12 tuổi khỏa thân không"] {
            let verdict = await uk.classify(text, language: .vietnamese)
            XCTAssertEqual(verdict.label, .child,
                           "\(text.debugDescription) → \(verdict.label.displayName) — "
                         + "the uk-vi child patterns are gone or stopped matching")
            guard case .rules = verdict.source else {
                return XCTFail("\(text.debugDescription) fell through to the model; "
                             + "at 0.538 for one of these that is not coverage")
            }
            let m = ChatMessage(sender: .friend, text: text)
            m.apply(verdict)
            XCTAssertTrue(m.isSensitive, "child safety must always blur")
        }
    }

    /// The patterns require sexual context alongside the age marker. Without that
    /// they would fire on every parent mentioning their child's age, which is the
    /// fastest way to make a safety feature unusable.
    func testChildPatternsDoNotFireOnOrdinaryMentionsOfChildren() async throws {
        let uk = try await engine(.uk)
        for text in ["Con mình 5 tuổi rồi.",
                     "Bé nhà mình 3 tuổi hay cười lắm.",
                     "Lớp em có 12 tuổi trở lên mới được đăng ký.",
                     "Hôm qua đi sinh nhật bé 7 tuổi nhà chị Lan."] {
            let verdict = await uk.classify(text, language: .vietnamese)
            XCTAssertFalse(verdict.isUnsafe,
                           "flagged an ordinary mention of a child: "
                         + "\(text.debugDescription) → \(verdict.label.displayName)")
        }
    }

    /// The port must not have dragged politics across with it. This is the specific
    /// failure the "never copy from vietnam-vi" rule exists to catch, and copying
    /// one category from that file is exactly when it would happen.
    func testTheChildPortDidNotImportPoliticalPatterns() async throws {
        let uk = try await engine(.uk)
        for text in ["đảng bán nước", "csvn tham nhũng", "lật đổ chính quyền"] {
            let verdict = await uk.classify(text, language: .vietnamese)
            XCTAssertFalse(verdict.isUnsafe,
                           "UK flagged protected political speech \(text.debugDescription) "
                         + "as \(verdict.label.displayName) after the child port")
        }
    }

    // MARK: - The ported categories

    /// Violence: 17 of Vietnam's 22 patterns, ported because threats to kill are
    /// an OSA Schedule 7 priority offence and a crime in Vietnam — no policy
    /// divergence. Each names a target, which is what keeps Vietnamese idiom out.
    func testUKCatchesVietnameseThreats() async throws {
        let uk = try await engine(.uk)
        for text in ["Tao sẽ chém mày ngay",
                     "Tao tạt axit vào mặt mày",
                     "Tao bóp cổ mày chết luôn"] {
            let verdict = await uk.classify(text, language: .vietnamese)
            XCTAssertEqual(verdict.label, .violence, text.debugDescription)
            guard case .rules = verdict.source else {
                return XCTFail("\(text.debugDescription) fell through to the model")
            }
        }
    }

    /// Idiom must survive the port. These are the phrases the Vietnamese set was
    /// tuned against, and the reason porting beat re-authoring.
    func testVietnameseViolenceIdiomStillDoesNotMatch() async throws {
        let uk = try await engine(.uk)
        for text in ["Ngồi giết thời gian thôi",       // "killing time"
                     "Nó toàn chém gió ấy mà"] {        // "shooting the breeze"
            let verdict = await uk.classify(text, language: .vietnamese)
            XCTAssertFalse(verdict.isUnsafe,
                           "flagged idiom \(text.debugDescription) as "
                         + "\(verdict.label.displayName)")
        }
    }

    /// Sexual: only the ABUSE subset ports. Rape, sextortion, NCII distribution,
    /// voyeurism and bestiality are criminal in both jurisdictions.
    func testUKCatchesVietnameseSexualAbuse() async throws {
        let uk = try await engine(.uk)
        for text in ["Nó hiếp dâm cô ấy",
                     "Không chuyển 20 tr tao tung clip",
                     "quay lén nude em trong nhà tắm"] {
            let verdict = await uk.classify(text, language: .vietnamese)
            XCTAssertEqual(verdict.label, .sexual, text.debugDescription)
            guard case .rules = verdict.source else {
                return XCTFail("\(text.debugDescription) fell through to the model")
            }
        }
    }

    /// **26 of Vietnam's 32 sexual patterns must NOT be here.**
    ///
    /// They are crude consensual-sex vocabulary. Sexual conversation between
    /// consenting adults is lawful in the UK, and `uk-en.yaml` reflects that —
    /// "Send me your nudes." is deliberately Sexual/Caution, labelled not blurred.
    /// Importing the Vietnamese vulgarity list would flag consenting adults
    /// sexting as a violation in a jurisdiction that does not regulate it.
    func testUKDoesNotImportConsensualSexVocabulary() async throws {
        let uk = try await engine(.uk)
        for text in ["Tối nay chịch nhau không", "chat sex không em"] {
            let verdict = await uk.classify(text, language: .vietnamese)
            if case .rules(let pattern) = verdict.source {
                XCTFail("a RULE matched consensual adult talk \(text.debugDescription) "
                      + "— pattern: \(pattern)")
            }
            let m = ChatMessage(sender: .friend, text: text)
            m.apply(verdict)
            XCTAssertFalse(m.isSensitive,
                           "blurred consensual adult talk: \(text.debugDescription) → "
                         + "\(verdict.label.displayName)/"
                         + "\(verdict.severity?.displayName ?? "no severity")")
        }
    }

    /// **The false positive the port surfaced, and the reason two child patterns
    /// were tightened.**
    ///
    /// Vietnam's originals matched a bare `em`/`con` — ordinary affectionate
    /// pronouns between adults, not only "child" — so "chat sex không em" came
    /// back Child safety/CRITICAL and was blurred. Adult sexting classified as
    /// child abuse. The bare pronouns were dropped in `uk-vi.yaml`; this fails if
    /// anyone relaxes them back.
    func testChildPatternsDoNotMatchAdultPronouns() async throws {
        let uk = try await engine(.uk)
        for text in ["chat sex không em", "Anh nhớ em quá", "em ơi mình đi ăn nhé"] {
            let verdict = await uk.classify(text, language: .vietnamese)
            XCTAssertNotEqual(verdict.label, .child,
                              "\(text.debugDescription) matched a child pattern — "
                            + "`em`/`con` are adult pronouns too")
        }
        // The real thing must still be caught, so this isn't passing because the
        // patterns stopped matching anything.
        let real = await uk.classify("có clip bé 12 tuổi khỏa thân không",
                                     language: .vietnamese)
        XCTAssertEqual(real.label, .child)
    }
}
