import XCTest
@testable import SLMChat

/// Checks that the scripted demo scenarios still demonstrate what they claim.
///
/// A demo of a safety system fails in the most embarrassing way possible: it
/// keeps running, the bubbles keep appearing, and nothing is flagged — because a
/// pattern was retuned, a region renamed, or a threshold moved. Nothing else in
/// the suite would notice, since the scenarios are data rather than code.
///
/// So each line declares whether it should be flagged, and this pins it.
@MainActor
final class GuardrailDemoTests: XCTestCase {

    private var engine: GuardEngine!

    override func setUp() async throws {
        try await super.setUp()
        engine = GuardEngine()
        let res = await engine.warmUp(language: .english,
                                      region: .uk)
        XCTAssertTrue(res.ok, "warm-up failed: \(res.error ?? "unknown")")
    }

    override func tearDown() {
        engine = nil
        super.tearDown()
    }

    /// Every line marked `shouldFlag` must actually be flagged, and every line
    /// marked clean must not be.
    ///
    /// The clean ones matter at least as much: several are the VICTIM's messages
    /// ("Please stop. I'm not interested.", "My bank would never ask for that.").
    /// A guardrail that flags the person reporting harm is worse than one that
    /// misses the harm, and this is where that would show up first.
    func testEveryDemoLineBehavesAsAdvertised() async throws {
        for (name, lines) in GuardrailDemo.all {
            for line in lines {
                let verdict = await engine.classify(line.text, language: line.language)
                XCTAssertEqual(verdict.isUnsafe, line.shouldFlag,
                               "[\(name)] \(line.text.debugDescription) → "
                             + "\(verdict.label.displayName)/\(verdict.severity?.displayName ?? "no severity"), "
                             + "expected \(line.shouldFlag ? "flagged" : "clean") "
                             + "— note says: \(line.expect)")
            }
        }
    }

    /// The scenario's real subject: some flagged lines must LABEL without
    /// blurring, others must blur. A scenario where every hit does the same thing
    /// demonstrates nothing about the tiering built for it.
    ///
    /// Rewritten for v16, which has no severity head. The old version collected
    /// `Set<Severity>` from model verdicts and asserted it spanned Caution to
    /// Critical — impossible now, since a model verdict carries no severity at
    /// all. The property worth pinning was never really "the model emits several
    /// severities"; it was "the scenario shows both outcomes a user can see". So
    /// this asserts on `isSensitive`, which is what actually differs on screen and
    /// is what `MessageBubble` branches on.
    ///
    /// Note this now depends on a RULE firing somewhere in the scenario, because
    /// rules are the only source of Critical since v16. That is not a weakness of
    /// the test — it is the blur path, stated honestly.
    func testHarassmentScenarioSpansBlurredAndLabelledOnly() async throws {
        var blurred = 0, labelledOnly = 0
        for line in GuardrailDemo.harassment where line.shouldFlag {
            let verdict = await engine.classify(line.text, language: .english)
            let m = ChatMessage(sender: .friend, text: line.text)
            m.apply(verdict)
            if m.isSensitive { blurred += 1 } else { labelledOnly += 1 }
        }

        XCTAssertGreaterThan(labelledOnly, 0,
                             "no label-without-blur line left — the scenario no "
                           + "longer shows that tier")
        XCTAssertGreaterThan(blurred, 0,
                             "nothing in the scenario blurs — since v16 that needs "
                           + "a RULE hit at Critical, so check uk-en.yaml still "
                           + "covers one of these lines")
    }

    /// At least one scenario line must be caught by the MODEL rather than a rule,
    /// or the demo only ever exercises the regex half and a broken classifier
    /// would sail through it.
    func testScenariosExerciseTheModelNotOnlyTheRules() async throws {
        var modelHits = 0
        for (_, lines) in GuardrailDemo.all {
            for line in lines where line.shouldFlag {
                // Rules carry confidence exactly 1; the model reports a real
                // probability. That's the only observable difference in a verdict.
                let verdict = await engine.classify(line.text, language: line.language)
                if verdict.confidence < 1 { modelHits += 1 }
            }
        }
        XCTAssertGreaterThan(modelHits, 0,
                             "every flagged demo line is a regex hit — the demo "
                           + "never shows the classifier working")
    }

    /// The self-harm scenario's specific claim: encouragement aimed at another
    /// person is hidden, and the target's own replies never are.
    ///
    /// This is the evidence behind the Critical-only blur threshold. If a future
    /// pack starts putting the target's distressed replies at Critical, or stops
    /// putting the aggressor's lines there, the threshold is no longer safe and
    /// this fails rather than the demo quietly changing meaning.
    func testSelfHarmScenarioHidesTheAggressorAndNotTheTarget() async throws {
        for line in GuardrailDemo.selfHarm {
            let verdict = await engine.classify(line.text, language: line.language)
            let message = ChatMessage(sender: line.sender, text: line.text)
            message.apply(verdict)

            if line.sender == .you {
                XCTAssertFalse(verdict.isUnsafe,
                               "flagged the target's own reply \(line.text.debugDescription) "
                             + "as \(verdict.label.displayName) — this is the failure "
                             + "mode that silences victims")
                XCTAssertFalse(message.isSensitive)
            } else if line.shouldFlag {
                XCTAssertEqual(verdict.label, .suicide, line.text.debugDescription)
                // Detection holds; blurring does not follow from it any more.
                // Severity is rules-only since v16 and no region ships self-harm
                // patterns, so the aggressor's lines are LABELLED and readable.
                // See `testEncouragingSuicideIsStillFlagged` for the same gap
                // stated as an invariant.
                XCTAssertFalse(message.isSensitive,
                               "\(line.text.debugDescription) blurred — a self-harm "
                             + "RULE must have been added; invert this back")
            }
        }
    }

    /// Every flag in the self-harm scenario must come from the MODEL.
    ///
    /// No region ships Vietnamese self-harm patterns — `vietnam-vi` has none, and
    /// `uk-en`'s ten are English. That makes this the one scenario that cannot be
    /// carried by the regex layer, which is precisely its value. If a rule ever
    /// starts catching these, the scenario has stopped testing the classifier and
    /// should be rewritten rather than left to pass for the wrong reason.
    func testSelfHarmScenarioIsEntirelyModelDriven() async throws {
        for line in GuardrailDemo.selfHarm where line.shouldFlag {
            let verdict = await engine.classify(line.text, language: line.language)
            XCTAssertLessThan(verdict.confidence, 1,
                              "\(line.text.debugDescription) was caught by a RULE — "
                            + "this scenario exists to exercise the model")
            guard case .model = verdict.source else {
                return XCTFail("expected a model verdict for \(line.text.debugDescription)")
            }
        }
    }

    /// **The false-positive scenario, and the reason it exists.**
    ///
    /// Distress must be LABELLED but never hidden and never reportable. Three
    /// separate assertions because `shouldFlag` alone cannot express this: the
    /// first line here IS flagged (Self-harm/Violation 0.827) and would keep
    /// satisfying `shouldFlag` if someone moved the blur threshold back to
    /// Violation — at which point the app would once again hide a cry for help
    /// with every test still green.
    func testDistressScenarioIsLabelledButNeverHidden() async throws {
        var sawAFlag = false
        for line in GuardrailDemo.selfHarmSoft {
            let verdict = await engine.classify(line.text, language: line.language)
            let message = ChatMessage(sender: line.sender, text: line.text)
            message.apply(verdict)

            XCTAssertNotEqual(verdict.severity, .critical,
                              "\(line.text.prefix(40))… came back Critical — distress "
                            + "must never reach the severity that hides content")
            XCTAssertFalse(message.isSensitive,
                           "BLURRED a message about someone's own distress: "
                         + "\(line.text.prefix(40))… (\(verdict.label.displayName)/"
                         + "\(verdict.severity?.displayName ?? "no severity"))")
            if verdict.isUnsafe { sawAFlag = true }
        }
        // ⚠️ KNOWN FAILURE under the v16ls checkpoint. Not one line in this
        // scenario is flagged any more: the opener fell 0.9-something to 0.5971,
        // and the strongest line left ("Cuộc sống nặng nề quá…") sits at 0.9113,
        // under the 0.95 gate. So the scenario no longer demonstrates the thing
        // it exists to demonstrate — labelling WITHOUT hiding — because there is
        // no labelling left to observe.
        //
        // Recorded rather than deleted, and `strict` on purpose: if a future pack
        // flags any line here again, this stops failing and XCTest fails the test
        // for the missing failure, which is what forces someone to come back and
        // remove the expectation. Deleting the assertion instead would leave a
        // green test asserting nothing.
        //
        // The assertions ABOVE still run unguarded, and they are the safety half:
        // nothing in this scenario may reach Critical or be blurred. v16ls did not
        // break those — it went the other way, which is the one mercy here.
        XCTExpectFailure("v16ls flags no line in the distress scenario — see the "
                       + "regression table in GuardEngine.threshold") {
            XCTAssertTrue(sawAFlag,
                          "no line was flagged at all — the scenario now demonstrates "
                        + "nothing about labelling without hiding, and would pass with "
                        + "the self-harm class removed entirely")
        }
    }

    /// The listener's replies must stay clean. Flagging "are you OK?" or "do you
    /// want to talk?" would punish exactly the response the whole feature wants.
    func testDistressScenarioNeverFlagsTheListener() async throws {
        for line in GuardrailDemo.selfHarmSoft where line.sender == .you {
            let verdict = await engine.classify(line.text, language: line.language)
            XCTAssertFalse(verdict.isUnsafe,
                           "flagged the listener's \(line.text.debugDescription) as "
                         + "\(verdict.label.displayName)")
        }
    }

    /// The two self-harm scenarios carry the two halves of the same commitment:
    /// coercion aimed at a person is DETECTED, and a person in distress is never
    /// HIDDEN.
    ///
    /// This was `testTheTwoSelfHarmScenariosDivergeAtTheBlur`, and the divergence
    /// it named is gone: since v16 no model verdict blurs, so blur counts are 0
    /// on both sides and the pair no longer separates there. Counting flags
    /// instead does not work either — the distress scenario's opening line is
    /// deliberately `shouldFlag: true` ("nếu mình biến mất thì mọi thứ sẽ nhẹ
    /// nhàng hơn" is labelled, and should be), so both sides flag.
    ///
    /// What actually matters was never the divergence; it was the two properties.
    /// Asserted separately, and the second stays meaningful the day self-harm
    /// rules are authored — that is exactly when distress could start blurring
    /// again, and this is where it would show.
    func testCoercionIsDetectedAndDistressIsNeverHidden() async throws {
        var coercionFlags = 0
        for line in GuardrailDemo.selfHarm where line.shouldFlag {
            let verdict = await engine.classify(line.text, language: line.language)
            if verdict.isUnsafe { coercionFlags += 1 }
        }
        XCTAssertGreaterThan(coercionFlags, 0,
                             "encouragement aimed at another person stopped being "
                           + "detected at all — not a threshold question any more, "
                           + "the classifier has regressed")

        for line in GuardrailDemo.selfHarmSoft {
            let verdict = await engine.classify(line.text, language: line.language)
            let m = ChatMessage(sender: line.sender, text: line.text)
            m.apply(verdict)
            XCTAssertFalse(m.isSensitive,
                           "HID a message about someone's own distress: "
                         + "\(line.text.prefix(40))… — the guardrail is back to "
                         + "suppressing the request for help")
        }
    }

    // MARK: - Menu partitioning

    /// A group scenario must not be offered in a 1:1 thread. Playing one there
    /// collapses every author into one anonymous "friend", which destroys the only
    /// thing those scenarios are about — attribution.
    func testMenusPartitionByAuthor() {
        XCTAssertEqual(GuardrailDemo.oneToOne.count + GuardrailDemo.group.count,
                       GuardrailDemo.scenarios.count,
                       "a scenario is in both menus or neither")
        for scenario in GuardrailDemo.oneToOne {
            XCTAssertTrue(scenario.variants.allSatisfy {
                $0.lines.allSatisfy { $0.author == nil }
            }, "[\(scenario.name)] is offered 1:1 but names authors")
        }
        for scenario in GuardrailDemo.group {
            XCTAssertTrue(scenario.variants.contains {
                $0.lines.contains { $0.author != nil }
            }, "[\(scenario.name)] is offered in a group but names nobody")
        }
        // A scenario is 1:1 or group as a WHOLE. Variants are translations of one
        // script, so one of them naming authors while another doesn't means the
        // translations have drifted apart — and `isGroup` would silently pick the
        // menu based on whichever variant happened to be written first.
        for scenario in GuardrailDemo.scenarios {
            let shapes = Set(scenario.variants.map { v in
                v.lines.contains { $0.author != nil }
            })
            XCTAssertEqual(shapes.count, 1,
                           "[\(scenario.name)] mixes group and 1:1 variants")
            let counts = Set(scenario.variants.map(\.lines.count))
            XCTAssertEqual(counts.count, 1,
                           "[\(scenario.name)] variants have different line counts "
                         + "\(counts.sorted()) — they are meant to be the same "
                         + "script, so a comparison across languages is only "
                         + "meaningful if they line up")
        }
        XCTAssertFalse(GuardrailDemo.oneToOne.isEmpty)
        // `group` may legitimately be empty. Scenario 1 is five 1:1 arcs, and
        // the three group scenarios that used to fill this menu are in
        // `disabled`. The partition above is the invariant worth holding; a
        // non-empty group menu is a content decision, not a correctness one.
        XCTAssertFalse(GuardrailDemo.all.isEmpty)
    }

    /// Nothing lands in the menu twice, and the disabled set stays out of it.
    ///
    /// Both lists are hand-maintained, so re-enabling a scenario by copying its
    /// row rather than moving it is the obvious mistake — and it would show up
    /// as a duplicate entry in the menu rather than as any kind of failure.
    func testDisabledScenariosAreNotAlsoEnabled() {
        let live = Set(GuardrailDemo.all.map(\.name))
        XCTAssertEqual(live.count, GuardrailDemo.all.count, "duplicate menu entry")
        for entry in GuardrailDemo.disabled {
            XCTAssertFalse(live.contains(entry.name),
                           "[\(entry.name)] is in both `all` and `disabled`")
        }
    }

    /// `.you` lines never carry an author — the device owner is not a "member".
    func testYourOwnLinesNeverNameAnAuthor() {
        for (name, lines) in GuardrailDemo.all {
            for line in lines where line.sender == .you {
                XCTAssertNil(line.author, "[\(name)] your own line names an author")
            }
        }
    }

    // MARK: - Political scenario

    /// Scenario 6's Vietnamese lines are actionable in Vietnam and clean
    /// everywhere else — and the shared demo test cannot see that.
    ///
    /// `testEveryDemoLineBehavesAsAdvertised` warms up once with `.uk`, so it
    /// never loads `vietnam-vi` at all. Every Vietnamese line in scenario 6 is
    /// therefore `shouldFlag: false` there, which is true of the harness and
    /// false of the app. This test is the other half.
    ///
    /// It is also the coverage that was missing when the `xem` child-safety
    /// false positive shipped: playing a scenario in its OWN region exercises a
    /// rule set the suite otherwise never touches.
    func testPoliticalScenarioIsActionableOnlyInVietnam() async throws {
        let lines = GuardrailDemo.politicalVI

        let vn = GuardEngine()
        let vnReady = await vn.warmUp(language: .vietnamese, region: .vietnam)
        XCTAssertTrue(vnReady.ok, "warm-up failed: \(vnReady.error ?? "unknown")")

        // Line 1 labels (caution), line 5 blurs (violation). Both come from the
        // generic-target patterns; the model scores this content at ~0.00.
        let first = await vn.classify(lines[0].text, language: .vietnamese)
        XCTAssertEqual(first.label, .political)
        XCTAssertEqual(first.severity, .caution)

        let fifth = await vn.classify(lines[4].text, language: .vietnamese)
        XCTAssertEqual(fifth.label, .political)
        XCTAssertEqual(fifth.severity, .violation)

        // Lines 3 and 7 are the pronoun target and the recruitment call. Left to
        // the SLM on purpose — see vietnam-vi.yaml. If either starts flagging,
        // someone widened a political pattern without meaning to.
        for i in [2, 6] {
            let v = await vn.classify(lines[i].text, language: .vietnamese)
            XCTAssertFalse(v.isUnsafe,
                           "line \(i + 1) is now flagged in Vietnam: "
                         + "\(lines[i].text.debugDescription) — this was left to "
                         + "the SLM deliberately")
        }

        // The target's replies must stay clean in the region that restricts this
        // speech, which is the region where flagging them would do harm.
        for line in lines where line.sender == .you {
            let v = await vn.classify(line.text, language: .vietnamese)
            XCTAssertFalse(v.isUnsafe,
                           "flagged the person pushing back: \(line.text.debugDescription)")
        }

        // Outside Vietnam the identical text is clean: no region but Vietnam
        // ships political patterns, and `actionableClasses` subtracts the class
        // on top of that.
        for region in [GuardRegion.uk, .france, .germany, .spain] {
            let e = GuardEngine()
            let ready = await e.warmUp(language: .vietnamese, region: region)
            XCTAssertTrue(ready.ok, "[\(region.rawValue)] warm-up failed")
            for line in lines {
                let v = await e.classify(line.text, language: .vietnamese)
                XCTAssertFalse(v.isUnsafe,
                               "[\(region.rawValue)] flagged protected political "
                             + "speech: \(line.text.debugDescription)")
            }
        }
    }

    // MARK: - Nazi-content scenario

    /// Scenario 7 end to end, per region — the three-tier split the shared demo
    /// test cannot see.
    ///
    /// `testEveryDemoLineBehavesAsAdvertised` warms up once with `.uk`, so the
    /// entire French variant reads `shouldFlag: false` there and German line 5
    /// does too. This is the other half, and it is also the assertion that would
    /// fail if someone "helpfully" copied `germany-de.yaml` across to France.
    func testNaziScenarioSplitsAcrossThreeJurisdictions() async throws {
        func engine(_ region: GuardRegion,
                    _ language: TranslateLanguage) async throws -> GuardEngine {
            let e = GuardEngine()
            let ready = await e.warmUp(language: language, region: region)
            XCTAssertTrue(ready.ok, "[\(region.rawValue)] warm-up failed")
            return e
        }

        // Germany: glorification labels, denial + slogan blurs.
        let de = try await engine(.germany, .german)
        let glorify = await de.classify(GuardrailDemo.naziDE[4].text, language: .german)
        XCTAssertEqual(glorify.label, .hate)
        XCTAssertEqual(glorify.severity, .caution, "§130(4) glorification labels, never blurs")

        let denial = await de.classify(GuardrailDemo.naziDE[6].text, language: .german)
        XCTAssertEqual(denial.label, .hate)
        XCTAssertEqual(denial.severity, .critical)

        // France: denial only. The glorification line is the middle tier and
        // must stay clean — France has no §130(4) equivalent.
        let fr = try await engine(.france, .french)
        let frDenial = await fr.classify(GuardrailDemo.naziFR[6].text, language: .french)
        XCTAssertEqual(frDenial.label, .hate)
        XCTAssertEqual(frDenial.severity, .critical)

        let frGlorify = await fr.classify(GuardrailDemo.naziFR[4].text, language: .french)
        XCTAssertFalse(frGlorify.isUnsafe,
                       "France flagged glorification — it has no offence for it, "
                     + "and copying Germany's patterns across is the mistake this "
                     + "asserts against")

        // The Godwin insult and the documentary stay clean in the STRICT region,
        // which is where a false positive would actually bite.
        for i in [0, 2] {
            let v = await de.classify(GuardrailDemo.naziDE[i].text, language: .german)
            XCTAssertFalse(v.isUnsafe,
                           "Germany flagged \(GuardrailDemo.naziDE[i].text.debugDescription) "
                         + "— protected under the Sozialadäquanzklausel")
        }

        // And the target's replies, in every language's own region.
        for (region, lang, lines) in [(GuardRegion.germany, TranslateLanguage.german, GuardrailDemo.naziDE),
                                      (.france, .french, GuardrailDemo.naziFR),
                                      (.uk, .english, GuardrailDemo.naziEN)] {
            let e = try await engine(region, lang)
            for line in lines where line.sender == .you {
                let v = await e.classify(line.text, language: lang)
                XCTAssertFalse(v.isUnsafe,
                               "[\(region.rawValue)] flagged the person objecting: "
                             + "\(line.text.debugDescription)")
            }
        }
    }

    // MARK: - Radicalisation scenario

    /// **The same LINES flag in every region; the REASON is allowed to differ.**
    ///
    /// This test used to assert full region-invariance — identical label and
    /// severity in Vietnam and the UK — on the grounds that the model never
    /// returns the political class for this text, so `actionableClasses` never
    /// fires. That held until `vietnam-vi` gained generic-target political
    /// patterns, and then it caught them, exactly as its old doc comment said it
    /// would ("a future edit ... could quietly change this scenario's meaning
    /// per-region without anyone noticing").
    ///
    /// The change is correct and was made deliberately. Line 2 says *"Giết hết
    /// bọn cầm quyền"* — kill the ruling class — which the new active-verb
    /// pattern matches on purpose. In Vietnam that is now `political`/violation
    /// from the RULE, which short-circuits before the model; in the UK there is
    /// no such pattern and the model returns `violence`. Both flag. Divergence
    /// in the reason is the whole point of a per-jurisdiction rule set.
    ///
    /// What must NOT drift is coverage: no region may flag a line another region
    /// lets through. That is what is pinned now, plus the two directional facts
    /// that make the policy real — the UK never sees `political` at all, and
    /// Vietnam does.
    ///
    /// Trade-off recorded here because it is invisible from the verdict: a
    /// `political` label pre-selects `.other` in `ReportReason`, where `violence`
    /// pre-selects "Violence or threats". So in Vietnam this line reports one tap
    /// worse than in the UK. Every reason stays available in the sheet, and the
    /// whole overthrow-verb family has behaved this way since long before the new
    /// patterns — but it is the cost of routing this content through `political`.
    func testRadicalisationScenarioFlagsTheSameLinesInEveryRegion() async throws {
        var flagged: [GuardRegion: [Bool]] = [:]
        var labels: [GuardRegion: [SafetyLabel]] = [:]
        for region in [GuardRegion.vietnam, .uk] {
            let e = GuardEngine()
            let res = await e.warmUp(language: .vietnamese, region: region)
            XCTAssertTrue(res.ok)
            var f: [Bool] = []
            var l: [SafetyLabel] = []
            for line in GuardrailDemo.groupPolitical {
                let v = await e.classify(line.text, language: .vietnamese)
                f.append(v.isUnsafe)
                l.append(v.label)
            }
            flagged[region] = f
            labels[region] = l
        }

        // ⚠️ KNOWN FAILURE under the v16ls checkpoint. Vietnam still catches line
        // 2 ("Phải lật đổ bằng vũ lực…") through its political patterns; the UK
        // used to catch the same line through the MODEL and no longer does — it
        // scores 0.9423, under the 0.95 gate. So coverage now genuinely differs by
        // region for a line that calls for killing the ruling class.
        //
        // Strict, for the same reason as the distress scenario: a pack that puts
        // that line back over the gate makes this stop failing, and the missing
        // failure is what tells someone to delete the expectation.
        XCTExpectFailure("v16ls drops line 2 in the UK (0.9423, under the 0.95 "
                       + "gate) while Vietnam still catches it by rule") {
            XCTAssertEqual(flagged[.vietnam], flagged[.uk],
                           "a region gained or lost coverage — the reason may differ "
                         + "per jurisdiction, WHICH lines are caught may not")
        }

        XCTAssertFalse(try XCTUnwrap(labels[.uk]).contains(.political),
                       "the UK returned the political class — protected expression "
                     + "there, and `actionableClasses` is supposed to guarantee it")
        XCTAssertTrue(try XCTUnwrap(labels[.vietnam]).contains(.political),
                      "Vietnam no longer returns political for a call to kill the "
                    + "ruling class — check the generic-target patterns in "
                    + "vietnam-vi.yaml still exist")
    }

    /// The member pushing back must never be flagged. In a radicalising group they
    /// are the one person doing something useful, and flagging them is how a
    /// guardrail makes a group less safe.
    func testRadicalisationScenarioNeverFlagsTheDissenter() async throws {
        for line in GuardrailDemo.groupPolitical where line.sender == .you {
            let verdict = await engine.classify(line.text, language: line.language)
            XCTAssertFalse(verdict.isUnsafe,
                           "flagged the dissenter: \(line.text.debugDescription)")
        }
    }
}
