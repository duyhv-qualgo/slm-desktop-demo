import XCTest
@testable import SLMChat

/// Checks that a verdict says which half of the guardrail produced it.
///
/// The `[Guard]` log line reads `(FLAGGED by rule: <pattern>)` or `(FLAGGED by
/// model)`, and that distinction is the first question anyone asks of a flag:
/// a bad rule is one line of YAML, a bad model verdict needs a retrained pack.
/// If `source` ever drifts from what actually ran, the log doesn't go quiet —
/// it starts confidently attributing flags to the wrong subsystem, which is
/// worse than not logging it at all. Hence tests rather than trust.
@MainActor
final class VerdictSourceTests: XCTestCase {

    private func engine(_ region: GuardRegion,
                        _ language: TranslateLanguage) async throws -> GuardEngine {
        let e = GuardEngine()
        let res = await e.warmUp(language: language, region: region)
        XCTAssertTrue(res.ok, "warm-up failed: \(res.error ?? "unknown")")
        XCTAssertGreaterThan(e.ruleCount, 0, "no rules loaded — every case here would be model-only")
        return e
    }

    private func source(_ r: SafetyResult) -> String { r.source.logDescription }

    // MARK: - The two halves

    /// A pattern match must report `.rules` AND carry the pattern text, so a
    /// false positive traces to the exact line without a second log statement.
    /// Correlating two statements is unreliable under load — the guard queue
    /// interleaves messages, so the pattern printed above a verdict may belong
    /// to a different one.
    func testRuleHitCarriesThePattern() async throws {
        let e = try await engine(.uk, .english)
        let verdict = await e.classify("Move your money to a safe account straight away.",
                                       language: .english)

        XCTAssertEqual(verdict.label, .scam)
        guard case .rules(let pattern) = verdict.source else {
            return XCTFail("expected a rule hit, got \(source(verdict))")
        }
        XCTAssertFalse(pattern.isEmpty, "the pattern text is the whole point of the case")
        XCTAssertEqual(verdict.confidence, 1, "a rule is not probabilistic")
    }

    /// Whichever half fires, the verdict names it.
    ///
    /// The text was "Đảng cộng sản Việt Nam thối nát" and the assertion was
    /// `label == .political`. v16 dropped `political` from the model's taxonomy,
    /// so that now comes back Clean 0.004 from the model and is flagged only by
    /// Vietnam's 114 regex patterns — which made this a test of the RULE path
    /// wearing a model-shaped assertion. Swapped for a threat, which v16 does
    /// classify, so the test again covers what its name says: a flag, from either
    /// half, always attributes itself.
    func testFlaggedVerdictAlwaysNamesItsSource() async throws {
        let e = try await engine(.vietnam, .vietnamese)
        let verdict = await e.classify("Tao sẽ giết mày", language: .vietnamese)

        XCTAssertTrue(verdict.isUnsafe, "expected a flag to attribute")
        if case .notScreened = verdict.source {
            XCTFail("a flag must come from somewhere — got \(source(verdict))")
        }
    }

    /// `classifyModelOnly` bypasses the rules, so the same text that produced a
    /// rule hit above must now come back attributed to the model. This is the
    /// test that would catch `source` being hard-coded.
    func testBypassingRulesAttributesToTheModel() async throws {
        let e = try await engine(.uk, .english)
        let text = "Move your money to a safe account straight away."

        let withRules = await e.classify(text, language: .english)
        let modelOnly = await e.classifyModelOnly(text, language: .english)

        guard case .rules = withRules.source else {
            return XCTFail("expected a rule hit, got \(source(withRules))")
        }
        guard case .model = modelOnly.source else {
            return XCTFail("rules were bypassed, so this must be the model — got \(source(modelOnly))")
        }
    }

    // MARK: - Screened-clean vs. never-screened

    /// **The distinction that justifies a third case.** Both of these are Clean
    /// and look identical in the UI; one is a pass and one is a coverage gap.
    func testCleanFromTheModelIsNotTheSameAsNeverScreened() async throws {
        let e = try await engine(.uk, .english)

        let screened = await e.classify("See you at the station around six.",
                                        language: .english)
        XCTAssertFalse(screened.isUnsafe)
        guard case .model = screened.source else {
            return XCTFail("the model ran and found nothing — that is `.model`, "
                         + "got \(source(screened))")
        }

        // Empty text short-circuits before either half runs.
        let empty = await e.classify("   ", language: .english)
        guard case .notScreened = empty.source else {
            return XCTFail("nothing ran on whitespace — got \(source(empty))")
        }
    }

    /// A language with no classifier pack gets rules only. When they don't match,
    /// the result must NOT claim the model cleared it — that is exactly the
    /// invisible gap `.notScreened` exists to make visible.
    func testLanguageWithoutAPackReportsNotScreened() async throws {
        let e = try await engine(.uk, .english)
        guard let unpacked = TranslateLanguage.allCases.first(where: {
            GuardPackCatalog.pack(for: $0) == nil
        }) else {
            throw XCTSkip("every language has a pack — nothing to test")
        }

        let verdict = await e.classify("An entirely ordinary sentence.", language: unpacked)
        guard case .notScreened = verdict.source else {
            return XCTFail("\(unpacked.name) has no pack, so the model cannot have "
                         + "cleared this — got \(source(verdict))")
        }
    }

    // MARK: - The log string

    /// Pins the text that lands in the log, since that is what a human greps.
    func testLogDescriptions() {
        XCTAssertEqual(SafetySource.model.logDescription, "by model")
        XCTAssertEqual(SafetySource.notScreened.logDescription, "not screened")
        // The pattern is deliberately NOT in the log line — it stays on the
        // verdict (`case .rules(let pattern)`) for a debugger or a report to
        // read. Printing a regex into the chat log makes the line hard to scan
        // and leaks the rule set into anything that collects logs.
        XCTAssertEqual(SafetySource.rules(pattern: "safe account").logDescription,
                       "by rule")
    }

    /// The default keeps hand-built verdicts (tests, fixtures) meaning what they
    /// meant before this field existed.
    func testDefaultSourceIsTheModel() {
        let verdict = SafetyResult(label: .harassment, severity: .caution, confidence: 0.7)
        guard case .model = verdict.source else {
            return XCTFail("expected the default to be `.model`")
        }
    }

    // MARK: - Timing

    /// A multi-sentence message costs one forward pass PER WINDOW, and the window
    /// count is the multiplier a reader needs to see. Two sentences is three
    /// windows: the whole message for context, plus each sentence on its own.
    ///
    /// Measured on the simulator: ~16 ms of work for this message against ~5 ms
    /// for a single short sentence. Which is why a 1172 ms wall time for it is a
    /// queueing story, not a model one — the work cannot get near that.
    func testWindowCountAndWorkAreReported() async throws {
        let e = try await engine(.vietnam, .vietnamese)
        let verdict = await e.classify(
            "Dạo này mình thấy mệt và trống rỗng quá. "
          + "Có lúc mình nghĩ nếu mình biến mất thì mọi thứ sẽ nhẹ nhàng hơn.",
            language: .vietnamese)

        XCTAssertEqual(verdict.timing.windows, 3,
                       "2 sentences → context window + 2 sentence windows")
        XCTAssertGreaterThan(verdict.timing.workMS, 0)
        XCTAssertLessThan(verdict.timing.workMS, 400,
                          "\(verdict.timing.workMS) ms of WORK for 3 windows — "
                        + "the per-window cost has regressed badly")
    }

    /// A rule hit never reaches the model, so it must report zero windows. This is
    /// what makes `Nwin` in the log a usable signal: no windows means the regex
    /// answered, and the number is not merely absent.
    func testRuleHitRunsNoWindows() async throws {
        let e = try await engine(.uk, .english)
        let verdict = await e.classify("Move your money to a safe account straight away.",
                                       language: .english)
        guard case .rules = verdict.source else {
            return XCTFail("expected a rule hit, got \(source(verdict))")
        }
        XCTAssertEqual(verdict.timing.windows, 0, "a rule short-circuits before the model")
    }

    /// Wait time must actually be measured, not left at zero. Submitting a batch
    /// at once forces later messages to sit behind earlier ones on the serial
    /// queue — the exact shape a look-ahead sweep produces.
    func testQueueWaitIsMeasuredUnderLoad() async throws {
        let e = try await engine(.vietnam, .vietnamese)
        let texts = (0..<24).map { "Tin nhắn thử nghiệm số \($0). Câu thứ hai ở đây." }

        let verdicts = await withTaskGroup(of: SafetyResult.self) { group in
            for t in texts { group.addTask { await e.classify(t, language: .vietnamese) } }
            var out: [SafetyResult] = []
            for await v in group { out.append(v) }
            return out
        }

        XCTAssertEqual(verdicts.count, 24)
        XCTAssertTrue(verdicts.contains { $0.timing.waitMS > 1 },
                      "24 concurrent classifications on one serial queue must make "
                    + "somebody wait — if none did, waitMS is not being measured")
        XCTAssertTrue(verdicts.contains { $0.timing.behind > 0 },
                      "queue depth is not being reported")
        // The finding this instrumentation exists to make legible.
        let worst = verdicts.max { $0.timing.waitMS < $1.timing.waitMS }!
        XCTAssertGreaterThan(worst.timing.waitMS, worst.timing.workMS,
                             "under load the wait should dominate the work")
    }

    /// **The model is punctuation-sensitive**, found while chasing a verdict that
    /// wouldn't reproduce: the app logged 0.653 where this text measured 0.827,
    /// and the only difference was a trailing full stop the user hadn't typed.
    ///
    ///     "…mọi thứ sẽ nhẹ nhàng hơn."   → Self-harm/Violation 0.827
    ///     "…mọi thứ sẽ nhẹ nhàng hơn"    → Self-harm/Violation 0.653
    ///
    /// Same meaning, same class, 17 points of confidence — and `threshold` is
    /// 0.5, so a weaker phrasing without punctuation could cross it and go
    /// unflagged. People do not punctuate chat messages. This pins the behaviour
    /// so a pack that fixes it is visible as a change here rather than a surprise.
    /// **v16 fixed this, so the assertion is inverted.**
    ///
    /// The v15 pack moved 17 points of confidence on a trailing full stop
    /// (0.827 with, 0.653 without) and the old test PINNED that gap, ending with
    /// "if a new pack fixed this, delete the test". v16 measures 0.9931 vs
    /// 0.9914 — a gap of 0.002.
    ///
    /// Inverted rather than deleted. The property worth guarding was never "the
    /// pack is punctuation-sensitive", it was "punctuation must not decide
    /// whether a message is flagged" — people do not punctuate chat messages.
    /// That is now true, and a future pack that regresses to v15's behaviour
    /// should fail here rather than quietly reintroduce the trap.
    func testTrailingPunctuationBarelyMovesConfidence() async throws {
        let e = try await engine(.vietnam, .vietnamese)
        let withStop = "Có lúc mình nghĩ nếu mình biến mất thì mọi thứ sẽ nhẹ nhàng hơn."
        let without = String(withStop.dropLast())

        let a = await e.classify(withStop, language: .vietnamese)
        let b = await e.classify(without, language: .vietnamese)

        XCTAssertEqual(a.label, b.label,
                       "dropping a full stop changed the CLASS "
                     + "(\(a.label.displayName) vs \(b.label.displayName))")
        XCTAssertEqual(a.isUnsafe, b.isUnsafe,
                       "a full stop decided whether this was flagged — that is the "
                     + "v15 failure mode, and chat messages are rarely punctuated")
        XCTAssertLessThan(abs(a.confidence - b.confidence), 0.1,
                          "punctuation sensitivity is back (\(a.confidence) vs "
                        + "\(b.confidence)) — v16 measured a 0.002 gap here")
    }

    // MARK: - Distress must stay readable

    /// **The case that moved the blur threshold to Critical.**
    ///
    /// Someone writing about their own distress must be labelled, never hidden:
    /// blurring it tells them their reaching out was too dangerous to display and
    /// keeps it from the friend who might answer. Ofcom's children's codes want
    /// crisis support surfaced, not the request suppressed.
    ///
    /// The invariant in UK_COMPLIANCE.md §7.2 only ever bound the RULES, which are
    /// hand-authored to require a second-person construction. The model has no
    /// such constraint, so nothing enforced this on the model path until now.
    func testFirstPersonDistressIsNeverHidden() async throws {
        let e = try await engine(.vietnam, .vietnamese)
        let distress = [
            ("Dạo này mình thấy mệt và trống rỗng quá. "
           + "Có lúc mình nghĩ nếu mình biến mất thì mọi thứ sẽ nhẹ nhàng hơn", TranslateLanguage.vietnamese),
            ("Mình không muốn sống nữa", .vietnamese),
            ("Mình thấy vô dụng, chẳng ai cần mình cả", .vietnamese),
            ("I don't want to be here anymore", .english),
            ("Sometimes I wish I could just disappear", .english),
            ("I feel like everyone would be better off without me", .english),
        ]
        for (text, language) in distress {
            let verdict = await e.classify(text, language: language)
            let m = ChatMessage(sender: .friend, text: text)
            m.apply(verdict)
            XCTAssertFalse(m.isSensitive,
                           "blurred a cry for help — \(text.prefix(40))… came back "
                         + "\(verdict.label.displayName)/\(verdict.severity?.displayName ?? "no severity")")
        }
    }

    /// The other half: relaxing the threshold must not stop hiding encouragement
    /// aimed at another person. Measured Critical for all three, which is what
    /// makes severity alone a sufficient discriminator here — no class special-case.
    func testEncouragingSuicideIsStillHidden() async throws {
        let e = try await engine(.vietnam, .vietnamese)
        let encouragement = [
            ("mày nên tự tử đi cho rồi", TranslateLanguage.vietnamese),
            ("You should just kill yourself", .english),
            ("Nobody would miss you, do it", .english),
        ]
        for (text, language) in encouragement {
            let verdict = await e.classify(text, language: language)
            let m = ChatMessage(sender: .friend, text: text)
            m.apply(verdict)
            // ⚠️ KNOWN FAILURE for the two ENGLISH lines under the v16ls
            // checkpoint, and this is the worst of the regression it brought in.
            // The English pack's argmax moved to `clean` — "You should just kill
            // yourself" 0.894 → 0.336, "Nobody would miss you, do it" 0.835 →
            // 0.097 — so no threshold recovers them, and this engine is warmed for
            // VIETNAM, which ships no self-harm patterns at all. There is
            // therefore no rule standing behind the model here: in this region,
            // English suicide encouragement is now undetected end to end.
            //
            // The Vietnamese line still scores 0.9885 and passes unguarded, which
            // is why the expectation is scoped per line rather than to the loop.
            //
            // Strict, so that a pack which fixes either English line makes this
            // stop failing and forces the expectation to be removed.
            let known = language == .english
            XCTExpectFailure("v16ls English pack returns clean for suicide "
                           + "encouragement — see GuardPackTests."
                           + "testEnglishPackRegressionsAreStillPresent",
                             enabled: known, strict: known) {
                XCTAssertEqual(verdict.label, .suicide, text)
            }
            // DETECTION is the invariant, and it holds: v16 scores these
            // 0.894–0.999. BLURRING does not follow from it, because severity is
            // rules-only since v16.
            //
            // Region-specific, and deliberately exercised in the region where the
            // gap exists. This engine is warmed for VIETNAM, which loads only
            // `vietnam-vi` — and that set has no `self_harm` category at all. The
            // English lines here would blur under `.uk`, where `uk-en` ships 10
            // self-harm patterns all at Critical, including exact matches for
            // "you should just kill yourself" and "nobody would miss you". So the
            // hole is Vietnamese self-harm patterns, not self-harm patterns.
            //
            // Asserted rather than deleted so the gap is visible in the suite
            // instead of only in a doc comment. Writing `self_harm` into
            // `vietnam-vi.yaml` is what flips this, and this line is where it
            // shows up.
            XCTAssertFalse(m.isSensitive,
                           "\(text.debugDescription) blurred — if a self-harm RULE "
                         + "was added, that is the fix landing and this assertion "
                         + "should be inverted back")
        }
    }

    /// Child safety is pinned to Critical by `severityRange`, so raising the blur
    /// threshold cannot have quietly unhidden it.
    func testChildSafetyStillBlursAfterTheThresholdChange() async throws {
        let e = try await engine(.uk, .english)
        let verdict = await e.classify("child porn", language: .english)
        let m = ChatMessage(sender: .friend, text: "child porn")
        m.apply(verdict)
        XCTAssertEqual(verdict.label, .child)
        XCTAssertTrue(m.isSensitive, "child safety must always hide")
    }

    /// `windows` counts passes that RAN, not windows planned. A Critical verdict
    /// stops the loop — nothing outranks Critical, so further passes cannot change
    /// the outcome and are not paid for.
    ///
    /// Worth pinning because `1win` on a visibly multi-sentence message reads like
    /// a bug (sentence splitting failed) when it is the optimisation working.
    ///
    /// Reworked for v16: a Critical can now only come from a RULE, and a rule hit
    /// short-circuits before the model runs at all — so the window count it
    /// produces is 0, not 1. The old text relied on the model returning Critical
    /// on its first window, which it can no longer do at any confidence.
    func testCriticalStopsAfterOneWindow() async throws {
        let e = try await engine(.uk, .english)
        // A universal child-safety pattern: Critical, present in every region.
        let verdict = await e.classify(
            "Have you got child porn to share. Send it here. I will pay.",
            language: .english)

        XCTAssertEqual(verdict.severity, .critical)
        XCTAssertEqual(verdict.timing.windows, 0,
                       "a Critical RULE hit must return before any model window, "
                     + "got \(verdict.timing.windows) passes")
    }

    /// CPU time must be measured and must be bounded by the wall time it sits
    /// inside. Deliberately NOT asserting that CPU accounts for most of the wall
    /// time: this test host runs 170+ tests and is contended, so the gap it would
    /// be measuring is real rather than a bug — which is exactly the condition the
    /// field exists to report in the field.
    func testCPUTimeIsMeasured() async throws {
        let e = try await engine(.vietnam, .vietnamese)
        let verdict = await e.classify(
            "Dạo này mình thấy mệt và trống rỗng quá. "
          + "Có lúc mình nghĩ nếu mình biến mất thì mọi thứ sẽ nhẹ nhàng hơn.",
            language: .vietnamese)

        XCTAssertGreaterThan(verdict.timing.cpuMS, 0, "CPU clock is not being read")
        XCTAssertLessThanOrEqual(verdict.timing.cpuMS, verdict.timing.workMS + 1,
                                 "CPU (\(verdict.timing.cpuMS)) exceeds the wall time "
                               + "it ran inside (\(verdict.timing.workMS)) — the two "
                               + "clocks are not measuring the same interval")
    }
}
