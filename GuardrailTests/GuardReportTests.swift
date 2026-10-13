import XCTest
@testable import SLMChat

/// Checks the one path in this design where a verdict may leave the device.
///
/// Everything else the guardrail produces stays local. A report is different in
/// kind: it is an assertion a human made about another person's conduct, and it
/// is the only signal that could reach a durable record about them. So the gates
/// on it are the tested part, not the plumbing.
@MainActor
final class GuardReportTests: XCTestCase {

    private var service: ChatService!

    override func setUp() {
        super.setUp()
        service = ChatService()
    }

    override func tearDown() {
        service = nil
        super.tearDown()
    }

    private func message(_ label: SafetyLabel, _ severity: Severity,
                         author: String? = nil) -> ChatMessage {
        let m = ChatMessage(sender: .friend, text: "reported content")
        m.authorName = author
        m.apply(SafetyResult(label: label, severity: severity, confidence: 0.9))
        return m
    }

    // MARK: - No severity gate

    /// **Any severity is reportable, including none.**
    ///
    /// This used to require Critical. The radicalisation scenario is why it
    /// doesn't: the classifier catches 2 of its 8 lines and returns Clean for the
    /// recruitment ("Cần tổ chức ngay. Ai dám tham gia thì khai tên ra." 0.916)
    /// and the purge-organising — so a Critical-only gate refused a report on
    /// exactly the six lines a person would want escalated.
    func testAnySeverityCanBeReported() throws {
        for severity in Severity.allCases {
            XCTAssertNotNil(GuardReport.submit(message(.harassment, severity),
                                               reason: .harassment,
                                               author: "Mai", context: "test"),
                            "\(severity.displayName) was refused")
        }
    }

    /// A message the guardrail never flagged is reportable, and the payload says
    /// so — `reasonMatchesVerdict` is nil rather than a false agreement, and
    /// `isMissedByGuardrail` marks it for review.
    func testUnflaggedMessageIsReportableAndMarkedAsAMiss() throws {
        let bare = ChatMessage(sender: .friend, text: "Cần tổ chức ngay.")
        let payload = try XCTUnwrap(GuardReport.submit(bare, reason: .violence,
                                                       author: "Mai", context: "test"),
                                    "an unflagged message must still be reportable — "
                                  + "the guardrail's misses are what need a human")
        XCTAssertNil(payload.contentClass)
        XCTAssertNil(payload.reasonMatchesVerdict,
                     "there was no verdict, so there is nothing to agree with")
        XCTAssertTrue(payload.isMissedByGuardrail)
        XCTAssertEqual(payload.reason, .violence)
    }

    /// A message screened and found Clean counts as a miss too — "screened, found
    /// nothing" and "a human disagreed" is the pair worth reviewing.
    func testCleanVerdictAlsoCountsAsAMiss() throws {
        let clean = ChatMessage(sender: .friend, text: "x")
        clean.apply(SafetyResult(label: .clean, severity: .safe, confidence: 0.9))
        let payload = try XCTUnwrap(GuardReport.submit(clean, reason: .harassment,
                                                       author: "Mai", context: "t"))
        XCTAssertTrue(payload.isMissedByGuardrail)
        XCTAssertNil(payload.reasonMatchesVerdict)
    }

    // MARK: - What the payload carries

    /// The payload must NOT carry the message text. The log escapes the device
    /// through sysdiagnose and crash reporters, and printing flagged content into
    /// it leaks exactly what the guardrail just decided to withhold.
    func testPayloadCarriesTheVerdictAndTheAuthorButNotTheText() throws {
        let m = message(.child, .critical, author: "Mai")
        let payload = try XCTUnwrap(GuardReport.submit(m, reason: .child,
                                                       note: "sent to the whole group",
                                                       author: "Mai",
                                                       context: "Group: Group"))
        XCTAssertEqual(payload.contentClass, .child)
        XCTAssertEqual(payload.severity, .critical)
        XCTAssertEqual(payload.reportedAuthor, "Mai",
                       "a group report that cannot name WHO is not actionable")

        let mirror = Mirror(reflecting: payload)
        for child in mirror.children {
            XCTAssertFalse("\(child.value)".contains(m.text),
                           "payload field \(child.label ?? "?") carries the message text")
        }
    }

    // MARK: - Persistence

    /// `reported` survives a save/load round trip. A report forgotten across a
    /// relaunch puts the button back with no trace the user already acted, which
    /// invites reporting the same message twice — and a double report inflates
    /// the counter enforcement would be based on.
    func testReportedFlagSurvivesPersistence() throws {
        let m = message(.violence, .critical)
        m.reported = true

        let data = try JSONEncoder().encode(m.snapshot)
        let restored = ChatMessage(try JSONDecoder().decode(StoredMessage.self, from: data))
        XCTAssertTrue(restored.reported)
    }

    /// A save written before reporting existed must still load, with `reported`
    /// reading false rather than failing to decode.
    func testOlderSavesWithoutTheKeyStillLoad() throws {
        let json = Data("""
        {"sender":"friend","date":0,"text":"hi","isRead":true}
        """.utf8)
        let stored = try JSONDecoder().decode(StoredMessage.self, from: json)
        XCTAssertNil(stored.reported)
        XCTAssertFalse(ChatMessage(stored).reported)
    }

    // MARK: - Wiring

    /// Reporting through a conversation marks the message and persists it.
    func testConversationReportMarksAndPersists() {
        let m = message(.violence, .critical)
        let convo = Conversation(contactName: "Test", messages: [m], service: service)
        var persisted = 0
        convo.onPersist = { persisted += 1 }

        m.reported = true          // the button sets this, then calls report
        convo.report(m, reason: .violence)

        XCTAssertTrue(m.reported)
        XCTAssertEqual(persisted, 1,
                       "a report must be written straight away, not coalesced — "
                     + "unlike a verdict, it cannot be recomputed")
    }

    // MARK: - The reason the USER chose

    /// The reason is pre-filled from the verdict but owned by the reporter, and a
    /// disagreement is flagged rather than quietly normalised. It is a labelled
    /// example of the model being wrong, volunteered by someone who read the
    /// message — the most useful thing a report can carry for improving a pack.
    func testDisagreementWithTheClassifierIsRecorded() throws {
        let m = message(.harassment, .critical)

        let agreeing = try XCTUnwrap(GuardReport.submit(m, reason: .harassment,
                                                        author: "Mai", context: "t"))
        XCTAssertEqual(agreeing.reasonMatchesVerdict, true)

        let correcting = try XCTUnwrap(GuardReport.submit(m, reason: .scam,
                                                          author: "Mai", context: "t"))
        XCTAssertEqual(correcting.reasonMatchesVerdict, false,
                       "the reporter said scam where the model said harassment — "
                     + "that correction must survive into the payload")
        XCTAssertEqual(correcting.reason, .scam,
                       "the submitted reason is the USER's, never overwritten by "
                     + "the model's suggestion")
    }

    /// Every reportable class has a pre-selected reason, so the common case is
    /// confirm-and-send rather than hunting a dropdown.
    func testEveryUnsafeClassSuggestsAReason() {
        for label in SafetyLabel.allCases where label.isUnsafe && label != .political {
            XCTAssertNotEqual(ReportReason.suggested(for: label), .other,
                              "\(label.displayName) falls through to 'Something else' — "
                            + "the reporter has to find the reason themselves")
        }
    }

    /// **`political` must never become reportable.**
    ///
    /// Vietnam ships 114 political patterns and the UK ships zero, precisely
    /// because political criticism is protected expression (UK_COMPLIANCE.md §3).
    /// A "political content" reason would hand every user a button for silencing
    /// dissent, and no downstream review makes that safe.
    func testPoliticalIsNotAReportableReason() {
        XCTAssertFalse(ReportReason.allCases.contains { $0.rawValue.contains("political") })
        XCTAssertEqual(ReportReason.suggested(for: .political), .other,
                       "political must fall through to 'Something else', never gain "
                     + "a reason of its own")
    }

    /// Notes are capped and trimmed. Unbounded free text about a message is the
    /// easiest way for content to reach a place this design promised it wouldn't.
    func testNoteIsTrimmedAndCapped() throws {
        let m = message(.violence, .critical)
        let long = String(repeating: "x", count: GuardReport.maxNoteLength + 200)
        let payload = try XCTUnwrap(GuardReport.submit(m, reason: .violence,
                                                       note: "  \(long)  ",
                                                       author: "Mai", context: "t"))
        XCTAssertEqual(payload.note.count, GuardReport.maxNoteLength)

        let blank = try XCTUnwrap(GuardReport.submit(m, reason: .violence, note: "   \n ",
                                                     author: "Mai", context: "t"))
        XCTAssertTrue(blank.note.isEmpty)
    }
}
