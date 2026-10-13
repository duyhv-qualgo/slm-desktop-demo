import XCTest
@testable import SLMChat

/// Checks that content the guardrail hides is not printed somewhere else.
///
/// The blur in `MessageBubble` is only one of several places message text
/// reaches a screen, and it was for a while the only one that respected the
/// verdict. A guardrail that hides a message in the thread while the inbox row
/// above prints it in full has done nothing except make the leak harder to
/// notice — the flagged text appears on a screen the user never even opened.
///
/// Every future surface that renders message text — notifications, search
/// results, widgets, share sheets — needs a test here.
@MainActor
final class ContentLeakTests: XCTestCase {

    private var service: ChatService!

    override func setUp() {
        super.setUp()
        service = ChatService()
    }

    override func tearDown() {
        service = nil
        super.tearDown()
    }

    private func conversation(_ messages: [ChatMessage]) -> Conversation {
        Conversation(contactName: "Test", messages: messages, service: service)
    }

    private func flagged(_ text: String,
                         _ label: SafetyLabel,
                         _ severity: Severity) -> ChatMessage {
        let m = ChatMessage(sender: .friend, text: text)
        m.apply(SafetyResult(label: label, severity: severity, confidence: 1))
        return m
    }

    /// The leak this file was written for.
    func testInboxPreviewDoesNotPrintHiddenContent() {
        let secret = "I've got your nudes. Pay me £500 or I'll send them to your contacts."
        let convo = conversation([flagged(secret, .sexual, .critical)])

        let preview = convo.lastMessagePreview
        XCTAssertFalse(preview.contains("nudes"), "inbox preview leaked hidden content")
        XCTAssertFalse(preview.contains("£500"), "inbox preview leaked hidden content")
        XCTAssertTrue(preview.contains("Sexual"), "preview should say what was hidden")
        XCTAssertTrue(preview.contains("Critical"))
    }

    /// Caution is labelled, not hidden — so its preview must NOT be substituted.
    /// Over-hiding is its own failure: an inbox of warning triangles where the
    /// thread shows readable text is both wrong and untrustworthy.
    func testCautionMessagesStillPreviewNormally() {
        let text = "they're banning zombie knives finally"
        let convo = conversation([flagged(text, .illegal, .caution)])

        XCTAssertEqual(convo.lastMessagePreview, text,
                       "a Caution message is readable in the thread and must "
                     + "stay readable in the inbox")
    }

    /// Clean and unscreened messages preview normally. Unscreened is deliberate:
    /// the guardrail fails open everywhere, and hiding every not-yet-classified
    /// message would blank the inbox on every launch.
    func testCleanAndUnscreenedMessagesPreviewNormally() {
        let clean = ChatMessage(sender: .friend, text: "See you at 3pm.")
        clean.apply(.clean)
        XCTAssertEqual(conversation([clean]).lastMessagePreview, "See you at 3pm.")

        let unscreened = ChatMessage(sender: .friend, text: "Not classified yet.")
        XCTAssertEqual(conversation([unscreened]).lastMessagePreview, "Not classified yet.")
    }

    /// Revealing inside the thread must not republish to the inbox. The reveal is
    /// a choice made for that screen, in that moment.
    func testRevealingInTheThreadDoesNotUnhideThePreview() {
        let secret = "I know where you work."
        let message = flagged(secret, .harassment, .critical)
        let convo = conversation([message])

        message.revealUnsafe = true

        XCTAssertFalse(convo.lastMessagePreview.contains("work"),
                       "revealing in the thread leaked the message to the inbox")
    }

    /// `isSensitive` is the single definition the bubble and the preview share.
    /// Two copies of this rule would drift silently, and the drift is exactly the
    /// bug above — so pin the boundary.
    ///
    /// Critical only. Violation used to hide content too; see `isSensitive` for
    /// why it stopped — first-person distress landed there and was blurred.
    func testSensitiveBoundaryIsCriticalOnly() {
        let cases: [(Severity, Bool)] = [
            (.safe, false), (.caution, false), (.violation, false), (.critical, true),
        ]
        for (severity, expected) in cases {
            let m = flagged("text", .harassment, severity)
            XCTAssertEqual(m.isSensitive, expected,
                           "\(severity.displayName) should \(expected ? "" : "not ")hide")
        }
        XCTAssertFalse(ChatMessage(sender: .friend, text: "x").isSensitive,
                       "an unscreened message is not sensitive — fail open")
    }

    /// The preview reflects the LAST message, so a flagged message followed by a
    /// clean one must stop being substituted.
    func testPreviewFollowsTheLatestMessage() {
        let convo = conversation([flagged("I know where you work.", .harassment, .critical)])
        XCTAssertTrue(convo.lastMessagePreview.contains("⚠️"))

        let reply = ChatMessage(sender: .you, text: "Blocking you now.")
        reply.apply(.clean)
        convo.messages.append(reply)

        XCTAssertEqual(convo.lastMessagePreview, "You: Blocking you now.")
    }
}
