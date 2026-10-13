import XCTest
@testable import SLMChat

/// Checks that group-chat messages are screened at all.
///
/// They weren't, until this. `GroupChat` kept its own `messages` array and never
/// called the guardrail on either append path — not what the user sent, not what
/// the bots posted. The failure was invisible by construction: `GroupChatView`
/// renders through the same `MessageBubble` as a 1:1 chat, so an unscreened
/// message looked exactly like one that had been screened and come back clean.
///
/// These tests exist mainly so that can't come back silently. A regression here
/// would otherwise show up only as "the group never flags anything", which is
/// indistinguishable from a well-behaved group.
@MainActor
final class GroupScreeningTests: XCTestCase {

    private var service: ChatService!

    override func setUp() async throws {
        try await super.setUp()
        service = ChatService()
        // Load the UK rules explicitly. Without this the engine runs MODEL-ONLY:
        // the pack still loads lazily, so messages are screened and verdicts come
        // back — they are just missing every rule-based detection, and a test
        // that asserts on rule hits fails for a reason that has nothing to do
        // with what it is testing. (`ChatService.warmUp` would do this, but it
        // also loads the 8-second SLM, which no test here needs.)
        let res = await service.guardEngine.warmUp(language: .english,
                                                   region: .uk)
        XCTAssertTrue(res.ok, "guard warm-up failed: \(res.error ?? "unknown")")
        XCTAssertGreaterThan(service.guardEngine.ruleCount, 0, "UK rules did not load")
    }

    override func tearDown() {
        service = nil
        super.tearDown()
    }

    private func group(_ lines: [(String, String?)] = []) -> GroupChat {
        let g = GroupChat(service: service)
        for (text, author) in lines {
            g.messages.append(ChatMessage(sender: author == nil ? .you : .friend,
                                          text: text, authorName: author))
        }
        return g
    }

    private func waitForScreening(_ g: GroupChat, count: Int,
                                  timeout: TimeInterval = 10) async -> Int {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            let done = g.messages.filter { $0.safetyLabel != nil }.count
            if done >= count { return done }
            try? await Task.sleep(nanoseconds: 50_000_000)
        }
        return g.messages.filter { $0.safetyLabel != nil }.count
    }

    /// The bug this file was written for: a flagged message posted into a group
    /// must actually come back flagged.
    func testGroupMessagesAreScreened() async throws {
        let g = group([("I know where you work.", "Mai")])
        await g.classifySafety(in: g.messages[0])

        XCTAssertEqual(g.messages[0].safety, .harassment,
                       "group messages are not being screened")
        XCTAssertTrue(g.messages[0].safety?.isUnsafe == true)
    }

    /// A verdict has to be attributable to a member, not just to "not me".
    /// In a 1:1 chat `sender` is enough; in a group it is not, and a per-user
    /// violation signal is built from `authorName`.
    func testFlaggedGroupMessageKeepsItsAuthor() async throws {
        let g = group([("all good here", "Sam"),
                       ("I know where you work.", "Mai")])
        for m in g.messages { await g.classifySafety(in: m) }

        let flagged = g.messages.filter { $0.safety?.isUnsafe == true }
        XCTAssertEqual(flagged.count, 1)
        XCTAssertEqual(flagged.first?.authorName, "Mai",
                       "the flag must name the member who sent it")
        XCTAssertEqual(g.messages.first(where: { $0.authorName == "Sam" })?.safety, .clean,
                       "a bystander must not be flagged")
    }

    /// Viewport screening works the same way it does in a 1:1 chat: bulk history
    /// waits until it is looked at.
    func testGroupViewportScreeningDefersBacklog() async throws {
        let g = group((0..<40).map { ("Group backlog message \($0).", "Sam") })

        g.screenFromViewport(g.messages[g.messages.count - 1])
        _ = await waitForScreening(g, count: GroupChat.screenLookAhead + 1)
        try? await Task.sleep(nanoseconds: 300_000_000)

        XCTAssertLessThanOrEqual(g.screeningRuns, GroupChat.screenLookAhead + 1,
                                 "ran \(g.screeningRuns) classifications for one viewport")
        XCTAssertGreaterThan(g.messages.filter { $0.safetyLabel == nil }.count, 0,
                             "nothing was deferred")
    }

    /// Live and viewport paths must not both classify the same message — the same
    /// gate `Conversation` needed, for the same reason.
    func testGroupPathsDoNotDoubleScreen() async throws {
        let g = group([("Let's meet tomorrow at 3pm.", "Linh")])
        let m = g.messages[0]

        Task { await g.classifySafety(in: m) }
        g.screenFromViewport(m)
        _ = await waitForScreening(g, count: 1)
        try? await Task.sleep(nanoseconds: 300_000_000)

        XCTAssertEqual(g.screeningRuns, 1,
                       "classified \(g.screeningRuns) times — the in-flight gate leaked")
    }

    /// The group scenarios must behave as scripted, including the bystanders.
    ///
    /// The clean lines carry the weight here: Linh defending the target and Sam
    /// intervening must not be flagged. A guardrail that flags the people
    /// stepping in makes a pile-on worse, and that is the most likely way this
    /// scenario silently stops being worth demoing.
    func testGroupScenariosBehaveAsAdvertised() async throws {
        let engine = GuardEngine()
        let res = await engine.warmUp(language: .english,
                                      region: .uk)
        XCTAssertTrue(res.ok, "warm-up failed: \(res.error ?? "unknown")")

        for scenario in [GuardrailDemo.groupPileOn, GuardrailDemo.groupScam] {
            for line in scenario {
                let verdict = await engine.classify(line.text, language: .english)
                XCTAssertEqual(verdict.isUnsafe, line.shouldFlag,
                               "\(line.author ?? "You"): \(line.text.debugDescription) → "
                             + "\(verdict.label.displayName)/"
                             + "\(verdict.severity?.displayName ?? "no severity") "
                             + "— note says: \(line.expect)")
            }
        }
    }

    /// The pile-on must concentrate on one author while the others stay clean.
    /// That separation IS the per-user signal; if every member accrues flags, the
    /// tally carries no information.
    func testPileOnConcentratesOnOneAuthor() async throws {
        let engine = GuardEngine()
        _ = await engine.warmUp(language: .english, region: .uk)

        var flagged: [String: Int] = [:]
        for line in GuardrailDemo.groupPileOn {
            let verdict = await engine.classify(line.text, language: .english)
            if verdict.isUnsafe { flagged[line.author ?? "You", default: 0] += 1 }
        }

        XCTAssertEqual(flagged.keys.count, 1,
                       "flags landed on \(flagged.keys.sorted()) — a pile-on demo "
                     + "only means something if the bystanders stay clean")
        XCTAssertEqual(flagged["Mai"], 3, "expected Mai to carry every flag")
        XCTAssertNil(flagged["You"], "the TARGET must never be flagged — they are "
                                   + "distressed, not violating")
    }
}
