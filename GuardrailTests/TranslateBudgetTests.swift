import XCTest
@testable import SLMChat

/// Translation's token budget: every chunk's translation must fit its output cap,
/// and the whole message must fit the hard maximum (`ChatService.translate*`).
///
/// The regression: a chunk could carry 720 tokens of source while its output was
/// capped at 256, so any English text over ~180 tokens was cut off in Vietnamese
/// (which takes 1.3-1.4x the tokens). The cut lands at the END of the message, so
/// these assert on facts from the last sentence — numbers survive translation.
final class TranslateBudgetTests: XCTestCase {

    /// 222 tokens of English; its Vietnamese is 282 (Qwen3.5) / 315 (Qwen2.5).
    private let paragraph = """
        The UEFA Euro 2016 final was the concluding match of UEFA Euro 2016, the fifteenth edition of \
        the European Championship, UEFA's quadrennial competition for men's national football teams. \
        The match was played before 75,868 spectators at the Stade de France (pictured) in Paris on 10 \
        July 2016 between Portugal and hosts France. Following a goalless 90 minutes, which saw Portugal \
        captain Cristiano Ronaldo depart the match through injury, Portugal then took the lead in the \
        108th minute through substitute Eder after his low shot beat France goalkeeper Hugo Lloris. This \
        proved to be the contest's only goal, as Portugal won 1–0 to claim their first major tournament \
        title. In winning the final, Portugal became the tenth different nation to win the European \
        Championship, twelve years after losing their first final at home in UEFA Euro 2004. As the \
        winners, Portugal gained entry into their first FIFA Confederations Cup, which was played in \
        Russia in 2017.
        """

    /// Over one chunk but under the maximum (~500 tokens): the paragraph plus a
    /// second one whose last sentence carries the only "2021" in the text.
    private var twoChunks: String {
        paragraph + " " + """
        France had reached the final after beating Germany 2–0 in the semi-final in Marseille, with \
        Antoine Griezmann scoring both goals and finishing as the tournament's top scorer with six. \
        Portugal had drawn all three of their group matches and only advanced as one of the best \
        third-placed teams, before beating Croatia, Poland and Wales on the way to the final. Fernando \
        Santos, the Portugal coach, said afterwards that his team had been written off by many but had \
        never stopped believing in itself. Ronaldo, who had left the pitch in tears in the first half, \
        spent extra time on the touchline with a strapped knee, shouting instructions at his teammates \
        and embracing the coaching staff. Back in Lisbon, hundreds of thousands of supporters filled the \
        streets through the night, and the squad was received by the President the following afternoon \
        before an open-top bus parade. In France, the defeat on home soil was a bitter end to a \
        tournament held under heavy security after the attacks in Paris the previous November. The next \
        edition of the tournament was postponed by a year and finally held across eleven cities in 2021.
        """
    }

    private func service(_ option: SLMModelOption) async throws -> ChatService {
        let svc = ChatService()
        svc.taskModelResourceName = option.resourceName
        svc.taskModelID = option.id
        guard await svc.warmUpIfNeeded().ok else { throw XCTSkip("\(option.id) failed to load") }
        return svc
    }

    private func translate(_ text: String, with svc: ChatService) async -> String {
        var last = ""
        for await partial in svc.translateLongStream(text, to: .vietnamese) { last = partial }
        return last
    }

    /// prompt overhead + one chunk + its output must fit the context, or the
    /// generation fails outright when it runs into the ceiling.
    func testBudgetFitsContext() {
        let svc = ChatService()
        XCTAssertLessThanOrEqual(
            ChatService.translatePromptOverheadTokens + svc.translateChunkInputTokens
                + Int(svc.translateChunkTokens),
            Int(ChatLM.contextTokens()))
        XCTAssertEqual(svc.translateMaxInputTokens, 2 * svc.translateChunkInputTokens)
    }

    func testParagraphTranslatesToTheEnd() async throws {
        var ran = 0
        for option in DesktopModels.taskModels where option.isAvailable {
            let svc = try await service(option)
            XCTAssertEqual(svc.chunkForTranslation(paragraph).count, 1, "\(option.id)")
            let out = await translate(paragraph, with: svc)
            print("TRANSBUDGET [\(option.id)] 1 chunk → \(out.count) chars: …\(out.suffix(90))")
            XCTAssertTrue(out.contains("2017"), "\(option.id): last sentence missing — cut off? \(out.suffix(160))")
            ran += 1
        }
        try XCTSkipIf(ran == 0, "No .gguf in the bundle.")
    }

    func testTwoChunksTranslateToTheEnd() async throws {
        var ran = 0
        for option in DesktopModels.taskModels where option.isAvailable {
            let svc = try await service(option)
            XCTAssertFalse(svc.isTooLongToTranslate(twoChunks), "\(option.id): fixture is over the maximum")
            let chunks = svc.chunkForTranslation(twoChunks)
            XCTAssertGreaterThan(chunks.count, 1, "\(option.id): fixture fits one chunk — not testing the split")
            let out = await translate(twoChunks, with: svc)
            print("TRANSBUDGET [\(option.id)] \(chunks.count) chunks → \(out.count) chars: …\(out.suffix(90))")
            XCTAssertTrue(out.contains("2017"), "\(option.id): first chunk cut off: \(out)")
            XCTAssertTrue(out.contains("2021"), "\(option.id): last chunk cut off: \(out.suffix(160))")
            ran += 1
        }
        try XCTSkipIf(ran == 0, "No .gguf in the bundle.")
    }

    /// Over the maximum: refused up front, and the stream yields nothing rather
    /// than translating part of it.
    func testOverTheMaximumIsRefused() async throws {
        guard let option = DesktopModels.taskModels.first(where: \.isAvailable) else {
            throw XCTSkip("No .gguf in the bundle.")
        }
        let svc = try await service(option)
        let tooLong = String(repeating: paragraph + " ", count: 4)
        XCTAssertTrue(svc.isTooLongToTranslate(tooLong))
        XCTAssertFalse(svc.isTooLongToTranslate(paragraph))
        XCTAssertFalse(svc.isTooLongToTranslate("ok, see you at 7"))
        let out = await translate(tooLong, with: svc)
        XCTAssertEqual(out, "")
    }
}
