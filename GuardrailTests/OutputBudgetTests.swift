import XCTest
@testable import SLMChat

// Token budgets: how many tokens each stage produces, and whether the caps
// allow it.
//
// Output tokens are the most expensive thing the app generates (decode is 60-70%
// of a turn and exactly linear in them), and a cap that binds mid-sentence
// corrupts every later stage of a map-reduce.

/// Measures where a turn's OUTPUT tokens actually go.
///
/// Decode is 60-70% of a turn on device and is exactly linear in output tokens, so
/// an output token is the most expensive thing the app produces. Nothing else
/// reports which of them carry information and which are scaffolding the parser
/// throws away.
final class OutputBudgetTests: XCTestCase {

    /// EVERY bundled model, because a token count is a property of the model's
    /// VOCABULARY, not of the string. Gemma-3 (262k vocab) and Qwen3 (151k) split
    /// the same text differently, so measuring one and quoting it for the other is
    /// simply wrong — which an earlier version of this test did.
    private func engines() throws -> [(id: String, lm: ChatLM)] {
        var out: [(String, ChatLM)] = []
        for option in ModelCatalog.taskModels where option.isAvailable {
            guard let path = ResourceLookup.path(option.resourceName, ext: "gguf") else { continue }
            let lm = ChatLM()
            if lm.load(modelPath: path, backend: SLMBackend.cpu, loraPath: nil) {
                out.append((option.id, lm))
            }
        }
        guard !out.isEmpty else { throw XCTSkip("No .gguf in the bundle.") }
        return out
    }

    /// A real auto-suggest output, copied verbatim from a device log.
    private let realOutput = """
    [[CLOSE]]
    Ok, 4:00 gặp nhé!
    [[INQUIRE]]
    Sam, 4:00 gặp ở nhà cậu đó.
    [[REACT]]
    Tuyệt vời, chúc hai bạn có bữa tối vui vẻ nhé!
    """

    func testIntentTagOverheadInSuggestOutput() throws {
        for (id, lm) in try engines() { try report(id: id, lm: lm) }
    }

    private func report(id: String, lm: ChatLM) throws {
        let total = lm.tokenCount(of: realOutput)

        // What the user actually receives: the parser keeps only the text.
        let kept = Prompts.parseSuggestions(realOutput).suggestions
        let keptTokens = kept.reduce(0) { $0 + lm.tokenCount(of: $1) }

        let tags = ["[[CLOSE]]", "[[INQUIRE]]", "[[REACT]]"]
        let tagTokens = tags.reduce(0) { $0 + lm.tokenCount(of: $1) }

        print("""
        BUDGET [\(id)] total=\(total) tok | kept(text)=\(keptTokens) tok \
        | intent tags=\(tagTokens) tok (\(Int(100.0 * Double(tagTokens) / Double(total)))% of output, discarded) \
        | chips=\(kept.count)
        """)

        XCTAssertEqual(kept.count, 3, "all three chips should parse")
        XCTAssertGreaterThan(tagTokens, 0)
    }

    /// The same three replies without the scaffolding — what a leaner output format
    /// would cost.
    func testAlternativeFormatsCost() throws {
        for (id, lm) in try engines() { formats(id: id, lm: lm) }
    }

    private func formats(id: String, lm: ChatLM) {
        let chips = Prompts.parseSuggestions(realOutput).suggestions

        let formats: [(String, String)] = [
            ("current [[INTENT]]", realOutput),
            ("newline-separated", chips.joined(separator: "\n")),
            ("numbered list", chips.enumerated().map { "\($0.offset + 1). \($0.element)" }
                                   .joined(separator: "\n")),
            ("JSON array", "[\(chips.map { "\"\($0)\"" }.joined(separator: ","))]"),
        ]
        for (name, text) in formats {
            print(String(format: "BUDGET [%@] format %-20s = %3d tok", id,
                         (name as NSString).utf8String!, lm.tokenCount(of: text)))
        }
    }
}

final class MapChunkBudgetTests: XCTestCase {

    /// ~600 chars of Vietnamese — one full `longChunkChars` map chunk.
    private let vietnameseChunk = """
    Mai chia sẻ nhật ký 24 giờ của mình, bắt đầu từ lúc thức dậy khoảng sáu giờ     sáng ở Kyoto, rồi đi dạo trong rừng tre Arashiyama suốt buổi sáng và ăn trưa     tại một tiệm mì ramen nhỏ chỉ có sáu chỗ ngồi, nơi mùi nước dùng thơm nồng nàn     và tiếng xì xụp của những nhân viên văn phòng vang lên khắp quán. Buổi chiều     cô lên tàu đi Osaka, ngắm cảnh vật thay đổi dần từ cổ kính sang hiện đại, rồi     đến khu Dotonbori đông đúc với tiếng loa quảng cáo và ánh đèn neon rực rỡ.     Buổi tối cô ghé một quán bar nhỏ nằm dưới tầng hầm, nghe nhạc jazz và nghe     những người lạ kể chuyện đời của họ.
    """

    func testMapChunkFinishesNaturallyAtTheConfiguredBudget() throws {
        guard let option = ModelCatalog.taskModels.first(where: { $0.isAvailable }),
              let path = ResourceLookup.path(option.resourceName, ext: "gguf") else {
            throw XCTSkip("No .gguf in the bundle.")
        }
        let pack = LoRACatalog.adapters(forBaseModel: option.id)
            .first { $0.languages.contains(.vietnamese) }
            .flatMap { LoRACatalog.bundledPath(for: $0) }
        let lm = ChatLM()
        guard lm.load(modelPath: path, backend: SLMBackend.cpu,
                       loraPath: pack) else {
            throw XCTSkip("load failed: \(lm.lastError)")
        }

        let svc = ChatService()
        let prompt = Prompts.summarizeText(vietnameseChunk, language: "Vietnamese", speaker: "Mai")
        let out = lm.generate(system: nil, history: [], user: prompt,
                              maxTokens: svc.summaryMapTokens, temperature: 0.3, adapterPath: nil) ?? ""
        let m = try XCTUnwrap(lm.lastTurnMetrics)
        let outTokens = Int(lm.tokenCount(of: out))

        print("MAPBUDGET in=\(vietnameseChunk.count) chars out=\(outTokens) tok "
              + "(\(out.count) chars) cap=\(svc.summaryMapTokens) stop=\(m.stopReason)")

        XCTAssertEqual(m.stopReason, "eos",
                       "map chunk hit the token cap — its partial summary is truncated, and "
                       + "every later fold will summarise damaged text (budget "
                       + "\(svc.summaryMapTokens))")
        // Headroom check: sitting right on the cap means the next slightly longer
        // chunk truncates, so the budget is only nominally sufficient.
        XCTAssertLessThan(Double(outTokens), Double(svc.summaryMapTokens) * 0.85,
                          "map output is within 15% of the cap — too little headroom")
    }
}

/// Where do chunk boundaries actually fall?
///
/// `chunk` splits on sentence terminators (. ! ? newline) and only falls back to
/// word-level `hardSplit` when ONE sentence exceeds the limit. Comma-spliced prose
/// — a diary entry, a chat message typed in one breath — is exactly that case, so
/// its chunks get cut mid-clause. A map summary of a fragment then ends mid-thought
/// no matter how much token budget it has, which is the failure that survives
/// fixing the cap.
final class ChunkBoundaryTests: XCTestCase {

    /// Vietnamese diary prose: long comma-separated runs, few full stops.
    private let commaSpliced = """
    Mai chia sẻ nhật ký 24h, thức dậy lúc 6 giờ sáng ở Kyoto, pha một tách cà phê     rồi ra ban công ngồi ngắm phố xá còn vắng người, sau đó đạp xe qua mấy con hẻm     nhỏ đến rừng tre Arashiyama lúc 9 giờ 30, đi bộ giữa hai hàng tre cao vút nghe     tiếng gió xào xạc, chụp vài tấm ảnh rồi ghé một tiệm mì ramen nhỏ chỉ có sáu     chỗ ngồi, mùi nước dùng thơm nồng nàn, tiếng xì xụp của mấy nhân viên văn phòng     vang lên khắp quán, ăn xong thì lên tàu đi Osaka lúc 14 giờ 30, ngắm cảnh vật     thay đổi dần từ cổ kính sang hiện đại qua ô cửa sổ, đến Dotonbori thì người     đông đúc, tiếng loa quảng cáo vang dội, đèn neon rực rỡ khắp nơi, tối ghé một     quán bar nhỏ nằm dưới tầng hầm nghe nhạc jazz chill chill.
    """

    /// Character fallback, kept only to document what it does when no model is
    /// loaded. The mid-clause cut it produces is exactly why it is the fallback.
    func testCharacterFallbackStillSplitsOnClauses() {
        let chunks = Prompts.chunk(commaSpliced)
        print("CHUNKBOUND inputChars=\(commaSpliced.count) chunks=\(chunks.count)")
        var midClause = 0
        for (i, c) in chunks.enumerated() {
            let end = c.trimmingCharacters(in: .whitespacesAndNewlines).suffix(1)
            let clean = ".!?".contains(end) 
            if !clean { midClause += 1 }
            print("CHUNKBOUND   [\(i + 1)] \(c.count) chars, ends: …\(c.suffix(34)) \(clean ? "" : "<-- mid-clause")")
        }
        print("CHUNKBOUND midClause=\(midClause) of \(chunks.count)")
        // Diagnostic, not a requirement — a single over-long sentence MUST be split
        // somewhere. This records how often it happens on realistic input.
        XCTAssertGreaterThan(chunks.count, 1)
    }
}

/// A prompt longer than the context must fail cleanly, not corrupt memory.
///
/// Regression test for a real crash: `_batch` holds 1024 tokens and
/// `-decode:count:` used to write `count` entries into it unchecked, so a
/// ~1120-token prompt overran four heap buffers and killed the process with no
/// diagnostic. Found by accident while sweeping prefill throughput.
final class OverlongPromptTests: XCTestCase {

    func testOverlongPromptFailsInsteadOfCrashing() throws {
        guard let name = ModelCatalog.taskModels.first(where: { $0.isAvailable })?.resourceName,
              let path = ResourceLookup.path(name, ext: "gguf") else {
            throw XCTSkip("No .gguf in the bundle.")
        }
        let lm = ChatLM()
        guard lm.load(modelPath: path, backend: SLMBackend.cpu, loraPath: nil) else {
            throw XCTSkip("load failed: \(lm.lastError)")
        }

        // ~2000 tokens: comfortably past the 1024-token context.
        let huge = String(repeating: "The quick brown fox jumps over the lazy dog. ", count: 200)
        XCTAssertGreaterThan(lm.tokenCount(of: huge), 1024, "test prompt must exceed the limit")

        // The bar is simply: we get here at all, with an error rather than a crash.
        let out = lm.generate(system: nil, history: [], user: huge,
                              maxTokens: 4, temperature: 0.1, adapterPath: nil)
        XCTAssertNil(out, "an over-long prompt should fail, not return text")
        XCTAssertFalse(lm.lastError.isEmpty, "the failure must be reported")
        print("OVERLONG rejected cleanly: \(lm.lastError)")

        // And the engine must still be usable afterwards.
        let ok = lm.generate(system: nil, history: [], user: "Say hello.",
                             maxTokens: 4, temperature: 0.1, adapterPath: nil)
        XCTAssertNotNil(ok, "engine should survive a rejected prompt")
    }
}

/// Tokenizer-driven chunking: the budget is measured, and the boundaries are
/// chosen so a chunk is never cut mid-word.
@MainActor
final class TokenChunkingTests: XCTestCase {

    private let commaSplicedVN = """
    Mai chia sẻ nhật ký 24h, thức dậy lúc 6 giờ sáng ở Kyoto, pha một tách cà phê     rồi ra ban công ngồi ngắm phố xá còn vắng người, sau đó đạp xe qua mấy con hẻm     nhỏ đến rừng tre Arashiyama lúc 9 giờ 30, đi bộ giữa hai hàng tre cao vút nghe     tiếng gió xào xạc, chụp vài tấm ảnh rồi ghé một tiệm mì ramen nhỏ chỉ có sáu     chỗ ngồi, mùi nước dùng thơm nồng nàn, tiếng xì xụp của mấy nhân viên văn phòng     vang lên khắp quán, ăn xong thì lên tàu đi Osaka lúc 14 giờ 30, ngắm cảnh vật     thay đổi dần từ cổ kính sang hiện đại qua ô cửa sổ, đến Dotonbori thì người     đông đúc, tiếng loa quảng cáo vang dội, đèn neon rực rỡ khắp nơi, tối ghé một     quán bar nhỏ nằm dưới tầng hầm nghe nhạc jazz chill chill.
    """

    private func loadedService() throws -> ChatService {
        guard let option = ModelCatalog.taskModels.first(where: { $0.isAvailable }) else {
            throw XCTSkip("No .gguf in the bundle.")
        }
        let svc = ChatService()
        svc.taskModelResourceName = option.resourceName
        svc.taskModelID = option.id
        return svc
    }

    func testChunksRespectTheTokenBudgetAndNeverCutMidWord() async throws {
        let svc = try loadedService()
        guard await svc.warmUpIfNeeded().ok else { throw XCTSkip("warm-up failed") }

        let budget = svc.summaryChunkTokens
        let chunks = svc.chunkForSummary(commaSplicedVN, maxTokens: budget)
        print("TOKCHUNK budget=\(budget) tok, chunks=\(chunks.count)")

        for (i, c) in chunks.enumerated() {
            let toks = svc.taskEngine.tokenCount(of: c) ?? -1
            let endsMidWord = !(c.last.map { $0 == "." || $0 == "," || $0 == ";" || $0 == ":"
                                             || $0 == "!" || $0 == "?" } ?? false)
            print("TOKCHUNK   [\(i + 1)] \(toks) tok — …\(c.suffix(30))")
            XCTAssertLessThanOrEqual(toks, budget, "chunk \(i + 1) exceeds the token budget")
            XCTAssertFalse(endsMidWord,
                           "chunk \(i + 1) ends mid-word — the map summary of a fragment "
                           + "reads as a cropped summary: …\(c.suffix(40))")
        }
    }

    /// Prompt + generated output must fit the context. This is the invariant the
    /// derived budget exists to hold; a hardcoded 800 would violate it the moment
    /// the output budget moved.
    func testBudgetLeavesRoomForTheGeneratedSummary() throws {
        let svc = try ChatService()
        let total = Int(ChatLM.contextTokens())
        let used = svc.summaryChunkTokens + Int(svc.summaryMapTokens)
        print("TOKCHUNK context=\(total) chunk=\(svc.summaryChunkTokens) "
              + "+ output=\(svc.summaryMapTokens) = \(used)")
        XCTAssertLessThan(used, total, "chunk + output must fit the context")
        XCTAssertGreaterThan(total - used, 24, "too little headroom for template + instruction")
    }
}

/// The long-message summarize instruction.
///
/// Pinned because it is the one line that tells the model who is talking to whom.
/// With the previous wording ("Summarize the following conversation in X") the
/// model had no second party to attribute anything to and used the author for
/// both, producing "Mai shared her diary with Mai" on a real device.
final class SummarizeTextPromptTests: XCTestCase {

    func testInstructionNamesBothParties() {
        let p = Prompts.summarizeText("Xin chào", language: "Vietnamese", speaker: "Mai")
        let first = p.split(separator: "\n").first.map(String.init) ?? ""
        print("SUMPROMPT \(first)")

        XCTAssertTrue(first.contains("Mai"), "instruction must name the message author")
        XCTAssertTrue(first.contains(Prompts.userName),
                      "instruction must name the device owner as the other party")
        XCTAssertTrue(first.contains("Vietnamese"), "instruction must name the output language")
        XCTAssertFalse(first.contains("following conversation"),
                       "old wording is back — the model loses the second party")
        // The body keeps the trained [Name]: transcript shape.
        XCTAssertTrue(p.contains("[Mai]: Xin chào"))
    }

    /// The author is interpolated, not hardcoded — a group message carries its own
    /// author name.
    func testSpeakerIsInterpolated() {
        for name in ["Mai", "Sam", "Linh"] {
            let p = Prompts.summarizeText("x", language: "English", speaker: name)
            XCTAssertTrue(p.hasPrefix("Summarize the message of \(name) talk with "))
        }
    }
}
