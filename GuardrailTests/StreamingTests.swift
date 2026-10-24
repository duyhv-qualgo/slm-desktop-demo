import XCTest
@testable import SLMChat

// Streaming: does the UI receive text progressively, or all at once?
//
// These guard a failure mode with no visible symptom — the feature still
// produces correct text, only the progressive rendering dies, because the final
// parse succeeds where the partial one gives up.

/// Does the summarize output actually contain the JSON key the streaming parser
/// needs? `Prompts.partialSummary` looks for a literal `"summary"` and returns nil
/// until it finds one — so if the model emits plain prose, NOTHING renders
/// progressively and the whole summary appears at once when the stream finishes.
final class SummaryStreamShapeTests: XCTestCase {

    func testWhatTheModelActuallyEmits() throws {
        var any = false
        for option in ModelCatalog.taskModels where option.isAvailable {
            guard let path = ResourceLookup.path(option.resourceName, ext: "gguf") else { continue }
            let pack = LoRACatalog.adapters(forBaseModel: option.id)
                .first { $0.languages.contains(.vietnamese) }
                .flatMap { LoRACatalog.bundledPath(for: $0) }
            let lm = ChatLM()
            guard lm.load(modelPath: path, backend: SLMBackend.cpu,
                           loraPath: pack) else { continue }
            any = true

            let transcript = """
            [Alex]: Hey, are we still on for dinner tomorrow at seven?
            [Duy]: Yes! I booked the Mexican place downtown.
            [Alex]: Perfect. I'll meet you there. Should I bring anything?
            [Duy]: Just yourself. See you at seven.
            """
            let prompt = Prompts.summarizeText(transcript, language: "Vietnamese", speaker: "Alex")

            // Replay the exact streaming path and count how many partials render.
            var buffer = ""
            var renderedPartials = 0
            var firstRenderAtChars = -1
            lm.stream(system: nil, history: [], user: prompt,
                      maxTokens: 96, temperature: 0.3) { chunk, _ in
                guard !chunk.isEmpty else { return }
                buffer += chunk
                if Prompts.partialSummary(from: buffer) != nil {
                    renderedPartials += 1
                    if firstRenderAtChars < 0 { firstRenderAtChars = buffer.count }
                }
            }

            let hasKey = buffer.contains("\"summary\"")
            print("""
            STREAMSHAPE [\(option.id)] outputChars=\(buffer.count) \
            hasJSONKey=\(hasKey) partialsRendered=\(renderedPartials) \
            firstRenderAt=\(firstRenderAtChars) chars
            """)
            print("STREAMSHAPE [\(option.id)] raw: \(buffer.replacingOccurrences(of: "\n", with: "\\n").prefix(220))")

            // The regression. Whatever shape the model emits, the UI must receive
            // progressive updates — one update at the end is what this caught.
            XCTAssertGreaterThan(renderedPartials, 1,
                                 "\(option.id): summary did not stream — partialSummary "
                                 + "returned nil throughout, so the UI shows everything at once")
            XCTAssertLessThan(firstRenderAtChars, max(40, buffer.count / 2),
                              "\(option.id): first rendered frame arrived too late to read as streaming")
        }
        try XCTSkipUnless(any, "No .gguf in the bundle.")
    }
}

/// Unit-level behaviour of the streaming partial parser, independent of any model.
///
/// The end-to-end test above catches "nothing streamed"; these pin the individual
/// cases so a future change can't fix one shape by breaking another.
final class PartialValueTests: XCTestCase {

    /// Plain prose — what both shipping checkpoints actually emit.
    func testProseRendersProgressively() {
        let frames = ["Alex", "Alex và Duy", "Alex và Duy sẽ gặp nhau lúc 7 giờ."]
        var lastLength = 0
        for f in frames {
            let partial = try? XCTUnwrap(Prompts.partialSummary(from: f))
            let text = partial ?? ""
            XCTAssertFalse(text.isEmpty, "prose frame produced nothing: \(f)")
            XCTAssertGreaterThanOrEqual(text.count, lastLength, "output must not shrink")
            lastLength = text.count
        }
    }

    /// The translation finetune opens with a `[[LANG:… -> …]]` tag. While the tag
    /// is still streaming nothing may render, or the text jumps backwards the
    /// moment it closes and `clean` strips it.
    func testUnclosedTagRendersNothing() {
        let frames = ["[", "[[", "[[LANG:English -> H", "[[LANG:English -> Vietnamese]",
                      "[[LANG:English -> Vietnamese]] Hẹn", "[[LANG:English -> Vietnamese]] Hẹn gặp"]
        let partials = frames.compactMap { Prompts.partialTranslation(from: $0) }
        XCTAssertEqual(partials, ["Hẹn", "Hẹn gặp"])
    }

    /// The trained JSON shape still works — the fix must not trade one for the other.
    func testJSONValueIsExtracted() {
        let mid = #"{"task":"summarization","summary":"They agreed to meet at seven"#
        XCTAssertEqual(Prompts.partialSummary(from: mid), "They agreed to meet at seven")

        let done = #"{"task":"summarization","summary":"They agreed to meet at seven."}"#
        XCTAssertEqual(Prompts.partialSummary(from: done), "They agreed to meet at seven.")
    }

    /// A buffer that has opened a JSON object but not reached the key must render
    /// NOTHING — otherwise the UI flashes `{"task":"summarizat…` for a few frames.
    func testJSONPrefixRendersNothingBeforeTheKey() {
        for f in [#"{"#, #"{"task""#, #"{"task":"summarizat"#, #"  {"task":"summarization","#] {
            XCTAssertNil(Prompts.partialSummary(from: f),
                         "should not render a bare JSON prefix as prose: \(f)")
        }
    }

    /// A reasoning model's unclosed <think> must stay hidden — `clean` handles it,
    /// and this pins that the prose path keeps going through `clean`.
    func testUnclosedThinkIsSuppressed() {
        XCTAssertNil(Prompts.partialSummary(from: "<think>Let me consider the"))
        // Once thinking closes, the answer renders.
        XCTAssertEqual(Prompts.partialSummary(from: "<think>hmm</think>They agreed."),
                       "They agreed.")
    }

    /// Empty / whitespace buffers yield nothing rather than an empty frame.
    func testEmptyBufferYieldsNil() {
        XCTAssertNil(Prompts.partialSummary(from: ""))
        XCTAssertNil(Prompts.partialSummary(from: "   \n "))
    }

    /// The translation key routes through the same implementation.
    ///
    /// This is also the case that proves the key must be matched as a KEY: the
    /// buffer contains `"translation"` twice and the first is the value of `task`.
    /// A plain substring search returns the literal word "translation".
    func testTranslationSharesTheImplementation() {
        let mid = #"{"task":"translation","translation":"Hen gap lai"#
        XCTAssertEqual(Prompts.partialTranslation(from: mid), "Hen gap lai")
        XCTAssertEqual(Prompts.partialTranslation(from: "Hen gap lai"), "Hen gap lai")
        XCTAssertNil(Prompts.partialTranslation(from: #"{"task""#))
    }

    /// The same trap for summaries, written explicitly so it can't regress: a key
    /// appearing as a VALUE earlier in the object must be skipped.
    func testKeyAppearingAsAValueIsSkipped() {
        let raw = #"{"kind":"summary","summary":"The real text"#
        XCTAssertEqual(Prompts.partialSummary(from: raw), "The real text")
    }
}

/// End-to-end check that translation streams, against a real model.
///
/// Same regression as the summary: `partialTranslation` returning nil throughout
/// would leave the UI with one update at the end, and the feature would still look
/// correct because the final parse succeeds. Only progressive rendering breaks,
/// and only silently.
final class TranslateStreamTests: XCTestCase {

    func testTranslationStreamsProgressively() throws {
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

        let source = "See you tomorrow at seven in front of the cinema, don't be late."
        let prompt = Prompts.translate(source, from: .english, to: .vietnamese)

        var buffer = ""
        var partials: [String] = []
        lm.stream(system: nil, history: [], user: prompt,
                  maxTokens: 160, temperature: 0.2) { piece, _ in
            guard !piece.isEmpty else { return }
            buffer += piece
            if let p = Prompts.partialTranslation(from: buffer) { partials.append(p) }
        }

        print("""
        TRANSTREAM [\(option.id)] outputChars=\(buffer.count) partials=\(partials.count) \
        first=\(partials.first ?? "-") last=\(partials.last ?? "-")
        """)

        XCTAssertGreaterThan(partials.count, 1,
                             "translation did not stream — the UI would show it all at once")
        // Text must accumulate, never jump backwards.
        for (a, b) in zip(partials, partials.dropFirst()) {
            XCTAssertGreaterThanOrEqual(b.count, a.count - 2,
                                        "partial shrank unexpectedly: \(a) -> \(b)")
        }
        // Never leak the JSON scaffolding into the UI, whichever shape the model used.
        for p in partials {
            XCTAssertFalse(p.contains("\"task\""), "JSON leaked into a rendered frame: \(p)")
            XCTAssertFalse(p.hasPrefix("{"), "raw JSON rendered as text: \(p)")
        }
    }
}

final class LongSummaryStreamTests: XCTestCase {

    private func service() throws -> ChatService {
        guard let option = ModelCatalog.taskModels.first(where: { $0.isAvailable }) else {
            throw XCTSkip("No .gguf in the bundle.")
        }
        let svc = ChatService()
        svc.taskModelResourceName = option.resourceName
        svc.taskModelID = option.id
        return svc
    }

    /// Long enough to exceed the TOKEN budget (784) and genuinely need several map
    /// chunks. Sized in tokens, not characters: an earlier version of this fixture
    /// was written against the old 600-CHARACTER limit and silently collapsed to a
    /// single chunk once chunking became tokenizer-driven, which made the test
    /// assert on a code path it was no longer exercising.
    private let longText = String(
        repeating: "The team met on Tuesday to review the quarterly plan and agreed "
                 + "to move the launch date after discussing the remaining risks. "
                 + "Sales reported that the northern region beat its target while the "
                 + "south fell short of forecast for the third quarter running. ",
        count: 20)

    func testMultiChunkReportsProgressThenStreamsText() async throws {
        let svc = try service()
        guard await svc.warmUpIfNeeded().ok else { throw XCTSkip("warm-up failed") }

        // Ask the SERVICE how it will chunk, not the character fallback — they are
        // different functions with different budgets, and the test must expect what
        // the code under test will actually do.
        let expectedChunks = svc.chunkForSummary(longText, maxTokens: svc.summaryChunkTokens).count
        XCTAssertGreaterThan(expectedChunks, 1,
                             "fixture no longer exceeds the \(svc.summaryChunkTokens)-token "
                             + "budget — it is not exercising the multi-chunk path")

        var mapping: [(Int, Int)] = []
        var combining = 0
        var texts: [String] = []
        for await event in svc.summarizeLongStream(longText, language: "English", speaker: "Alex") {
            switch event {
            case .mapping(let done, let total): mapping.append((done, total))
            case .combining:                    combining += 1
            case .text(let t):                  texts.append(t)
            }
        }

        print("""
        LONGSUM chunks=\(expectedChunks) mapping=\(mapping.count) combining=\(combining) \
        text=\(texts.count) final=\(texts.last?.prefix(80) ?? "-")
        """)

        // Progress covers every MAP call, in order, and is reported before any text.
        XCTAssertEqual(mapping.count, expectedChunks, "one progress event per map chunk")
        XCTAssertEqual(mapping.map(\.0), Array(1...expectedChunks), "map progress must be in order")
        XCTAssertTrue(mapping.allSatisfy { $0.1 == expectedChunks }, "total must be the chunk count")

        // The final fold streams rather than arriving whole.
        XCTAssertGreaterThan(texts.count, 1, "final fold did not stream")
        XCTAssertFalse((texts.last ?? "").isEmpty, "no summary produced")
    }

    /// A message that fits one chunk needs no map-reduce, so it must not flash a
    /// progress line — it goes straight to streamed text.
    func testSingleChunkSkipsProgressEntirely() async throws {
        let svc = try service()
        guard await svc.warmUpIfNeeded().ok else { throw XCTSkip("warm-up failed") }

        let short = "Alex asked whether dinner was still on for seven, and Duy confirmed it was."
        XCTAssertEqual(Prompts.chunk(short).count, 1, "fixture must be single-chunk")

        var sawProgress = false
        var texts: [String] = []
        for await event in svc.summarizeLongStream(short, language: "English", speaker: "Alex") {
            switch event {
            case .mapping, .combining: sawProgress = true
            case .text(let t):         texts.append(t)
            }
        }
        XCTAssertFalse(sawProgress, "a single-chunk message should not report map-reduce progress")
        XCTAssertGreaterThan(texts.count, 1, "single-chunk summary did not stream")
    }
}

final class SummaryCompletenessTests: XCTestCase {

    private func service() throws -> ChatService {
        guard let option = ModelCatalog.taskModels.first(where: { $0.isAvailable }) else {
            throw XCTSkip("No .gguf in the bundle.")
        }
        let svc = ChatService()
        svc.taskModelResourceName = option.resourceName
        svc.taskModelID = option.id
        return svc
    }

    /// The last streamed frame must equal what the authoritative parse produces
    /// from the same generation — not a prefix of it.
    func testLastStreamedFrameIsComplete() async throws {
        let svc = try service()
        guard await svc.warmUpIfNeeded().ok else { throw XCTSkip("warm-up failed") }

        let text = "Alex asked whether dinner was still on for seven tonight, and Duy "
                 + "confirmed the booking at the Mexican place downtown and said he would "
                 + "arrive a little early to get a table by the window."

        for trial in 1...3 {
            var frames: [String] = []
            for await e in svc.summarizeLongStream(text, language: "English", speaker: "Alex") {
                if case .text(let t) = e { frames.append(t) }
            }
            let last = frames.last ?? ""
            let longest = frames.max(by: { $0.count < $1.count }) ?? ""
            print("COMPLETE trial=\(trial) frames=\(frames.count) lastLen=\(last.count) "
                  + "longestLen=\(longest.count) last=\(last)")

            // The final frame must be the longest — anything else means a later,
            // shorter frame overwrote it, which is exactly "cropped" on screen.
            XCTAssertEqual(last.count, longest.count,
                           "final frame is shorter than an earlier one — the stream "
                           + "delivered a truncated last value")
            XCTAssertFalse(last.isEmpty)
            // A summary that ends mid-word is the other face of the same bug.
            XCTAssertTrue(last.hasSuffix(".") || last.hasSuffix("!") || last.hasSuffix("?"),
                          "summary does not end on sentence punctuation: \(last)")
        }
    }

    /// The same completeness check on the MULTI-CHUNK path, which is where the
    /// report came from. This one has extra ways to lose text: three model calls
    /// feed a fourth, and each stage re-compresses.
    func testMultiChunkFinalFrameIsComplete() async throws {
        let svc = try service()
        guard await svc.warmUpIfNeeded().ok else { throw XCTSkip("warm-up failed") }

        // VARIED prose, not a repeated block. Repetition makes the map summaries
        // near-identical, and folding near-identical summaries pushes a small model
        // into repeating itself and trailing off — which looks like a truncation bug
        // and is really the fixture being unlike anything a user would send.
        let long = (1...14).map { i in
            "On day \(i) the team reviewed a different part of the plan. "
          + "Sales walked through region \(i) and flagged that its forecast had moved. "
          + "Engineering raised issue number \(i), which affects the release checklist. "
          + "Design showed revision \(i) of the onboarding screens and asked for feedback. "
        }.joined()
        let chunks = Prompts.chunk(long)
        XCTAssertGreaterThan(chunks.count, 1)

        var frames: [String] = []
        for await e in svc.summarizeLongStream(long, language: "English", speaker: "Alex") {
            if case .text(let t) = e { frames.append(t) }
        }
        let last = frames.last ?? ""
        let longest = frames.max(by: { $0.count < $1.count }) ?? ""
        print("""
        COMPLETE-MULTI chunks=\(chunks.count) inputChars=\(long.count) frames=\(frames.count) \
        lastLen=\(last.count) longestLen=\(longest.count)
        COMPLETE-MULTI last=\(last)
        """)

        // Small tolerance: the final frame goes through `parseSummary`, which trims
        // trailing whitespace, so it can be a character or two shorter than the last
        // streamed partial. A materially shorter final frame is the real bug.
        XCTAssertGreaterThanOrEqual(last.count, longest.count - 3,
                                    "final frame is materially shorter than an earlier one "
                                    + "— the stream delivered a truncated last value")
        XCTAssertFalse(last.isEmpty)
    }
}

/// A conversation long enough to exceed the context must still summarize.
///
/// Regression test for a real failure: an accumulated backlog rendered a
/// 1385-token prompt, the engine refused it (correctly — it will not overrun its
/// batch), and the summary came back empty with `stop=error`. The conversation
/// summarizer had no chunking at all, unlike the long-MESSAGE one.
@MainActor
final class ConversationChunkingTests: XCTestCase {

    private func service() throws -> ChatService {
        guard let option = ModelCatalog.taskModels.first(where: { $0.isAvailable }) else {
            throw XCTSkip("No .gguf in the bundle.")
        }
        let svc = ChatService()
        svc.taskModelResourceName = option.resourceName
        svc.taskModelID = option.id
        return svc
    }

    /// ~60 messages of real length — comfortably past the 1024-token context.
    private func backlog() -> [ChatMessage] {
        (0..<60).map { i in
            ChatMessage(sender: i % 3 == 0 ? .you : .friend,
                        text: "Message \(i): we talked about the trip to Osaka, the "
                            + "restaurant booking, and whether the schedule still works.")
        }
    }

    func testLongBacklogIsChunkedAndSummarized() async throws {
        let svc = try service()
        guard await svc.warmUpIfNeeded().ok else { throw XCTSkip("warm-up failed") }

        let msgs = backlog()
        let whole = Prompts.summarize(msgs, friendName: "Alex", native: .english)
        let wholeTokens = svc.taskEngine.tokenCount(of: whole) ?? 0
        XCTAssertGreaterThan(wholeTokens, Int(ChatLM.contextTokens()),
                             "fixture must exceed the context to exercise chunking")

        var mapping: [(Int, Int)] = []
        var texts: [String] = []
        for await e in svc.summarizeConversationStream(msgs, friendName: "Alex", native: .english) {
            switch e {
            case .mapping(let d, let t): mapping.append((d, t))
            case .combining:             break
            case .text(let s):           texts.append(s)
            }
        }
        print("CONVCHUNK wholePrompt=\(wholeTokens) tok chunks=\(mapping.last?.1 ?? 0) "
              + "textFrames=\(texts.count) final=\((texts.last ?? "").prefix(90))")

        XCTAssertGreaterThan(mapping.count, 1, "an over-long backlog must be chunked")
        XCTAssertFalse((texts.last ?? "").isEmpty,
                       "summary came back empty — this is the reported failure")
    }

    /// A short thread must NOT pay for chunking: one call, no progress line.
    func testShortThreadStaysASingleCall() async throws {
        let svc = try service()
        guard await svc.warmUpIfNeeded().ok else { throw XCTSkip("warm-up failed") }

        let msgs = (0..<4).map { i in
            ChatMessage(sender: i % 2 == 0 ? .friend : .you, text: "Short line \(i).")
        }
        var sawProgress = false
        var texts: [String] = []
        for await e in svc.summarizeConversationStream(msgs, friendName: "Alex", native: .english) {
            switch e {
            case .mapping, .combining: sawProgress = true
            case .text(let s):         texts.append(s)
            }
        }
        XCTAssertFalse(sawProgress, "a short thread should not report chunk progress")
        XCTAssertGreaterThan(texts.count, 1, "short-thread summary should still stream")
    }
}
