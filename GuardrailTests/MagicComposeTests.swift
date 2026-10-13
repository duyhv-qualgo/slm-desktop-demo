import XCTest
@testable import SLMChat

/// Pins the two halves of Magic Compose that are a MODEL CONTRACT rather than UI:
/// the prompt shape the finetune was trained on, and the tag protocol it answers
/// with.
///
/// Both fail silently if broken. A prompt that drops the `Draft to rewrite:`
/// block still generates — it just quietly ignores what the user typed and writes
/// a fresh reply, which reads as a bad rewrite rather than a bug. A parser that
/// misses `[[REFUSE:HARMFUL_INSTRUCTION]]` doesn't throw either: it strips the tag
/// as ordinary `[[…]]` noise and drops whatever follows straight into the input
/// box the user is about to send from. Neither shows up in a build.
final class MagicComposeTests: XCTestCase {

    private func service() -> ChatService { ChatService() }

    private func history() -> [ChatMessage] {
        [ChatMessage(sender: .friend, text: "Can you send the deck before the meeting?"),
         ChatMessage(sender: .you, text: "sure")]
    }

    private func chip(_ id: String) -> Prompts.ComposeInstruction {
        let all = Prompts.rewriteChips + Prompts.smartReplyChips
        return all.first { $0.id == id }!
    }

    // MARK: - Chip catalog

    /// The instruction strings are what the model was trained against, so they are
    /// asserted verbatim. Reword one for readability and output quality drops with
    /// no compile error — which is exactly the failure this file exists to catch.
    func testInstructionStringsAreTheTrainedValues() {
        XCTAssertEqual(chip("fix").instruction, "Fix spelling and diacritics")
        XCTAssertEqual(chip("professional").instruction, "Make it more professional")
        XCTAssertEqual(chip("friendly").instruction, "Make it friendlier")
        XCTAssertEqual(chip("shorten").instruction, "Make it shorter")
        XCTAssertEqual(chip("elaborate").instruction, "Make it longer and more detailed")
        XCTAssertEqual(chip("soften").instruction, "Soften it")
        XCTAssertEqual(chip("casual").instruction, "Make it more casual")

        XCTAssertEqual(chip("agree").instruction, "Agree / accept")
        XCTAssertEqual(chip("decline").instruction, "Politely decline")
        XCTAssertEqual(chip("reply-pro").instruction, "Reply professionally")
        XCTAssertEqual(chip("busy").instruction, "Say I'm busy")
        XCTAssertEqual(chip("get-back-later").instruction, "I'll get back to you")
    }

    /// Ids are the `Identifiable` key AND how the bar marks the active chip, so a
    /// collision across the two sets would highlight the wrong one.
    func testChipIdsAreUnique() {
        let ids = (Prompts.rewriteChips + Prompts.smartReplyChips).map(\.id)
        XCTAssertEqual(Set(ids).count, ids.count)
    }

    /// The chips are the whole input surface — the bar offers no free-typed
    /// instruction — so these two catalogs are exhaustive. A chip added to
    /// neither is unreachable; the count is here so adding one is a deliberate
    /// edit rather than a silent drift from the spec.
    func testChipCatalogsAreTheCompleteInputSurface() {
        XCTAssertEqual(Prompts.rewriteChips.count, 7)
        XCTAssertEqual(Prompts.smartReplyChips.count, 5)
        for chip in Prompts.rewriteChips { XCTAssertEqual(chip.mode, .rewrite) }
        for chip in Prompts.smartReplyChips { XCTAssertEqual(chip.mode, .smartReply) }
    }

    /// A draft is the only input to the mode choice.
    func testDraftPresenceSelectsTheChipSet() {
        XCTAssertEqual(Prompts.composeChips(hasDraft: true), Prompts.rewriteChips)
        XCTAssertEqual(Prompts.composeChips(hasDraft: false), Prompts.smartReplyChips)
    }

    // MARK: - Prompt assembly

    /// Mode B: the draft block is what makes this a rewrite instead of a fresh
    /// reply. Its absence is invisible in the output.
    func testDraftModeAppendsTheRewriteBlock() {
        let prompt = Prompts.magicCompose(history: history(),
                                          instruction: chip("shorten"),
                                          language: .english,
                                          draft: "i'll get it over to you soon ok",
                                          user: "Duy")
        XCTAssertTrue(prompt.hasPrefix(
            "Write a reply in English for Duy in the conversation below, "
          + "following this instruction: Make it shorter"))
        XCTAssertTrue(prompt.contains("Conversation:\n"))
        XCTAssertTrue(prompt.contains("Draft to rewrite:\ni'll get it over to you soon ok"))
    }

    /// Mode A: no draft, so the block is omitted entirely — an empty
    /// `Draft to rewrite:` header would be a prompt asking the model to rewrite
    /// nothing.
    func testSmartReplyModeOmitsTheRewriteBlock() {
        let prompt = Prompts.magicCompose(history: history(),
                                          instruction: chip("busy"),
                                          language: .vietnamese,
                                          draft: "",
                                          user: "Duy")
        XCTAssertTrue(prompt.contains("following this instruction: Say I'm busy"))
        XCTAssertFalse(prompt.contains("Draft to rewrite:"))
    }

    /// A box holding only spaces is an EMPTY box as far as the user is concerned,
    /// so it must take the smart-reply shape rather than ask for a rewrite of "  ".
    func testWhitespaceOnlyDraftIsTreatedAsNoDraft() {
        let prompt = Prompts.magicCompose(history: history(),
                                          instruction: chip("agree"),
                                          language: .english,
                                          draft: "   \n  ",
                                          user: "Duy")
        XCTAssertFalse(prompt.contains("Draft to rewrite:"))
    }

    /// The language named in the prompt is the one the caller resolved — the same
    /// value that picks the LoRA pack the generation runs under.
    func testLanguageNameComesFromTheResolvedLanguage() {
        for language in TranslateLanguage.allCases {
            let prompt = Prompts.magicCompose(history: history(),
                                              instruction: chip("friendly"),
                                              language: language,
                                              draft: "hey")
            XCTAssertTrue(prompt.hasPrefix("Write a reply in \(language.name) for "),
                          "\(language.rawValue) prompt: \(prompt.prefix(60))")
        }
    }

    /// Mirrors the reference builder's `or "me"` fallback. Without it the line
    /// reads "for  in the conversation below".
    func testBlankUserFallsBackToMe() {
        let prompt = Prompts.magicCompose(history: history(),
                                          instruction: chip("casual"),
                                          language: .english,
                                          draft: "hey",
                                          user: "   ")
        XCTAssertTrue(prompt.hasPrefix("Write a reply in English for me in the"))
    }

    /// The conversation is rendered in the finetune's `<Handle>: text` form, the
    /// same as auto-suggest — the friend under the CONTACT's name, not a generic
    /// one.
    func testConversationUsesHandleFormWithTheContactName() {
        let prompt = Prompts.magicCompose(history: history(),
                                          instruction: chip("fix"),
                                          language: .english,
                                          draft: "hey",
                                          friendName: "Mai",
                                          user: "Duy")
        XCTAssertTrue(prompt.contains("<Mai>: Can you send the deck before the meeting?"))
        XCTAssertTrue(prompt.contains("<Duy>: sure"))
    }

    // MARK: - Language resolution

    /// The rule the whole feature turns on: a rewrite follows the DRAFT's
    /// language, not the thread's.
    ///
    /// This fails silently and expensively. Resolve a Vietnamese thread's
    /// language for an English draft and the prompt asks for Vietnamese under the
    /// Vietnamese pack — so "Make it shorter" returns a fluent Vietnamese
    /// TRANSLATION of the draft. The user sees a confident answer to a question
    /// they didn't ask, and nothing anywhere reports an error.
    func testDraftLanguageWinsOverTheThreadLanguage() {
        let vietnameseThread = [
            ChatMessage(sender: .friend, text: "Mai họp lúc mấy giờ vậy anh?"),
            ChatMessage(sender: .you, text: "Để anh kiểm tra lại lịch nhé"),
        ]
        let englishDraft = "could you please send over the deck before the meeting tomorrow"

        XCTAssertEqual(Prompts.conversationLanguage(vietnameseThread, native: .vietnamese),
                       .vietnamese, "precondition: the thread really is Vietnamese")
        XCTAssertEqual(Prompts.composeLanguage(history: vietnameseThread,
                                               draft: englishDraft,
                                               native: .vietnamese),
                       .english)
    }

    /// And the same in the other direction, so the test isn't just pinning
    /// "English wins".
    func testVietnameseDraftInAnEnglishThreadResolvesVietnamese() {
        let englishThread = [
            ChatMessage(sender: .friend, text: "Are we still on for the review tomorrow morning?"),
            ChatMessage(sender: .you, text: "Yes, I think so"),
        ]
        let vietnameseDraft = "Anh gửi tài liệu cho em trước buổi họp ngày mai nhé"

        XCTAssertEqual(Prompts.composeLanguage(history: englishThread,
                                               draft: vietnameseDraft,
                                               native: .english),
                       .vietnamese)
    }

    /// A draft too short to identify must NOT be guessed at — it falls through to
    /// the conversation, which is where a short message lands in
    /// `conversationLanguage` too.
    func testUnidentifiableDraftFallsBackToTheConversation() {
        let vietnameseThread = [
            ChatMessage(sender: .friend, text: "Mai họp lúc mấy giờ vậy anh?"),
            ChatMessage(sender: .you, text: "Để anh kiểm tra lại lịch nhé"),
        ]
        for draft in ["ok", "haha", "?", "   "] {
            XCTAssertEqual(Prompts.composeLanguage(history: vietnameseThread,
                                                   draft: draft,
                                                   native: .english),
                           Prompts.conversationLanguage(vietnameseThread, native: .english),
                           "draft \"\(draft)\" should not have been identified")
        }
    }

    /// Smart-reply mode has no draft, so it must resolve through exactly the path
    /// auto-suggest takes — the two features proposing text in different languages
    /// for the same thread would be visible to the user.
    func testEmptyDraftMatchesAutoSuggestResolution() {
        let threads: [[ChatMessage]] = [
            [ChatMessage(sender: .friend, text: "Mai họp lúc mấy giờ vậy anh?")],
            [ChatMessage(sender: .friend, text: "Are we still on for tomorrow morning?")],
            [ChatMessage(sender: .friend, text: "¿Nos vemos mañana por la tarde entonces?")],
            [ChatMessage(sender: .friend, text: "ok")],
        ]
        for thread in threads {
            for native in [TranslateLanguage.vietnamese, .english] {
                XCTAssertEqual(Prompts.composeLanguage(history: thread, draft: "", native: native),
                               Prompts.conversationLanguage(thread, native: native),
                               "thread: \(thread.first?.text ?? "") native: \(native.rawValue)")
            }
        }
    }

    // MARK: - Send interaction

    /// `send` closes the compose bar, and the call sits AFTER the empty-text
    /// guard — so a send the guard rejects leaves the bar exactly as it was
    /// rather than tearing it down under the user.
    ///
    /// Only the rejected path is covered here, and deliberately so: a successful
    /// `send` spawns the friend-reply task, which reaches the live chat API. The
    /// closing behaviour itself was verified by hand against the running app.
    @MainActor
    func testRejectedSendLeavesTheComposeBarOpen() {
        // Retained for the length of the test: `Conversation` holds its service
        // `unowned` on the premise that the store outlives every conversation.
        let service = ChatService()
        let convo = Conversation(
            contactName: "Alex",
            messages: [ChatMessage(sender: .friend, text: "you free tomorrow?")],
            service: service)

        convo.beginCompose(draft: "yeah sounds good")
        XCTAssertTrue(convo.isComposeOpen)

        convo.send("   ")   // whitespace only — rejected by the guard

        XCTAssertTrue(convo.isComposeOpen, "an empty send must not close the bar")
        XCTAssertEqual(convo.composeDraft, "yeah sounds good")
        XCTAssertTrue(convo.messages.allSatisfy { $0.sender == .friend },
                      "precondition: nothing was actually sent")
    }

    // MARK: - Output parsing

    func testRefusalTagIsDetectedWithItsReason() {
        XCTAssertEqual(Prompts.parseCompose("[[REFUSE:HARMFUL_INSTRUCTION]]"),
                       .refused(reason: "HARMFUL_INSTRUCTION"))
    }

    func testBareRefusalTagHasNoReason() {
        XCTAssertEqual(Prompts.parseCompose("[[REFUSE]]"), .refused(reason: nil))
    }

    /// The refusal must win even when the model adds prose around it — otherwise
    /// the prose becomes the "rewrite" and lands in the input box.
    func testRefusalWinsOverSurroundingProse() {
        let outcome = Prompts.parseCompose("Sorry.\n[[REFUSE:HARMFUL_INSTRUCTION]]\nI can't help.")
        XCTAssertEqual(outcome, .refused(reason: "HARMFUL_INSTRUCTION"))
    }

    func testReplyTagIsStripped() {
        XCTAssertEqual(Prompts.parseCompose("[[REPLY]] I'll send it right over."),
                       .text("I'll send it right over."))
    }

    /// A model that narrates before the tag must not leak that preamble.
    func testPreambleBeforeTheReplyTagIsDropped() {
        XCTAssertEqual(Prompts.parseCompose("Here is a shorter version:\n[[REPLY]] Sending it now."),
                       .text("Sending it now."))
    }

    /// The small models drop the marker often enough that treating an untagged
    /// rewrite as a failure would throw away good output.
    func testUntaggedOutputIsAccepted() {
        XCTAssertEqual(Prompts.parseCompose("Sending it now."), .text("Sending it now."))
    }

    /// A reasoning model musing about the refusal tag in its scratchpad has NOT
    /// refused. This is why tag detection runs after `stripThinking` and before
    /// `clean` — get the order wrong in either direction and one of these two
    /// tests fails.
    func testRefusalInsideThinkingIsNotARefusal() {
        let raw = "<think>Should I emit [[REFUSE:HARMFUL_INSTRUCTION]]? No, it's fine.</think>"
                + "[[REPLY]] Sounds good, see you at 3:30."
        XCTAssertEqual(Prompts.parseCompose(raw), .text("Sounds good, see you at 3:30."))
    }

    /// An unclosed `<think>` (a truncated generation) leaves nothing usable rather
    /// than dumping the reasoning into the input box.
    func testUnclosedThinkingYieldsEmpty() {
        XCTAssertEqual(Prompts.parseCompose("<think>Let me consider the tone"), .empty)
    }

    /// `.empty` is retryable and `.refused` is not, so a blank generation must not
    /// come back as either a refusal or usable text.
    func testBlankAndTagOnlyOutputIsEmpty() {
        XCTAssertEqual(Prompts.parseCompose(""), .empty)
        XCTAssertEqual(Prompts.parseCompose("   \n "), .empty)
        XCTAssertEqual(Prompts.parseCompose("[[REPLY]]"), .empty)
    }

    /// `parseCompose` leans on `clean` for everything after the tag, including the
    /// anchored list-marker rule — so a time at either edge has to survive.
    func testNumbersAtTheEdgesSurvive() {
        XCTAssertEqual(Prompts.parseCompose("[[REPLY]] 3:30 works for me — see you there."),
                       .text("3:30 works for me — see you there."))
    }

    // MARK: - `clean` refactor guard

    /// `stripThinking` was lifted out of `clean`; `clean` must behave exactly as
    /// it did, since every other feature's output still goes through it.
    func testCleanStillStripsThinkingAndTags() {
        XCTAssertEqual(Prompts.clean("<think>hmm</think>[[INQUIRE]] They agreed."),
                       "They agreed.")
        XCTAssertEqual(Prompts.clean("<think>unclosed reasoning"), "")
    }
}
