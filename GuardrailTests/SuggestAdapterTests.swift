import XCTest
@testable import SLMChat

/// Pins the one rule that `suggestAdapter` exists to enforce: a conversation in a
/// language whose pack is NOT installed must block, never quietly generate under
/// the native pack.
///
/// This was previously a `String?`, where "use this pack", "run under the loaded
/// one" and "the user hasn't installed the pack yet" all collapsed into nil, and
/// `ChatLM` turned every nil into `_defaultLoRAPath`. The symptom was a Spanish
/// chat logging `lora=…vietnamese.gguf` and producing Spanish-shaped mush. A
/// regression here is invisible in the output — the app still generates, it just
/// generates badly — which is exactly what makes it worth a test.
///
/// Deliberately written against whatever is installed on the test host rather than
/// a hardcoded expectation: install state is per-container and changes as soon as
/// anyone taps Get, so asserting "Spanish is missing" would fail on a machine where
/// it isn't. The invariant is the mapping, not the disk.
final class SuggestAdapterTests: XCTestCase {

    private func service(model: String, native: TranslateLanguage) -> ChatService {
        let s = ChatService()
        s.taskModelID = model
        s.nativeLanguage = native
        return s
    }

    /// The conversation is in the user's own language, so the pack the engine
    /// loaded with is already the right one.
    func testNativeLanguageRunsUnderTheLoadedPack() {
        let s = service(model: "qwen3.5-0.8b", native: .vietnamese)
        XCTAssertEqual(s.suggestAdapter(for: .vietnamese), .loaded)
    }

    /// No pack was ever published for this pair, so there is nothing to install and
    /// nothing to raise — generating under the loaded pack is the only option that
    /// produces anything at all. Gemma-3 ships Vietnamese and English only.
    func testUnpublishedPairRunsRatherThanBlocking() {
        let s = service(model: "gemma3-270m", native: .vietnamese)
        XCTAssertNil(LoRACatalog.adapter(baseModelID: "gemma3-270m", language: .arabic),
                     "fixture assumption: gemma3-270m publishes no Arabic pack")
        XCTAssertEqual(s.suggestAdapter(for: .arabic), .loaded)
    }

    /// The rule. For every non-native language on a multilingual checkpoint, the
    /// resolution must agree with the catalog and with disk — and critically, a
    /// published-but-uninstalled pack must come back `.missing`, not `.loaded`.
    func testUninstalledPackBlocksInsteadOfFallingBackToNative() {
        let model = "qwen3.5-0.8b"
        let s = service(model: model, native: .vietnamese)

        for language in TranslateLanguage.allCases where language != .vietnamese {
            let published = LoRACatalog.adapter(baseModelID: model, language: language)
            switch s.suggestAdapter(for: language) {
            case .pack(let path):
                XCTAssertEqual(path, published.flatMap(LoRACatalog.installedPath(for:)),
                               "\(language.name) resolved to a path that isn't its installed pack")
            case .missing(let pack):
                XCTAssertEqual(pack.id, published?.id)
                XCTAssertFalse(LoRACatalog.isInstalled(pack),
                               "\(language.name) blocked on a pack that IS installed")
            case .loaded:
                XCTAssertNil(published,
                             "\(language.name) has a published pack — an uninstalled one must "
                           + "block, not fall back to the native pack")
            }
        }
    }

    /// A blocked turn has to say something the user can act on: the language it
    /// needs, and the pack to install.
    func testMissingPackNoticeNamesTheLanguageAndThePack() {
        guard let pack = LoRACatalog.adapter(baseModelID: "qwen3.5-0.8b", language: .spanish) else {
            return XCTFail("fixture assumption: qwen3.5-0.8b publishes a Spanish pack")
        }
        let notice = ChatService.missingPackNotice(pack, language: .spanish)
        XCTAssertTrue(notice.contains("Spanish"))
        XCTAssertTrue(notice.contains(pack.displayName))
    }
}
