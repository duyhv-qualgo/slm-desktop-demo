import XCTest
@testable import SLMChat

/// Parity check for `GuardTokenizer` against the reference Hugging Face
/// tokenizer.
///
/// `tokenizer_fixtures.json` holds 70 `text → ids` pairs produced by the REAL HF
/// tokenizer for the guardrail model. The Swift port reimplements that pipeline
/// (NFKC + lowercase → regex pre-tokenize → BPE over Unicode scalars), and a
/// divergence in any of those steps does not crash — it silently produces
/// different token ids, which the classifier then turns into a confident but
/// wrong verdict. Hence a fixture-driven test rather than hand-written cases.
final class GuardTokenizerTests: XCTestCase {

    private struct Fixture: Decodable {
        let text: String
        let ids: [Int32]
    }

    /// Fixtures live in the TEST bundle; the tokenizer vocab is a resource of the
    /// APP bundle (reached through `ResourceLookup`, i.e. `Bundle.main`).
    private func loadFixtures() throws -> [Fixture] {
        let bundle = Bundle(for: Self.self)
        let url = try XCTUnwrap(bundle.url(forResource: "tokenizer_fixtures",
                                           withExtension: "json"),
                                "tokenizer_fixtures.json missing from the test bundle")
        return try JSONDecoder().decode([Fixture].self, from: Data(contentsOf: url))
    }

    /// The Vietnamese pack's tokenizer. Tokenizers are per-pack now — the
    /// fixtures were captured from the Hugging Face tokenizer that ships with the
    /// VN/EN checkpoint, so they pin THAT pack's vocabulary. A future pack with a
    /// different vocabulary needs its own fixtures, not this one relaxed.
    private func makeTokenizer() throws -> GuardTokenizer {
        let pack = try XCTUnwrap(GuardPackCatalog.pack(for: .vietnamese),
                                 "no Vietnamese guardrail pack in the catalog")
        let url = try XCTUnwrap(
            ResourceLookup.url(pack.tokenizerResource, ext: "json"),
            "\(pack.tokenizerResource).json not bundled in the app")
        return try XCTUnwrap(GuardTokenizer(tokenizerJSONURL: url),
                             "tokenizer JSON present but could not be parsed")
    }

    func testEncodeMatchesReferenceTokenizer() throws {
        let tokenizer = try makeTokenizer()
        let fixtures = try loadFixtures()
        XCTAssertFalse(fixtures.isEmpty)

        for fixture in fixtures {
            let (ids, mask) = tokenizer.encode(fixture.text)

            // `encode` returns a fixed-length, PAD-filled buffer; the fixtures hold
            // only the real tokens. Compare the unpadded prefix, which the mask
            // delimits. Fixtures longer than maxLen are truncated by design.
            let real = zip(ids, mask).filter { $0.1 == 1 }.map(\.0)
            let expected = Array(fixture.ids.prefix(tokenizer.maxLen))

            XCTAssertEqual(real, expected,
                           "token ids differ for \(fixture.text.debugDescription)")
        }
    }

    /// The padding contract the TFLite input tensors depend on: both arrays are
    /// exactly `maxLen`, padding is id 0, and the mask marks the real tokens.
    func testEncodePadsToFixedLength() throws {
        let tokenizer = try makeTokenizer()
        let (ids, mask) = tokenizer.encode("Xin chào bạn")

        XCTAssertEqual(ids.count, tokenizer.maxLen)
        XCTAssertEqual(mask.count, tokenizer.maxLen)

        let real = mask.filter { $0 == 1 }.count
        XCTAssertGreaterThan(real, 0)
        XCTAssertTrue(ids[real...].allSatisfy { $0 == 0 }, "padding must be id 0")
        XCTAssertTrue(mask[real...].allSatisfy { $0 == 0 }, "mask must be 0 past the text")
    }

    /// An empty string must still yield a well-formed, all-padding input rather
    /// than an empty array the TFLite tensor copy would under-fill.
    func testEncodeEmptyString() throws {
        let tokenizer = try makeTokenizer()
        let (ids, mask) = tokenizer.encode("")

        XCTAssertEqual(ids.count, tokenizer.maxLen)
        XCTAssertEqual(mask, [Int32](repeating: 0, count: tokenizer.maxLen))
    }
}
