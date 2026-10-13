import XCTest
@testable import SLMChat

/// EmbeddingGemma 2 (text + vision) through LiteRT-LM, inside the app process.
///
/// LiteRT-LM's dylib exports its own copy of the TFLite C API (`TfLite*`), the
/// same symbols our `libtensorflowlite_c` provides for GuardLM / langdetect /
/// image tagging. These tests run in the same host as the guardrail parity
/// tests, so a wrong binding shows up there; here we check retrieval itself.
final class MessageEmbedderTests: XCTestCase {

    private static let docs = [
        "tối nay ăn gì? mình đói quá",
        "Can we move the meeting to Thursday afternoon?",
        "Anh gửi file báo cáo quý 3 cho em nhé",
        "I just landed in Hanoi, the weather is so humid",
        "Đừng quên mang ô, chiều nay trời mưa to đấy",
        "Happy birthday!! 🎂 hope you have a great one",
        "Con mèo nhà mình bị ốm, phải đưa đi bác sĩ thú y",
        "The deploy failed again, the build server is down",
        "ok",
        "haha",
        "Cuối tuần này đi biển Vũng Tàu không?",
        "Did you pay the electricity bill this month?",
    ]

    /// (query, index of the one relevant doc). Half cross-language.
    private static let queries: [(String, Int)] = [
        ("dinner plans", 0), ("reschedule meeting", 1), ("quarterly report", 2),
        ("arrived at the airport", 3), ("umbrella rain", 4), ("sinh nhật", 5),
        ("pet is sick", 6), ("server outage", 7), ("beach weekend", 10),
        ("tiền điện", 11), ("họp lại vào thứ năm", 1), ("thời tiết nóng ẩm", 3),
    ]

    func testCrossLingualRetrieval() async throws {
        let embedder = MessageEmbedder()
        do { try await embedder.load() } catch { throw XCTSkip("embedding model not bundled: \(error)") }

        let dv = try await embedder.embed(documents: Self.docs)
        XCTAssertEqual(dv.count, Self.docs.count)
        XCTAssertEqual(dv[0].count, MessageEmbedder.dimension)

        var top1 = 0
        for (q, gold) in Self.queries {
            let qv = try await embedder.embed(query: q)
            let scores = dv.map { d in zip(d, qv).reduce(0) { $0 + $1.0 * $1.1 } }
            let best = scores.indices.max { scores[$0] < scores[$1] }!
            if best == gold { top1 += 1 } else { print("miss: \(q) → \(Self.docs[best])") }
        }
        print("text→text top-1: \(top1)/\(Self.queries.count)")
        XCTAssertGreaterThanOrEqual(top1, 11)
        await embedder.unload()
    }

    /// Text → photo, EN and VI queries, against the stock macOS account
    /// pictures (present on every Mac, so no fixtures to ship).
    func testTextToPhotoRetrieval() async throws {
        let root = "/Library/User Pictures"
        let photos = ["Animals/Eagle", "Animals/Owl", "Animals/Parrot", "Animals/Penguin", "Animals/Zebra",
                      "Flowers/Sunflower", "Flowers/Red Rose", "Flowers/Lotus", "Instruments/Guitar",
                      "Instruments/Piano", "Instruments/Violin", "Instruments/Drum", "Nature/Cactus",
                      "Nature/Snowflake", "Nature/Lightning", "Nature/Earth", "Sports/Basketball",
                      "Sports/Soccer", "Sports/Tennis", "Fun/Gingerbread Man"]
        let queries: [(String, String)] = [
            ("a bird of prey", "Eagle"), ("con cú", "Owl"), ("colorful tropical bird", "Parrot"),
            ("animal living in Antarctica", "Penguin"), ("striped horse", "Zebra"),
            ("hoa hướng dương", "Sunflower"), ("valentine flower", "Red Rose"), ("hoa sen", "Lotus"),
            ("đàn piano", "Piano"), ("desert plant", "Cactus"), ("winter cold", "Snowflake"),
            ("thunderstorm", "Lightning"), ("our planet from space", "Earth"), ("NBA", "Basketball"),
            ("bóng đá", "Soccer"), ("christmas cookie", "Gingerbread Man"), ("đánh trống", "Drum"),
        ]
        let available = photos.filter { FileManager.default.fileExists(atPath: "\(root)/\($0).heic") }
        guard available.count == photos.count else { throw XCTSkip("stock account pictures not on this Mac") }

        let embedder = MessageEmbedder()
        do { try await embedder.load() } catch { throw XCTSkip("embedding model not bundled: \(error)") }

        var vectors: [String: [Float]] = [:]
        for p in photos {
            let data = try Data(contentsOf: URL(fileURLWithPath: "\(root)/\(p).heic"))
            vectors[(p as NSString).lastPathComponent] = try await embedder.embed(image: data)
        }
        var top1 = 0
        for (q, gold) in queries {
            let qv = try await embedder.embed(query: q)
            let best = vectors.max { zip($0.value, qv).reduce(0) { $0 + $1.0 * $1.1 }
                                   < zip($1.value, qv).reduce(0) { $0 + $1.0 * $1.1 } }!.key
            if best == gold { top1 += 1 } else { print("miss: \(q) → \(best)") }
        }
        print("text→photo top-1: \(top1)/\(queries.count)")
        XCTAssertGreaterThanOrEqual(top1, 15)
        await embedder.unload()
    }
}
