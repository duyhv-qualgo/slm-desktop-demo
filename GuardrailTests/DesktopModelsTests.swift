import XCTest
@testable import SLMChat

/// The macOS build ships two checkpoints out of the catalog's four, plus the
/// evaluation models that are present.
final class DesktopModelsTests: XCTestCase {

    private var presentEvaluationIDs: Set<String> {
        Set(DesktopModels.evaluationModels.filter(\.isAvailable).map(\.id))
    }

    func testOffersQwen25AndQwen35PlusPresentEvaluationModels() {
        XCTAssertEqual(Set(DesktopModels.taskModels.map(\.id)),
                       Set(["qwen2.5-0.5b-v6.1", "qwen3.5-0.8b"]).union(presentEvaluationIDs))
        XCTAssertEqual(DesktopModels.defaultTaskModel.id, "qwen3.5-0.8b")
    }

    /// A choice saved before the subset (or the catalog's own Gemma default)
    /// lands on Qwen3.5, never on a model the app doesn't ship.
    func testDroppedOrUnknownChoiceFallsBackToQwen35() {
        for saved in ["gemma3-270m", "qwen3-0.6b-v6", "nonsense", nil] as [String?] {
            XCTAssertEqual(DesktopModels.resolve(saved).id, "qwen3.5-0.8b", "\(saved ?? "nil")")
        }
        XCTAssertEqual(DesktopModels.resolve("qwen2.5-0.5b-v6.1").id, "qwen2.5-0.5b-v6.1")
    }

    /// An evaluation model resolves to itself only while its file is present.
    func testEvaluationModelResolvesOnlyWhenPresent() {
        for option in DesktopModels.evaluationModels {
            XCTAssertNil(ModelCatalog.taskModels.first { $0.id == option.id },
                         "\(option.id) is in ModelCatalog — it shouldn't be an evaluation model")
            let expected = option.isAvailable ? option.id : DesktopModels.defaultTaskModel.id
            XCTAssertEqual(DesktopModels.resolve(option.id).id, expected)
            XCTAssertEqual(DesktopModels.taskModel(for: option.id).id, expected)
        }
    }

    /// Evaluation packs resolve through the catalog's normal lookups, stay out of
    /// the iOS rows, and the bundled file is the one the row describes.
    func testEvaluationPacksResolveThroughTheCatalog() throws {
        for pack in DesktopModels.evaluationAdapters {
            XCTAssertNil(LoRACatalog.adapters.first { $0.id == pack.id }, "\(pack.id) leaked into the iOS rows")
            for language in pack.languages {
                XCTAssertEqual(LoRACatalog.adapter(baseModelID: pack.baseModelID, language: language)?.id, pack.id)
            }
            XCTAssertTrue(LoRACatalog.adapters(forBaseModel: pack.baseModelID).contains(pack))
            guard let path = LoRACatalog.installedPath(for: pack) else {
                throw XCTSkip("\(pack.fileName) not bundled — run scripts/bootstrap.sh")
            }
            let size = try FileManager.default.attributesOfItem(atPath: path)[.size] as? Int64
            XCTAssertEqual(size, pack.sizeBytes, "\(pack.fileName) is not the file the row describes")
        }
    }

    func testShippedModelsAreBundled() {
        for option in DesktopModels.taskModels {
            XCTAssertTrue(option.isAvailable, "\(option.resourceName).gguf missing — run scripts/bootstrap.sh")
        }
    }
}
