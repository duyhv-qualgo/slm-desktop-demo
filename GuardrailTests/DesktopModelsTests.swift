import XCTest
@testable import SLMChat

/// The macOS build ships two checkpoints out of the catalog's four.
final class DesktopModelsTests: XCTestCase {

    func testOffersQwen25AndQwen35Only() {
        XCTAssertEqual(Set(DesktopModels.taskModels.map(\.id)), ["qwen2.5-0.5b-v6.1", "qwen3.5-0.8b"])
        XCTAssertEqual(DesktopModels.defaultTaskModel.id, "qwen3.5-0.8b")
    }

    /// A choice saved before the subset (or the catalog's own Gemma default)
    /// lands on Qwen3.5, never on a model the app doesn't ship.
    func testDroppedOrUnknownChoiceFallsBackToQwen35() {
        for saved in ["gemma3-270m", "qwen3-0.6b-v6", "nonsense", nil] as [String?] {
            XCTAssertEqual(DesktopModels.resolve(saved).id, "qwen3.5-0.8b", "\(saved ?? "nil")")
            XCTAssertEqual(DesktopModels.taskModel(for: saved ?? "").id, "qwen3.5-0.8b", "\(saved ?? "nil")")
        }
        XCTAssertEqual(DesktopModels.resolve("qwen2.5-0.5b-v6.1").id, "qwen2.5-0.5b-v6.1")
    }

    func testShippedModelsAreBundled() {
        for option in DesktopModels.taskModels {
            XCTAssertTrue(option.isAvailable, "\(option.resourceName).gguf missing — run scripts/bootstrap.sh")
        }
    }
}
