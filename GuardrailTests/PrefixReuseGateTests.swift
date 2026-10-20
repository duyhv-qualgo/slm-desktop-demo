import XCTest
@testable import SLMChat

/// Prefix reuse is off for the hybrid attention/SSM checkpoints whatever the
/// setting, and follows the setting for the rest (`ChatService.prefixReuseActive`).
final class PrefixReuseGateTests: XCTestCase {

    func testOffForQwen35AndBonsaiWhateverTheSetting() {
        let svc = ChatService()
        for id in ["qwen3.5-0.8b", "bonsai2-2b"] {
            svc.taskModelID = id
            for setting in [true, false] {
                svc.prefixReuseEnabled = setting
                XCTAssertFalse(svc.prefixReuseActive, "\(id) setting=\(setting)")
            }
        }
    }

    func testFollowsTheSettingForOtherModels() {
        let svc = ChatService()
        svc.taskModelID = "qwen2.5-0.5b-v6.1"
        svc.prefixReuseEnabled = true
        XCTAssertTrue(svc.prefixReuseActive)
        svc.prefixReuseEnabled = false
        XCTAssertFalse(svc.prefixReuseActive)
    }

    /// The engine follows the gate when the model changes, not only when the
    /// setting does — switching onto Qwen3.5 must turn reuse off in the engine.
    func testEngineFollowsAModelSwitch() {
        let svc = ChatService()
        svc.prefixReuseEnabled = true
        svc.taskModelID = "qwen2.5-0.5b-v6.1"
        XCTAssertTrue(svc.taskEngine.prefixReuseEnabled)
        svc.taskModelID = "qwen3.5-0.8b"
        XCTAssertFalse(svc.taskEngine.prefixReuseEnabled)
        svc.taskModelID = "qwen2.5-0.5b-v6.1"
        XCTAssertTrue(svc.taskEngine.prefixReuseEnabled)
    }
}
