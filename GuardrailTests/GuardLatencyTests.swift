import XCTest
@testable import SLMChat

/// Guardrail model latency per message — the classifier runs on every screened
/// message, so a runtime change must not slow it down. Prints p50/p90 and which
/// library the TFLite symbols came from; asserts only a generous ceiling.
final class GuardLatencyTests: XCTestCase {

    func testClassifierLatency() async throws {
        let engine = GuardEngine()
        let warm = await engine.warmUp(language: .vietnamese, region: .vietnam)
        try XCTSkipUnless(warm.ok, "guardrail model not bundled")

        let samples = [
            "Tối nay đi ăn phở không?", "Can we move the meeting to Thursday afternoon?",
            "Anh gửi file báo cáo quý 3 cho em nhé", "Happy birthday!! hope you have a great one",
            "Đừng quên mang ô, chiều nay trời mưa to đấy", "The deploy failed again, the build server is down",
        ]
        for s in samples { _ = await engine.classifyModelOnly(s, language: .vietnamese) }   // warm caches

        var ms: [Double] = []
        for i in 0..<240 {
            let s = samples[i % samples.count]
            let t = DispatchTime.now().uptimeNanoseconds
            _ = await engine.classifyModelOnly(s, language: .vietnamese)
            ms.append(Double(DispatchTime.now().uptimeNanoseconds - t) / 1e6)
        }
        ms.sort()
        let lib = Self.tfliteLibrary()
        print(String(format: "GUARD LATENCY | p50 %.3f ms  p90 %.3f ms  (TFLite from %@)",
                     ms[ms.count / 2], ms[ms.count * 9 / 10], lib))
        XCTAssertLessThan(ms[ms.count / 2], 20, "guardrail classify got much slower")
    }

    /// Which loaded image provides `TfLiteInterpreterInvoke`.
    static func tfliteLibrary() -> String {
        guard let sym = dlsym(UnsafeMutableRawPointer(bitPattern: -2), "TfLiteInterpreterInvoke") else { return "?" }
        var info = Dl_info()
        guard dladdr(sym, &info) != 0, let name = info.dli_fname else { return "?" }
        return (String(cString: name) as NSString).lastPathComponent
    }
}
