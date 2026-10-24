import Foundation

// Logging for lines that carry chat content: message text, prompts built from
// it, model output, or data derived from it (summaries, events, photo tags).
//
// Compiled out of release builds. The app is designed as if chats are end-to-end
// encrypted, so derived data stays on the device — and the unified system log is
// not "on the device" in that sense: it is collected by sysdiagnose and readable
// by other tools. Lines without content (timings, counts, errors) keep using
// NSLog directly.

/// `NSLog` for a line that contains chat content. A no-op in release builds.
func contentLog(_ format: String, _ args: CVarArg...) {
    #if DEBUG
    withVaList(args) { NSLogv(format, $0) }
    #endif
}

/// `print` for a long line that contains chat content — prompts, which `NSLog`
/// truncates. A no-op in release builds; the text isn't even built there.
func contentPrint(_ text: @autoclosure () -> String) {
    #if DEBUG
    print(text())
    #endif
}
