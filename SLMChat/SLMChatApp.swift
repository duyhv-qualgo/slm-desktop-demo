import SwiftUI

@main
struct SLMChatApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var store = ConversationStore()

    /// The GuardrailTests bundle is hosted in this app. Its tests build their own
    /// engines; the window must not warm up a second set of models alongside.
    private static let isHostingTests =
        ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] != nil

    /// Opt-in memory tracing, for attributing a footprint climb to what caused it.
    ///
    /// Off by default — a sampling timer in every build is the kind of diagnostic
    /// that quietly becomes load-bearing. Turn it on for a session with
    /// `SLMCHAT_TRACK_MEMORY=1` in the scheme's environment, then reproduce the
    /// growth and read the `[MEM]` lines. See `MemoryTracker`.
    init() {
        if ProcessInfo.processInfo.environment["SLMCHAT_TRACK_MEMORY"] == "1" {
            MemoryTracker.start(label: "launch")
        }
    }

    var body: some Scene {
        // One window: the store owns a single on-device engine, so multiple
        // windows would only contend for it.
        Window("SLM Chat", id: "main") {
            if Self.isHostingTests {
                Text("Running tests…").frame(width: 300, height: 120)
            } else {
                ConversationListView(store: store)
                    .frame(minWidth: 820, minHeight: 560)
                    .preferredColorScheme(store.appearance.colorScheme)
                    .onAppear { appDelegate.store = store }
            }
        }
        // The rail and panels draw under the traffic lights, as in the design.
        .windowStyle(.hiddenTitleBar)
        .defaultSize(width: 1200, height: 780)
        .commands { AppCommands() }

        Settings {
            SettingsView(store: store)
                .frame(width: 560, height: 640)
                .preferredColorScheme(store.appearance.colorScheme)
        }
    }
}

private struct AppCommands: Commands {
    @FocusedValue(\.newChatAction) private var newChat

    var body: some Commands {
        CommandGroup(replacing: .newItem) {
            Button("New Chat") { newChat?() }
                .keyboardShortcut("n")
                .disabled(newChat == nil)
        }
    }
}

/// Unloads the SLM before the process exits.
///
/// llama.cpp's Metal backend asserts at static teardown that every GPU buffer was
/// freed (`ggml-metal-device.m: GGML_ASSERT([rsets->data count] == 0)`). On iOS
/// the app is killed rather than exited, so it never mattered; on macOS ⌘Q runs
/// `exit()` and a still-loaded model turns every quit into a crash report — and
/// can leave `SLMBackend`'s GPU-probe flag set, pinning the next launch to CPU.
final class AppDelegate: NSObject, NSApplicationDelegate {
    weak var store: ConversationStore?

    func applicationDidFinishLaunching(_ notification: Notification) {
        #if DEBUG
        // Dev aid: `SLMCHAT_SNAPSHOT=/path/out.png` writes the main window to a
        // PNG after a few seconds (and again on every later request via
        // `SLMCHAT_SNAPSHOT_DELAY`). The app draws its own view, so no Screen
        // Recording permission is needed.
        if let path = ProcessInfo.processInfo.environment["SLMCHAT_SNAPSHOT"] {
            let delay = Double(ProcessInfo.processInfo.environment["SLMCHAT_SNAPSHOT_DELAY"] ?? "") ?? 6
            DispatchQueue.main.asyncAfter(deadline: .now() + delay) {
                Self.snapshotMainWindow(to: URL(fileURLWithPath: path))
            }
        }
        #endif
    }

    #if DEBUG
    static func snapshotMainWindow(to url: URL) {
        guard let window = NSApp.windows.first(where: { $0.isVisible && $0.contentView != nil
                                                          && $0.frame.width > 400 }),
              let view = window.contentView?.superview ?? window.contentView,
              let rep = view.bitmapImageRepForCachingDisplay(in: view.bounds) else { return }
        view.cacheDisplay(in: view.bounds, to: rep)
        try? rep.representation(using: .png, properties: [:])?.write(to: url)
        NSLog("[Snapshot] wrote %@", url.path)
    }
    #endif

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard let store else { return .terminateNow }
        Task { @MainActor in
            await store.service.releaseSLMs()
            await store.search.shutdown()
            UserDefaults.standard.synchronize()
            sender.reply(toApplicationShouldTerminate: true)
        }
        return .terminateLater
    }
}
