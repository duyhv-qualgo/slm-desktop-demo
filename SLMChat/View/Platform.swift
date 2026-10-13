import SwiftUI
#if canImport(UIKit)
import UIKit
typealias PlatformImage = UIImage
#else
import AppKit
typealias PlatformImage = NSImage
#endif

/// The few UIKit/AppKit differences the views touch, kept in one place so the
/// view files stay close to their iOS originals.
extension Image {
    init(platformImage: PlatformImage) {
        #if canImport(UIKit)
        self.init(uiImage: platformImage)
        #else
        self.init(nsImage: platformImage)
        #endif
    }
}

enum Clipboard {
    static func copy(_ string: String) {
        #if canImport(UIKit)
        UIPasteboard.general.string = string
        #else
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(string, forType: .string)
        #endif
    }
}

extension ToolbarItemPlacement {
    /// iOS `.topBarTrailing` / `.topBarLeading` don't exist on macOS.
    static var trailingBar: ToolbarItemPlacement {
        #if os(macOS)
        .primaryAction
        #else
        .topBarTrailing
        #endif
    }
    static var leadingBar: ToolbarItemPlacement {
        #if os(macOS)
        .navigation
        #else
        .topBarLeading
        #endif
    }
}

extension View {
    /// `.navigationBarTitleDisplayMode(.inline)` on iOS; macOS has no nav bar.
    @ViewBuilder func inlineNavigationTitle() -> some View {
        #if os(iOS)
        navigationBarTitleDisplayMode(.inline)
        #else
        self
        #endif
    }
}

extension View {
    /// iOS: tap the message area to dismiss the keyboard. macOS has no soft
    /// keyboard, and a tap gesture there would fight text selection.
    @ViewBuilder func dismissKeyboardOnTap() -> some View {
        #if os(iOS)
        contentShape(Rectangle())
            .onTapGesture {
                UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder),
                                                to: nil, from: nil, for: nil)
            }
        #else
        self
        #endif
    }
}

extension Color {
    /// iOS `secondarySystemBackground` — the grey of incoming bubbles and input
    /// fields. macOS has no equivalent; a system fill reads the same in both
    /// appearances.
    static var secondaryBackground: Color {
        #if canImport(UIKit)
        Color(.secondarySystemBackground)
        #else
        Color(nsColor: .secondarySystemFill)
        #endif
    }
}
