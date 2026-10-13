import SwiftUI

/// Deterministic monogram avatar derived from a contact name — shared by the inbox
/// rows, the new-message picker, and `Conversation`, so a person looks the same
/// everywhere. Initials render reliably on every device/simulator (unlike emoji).
enum Avatar {

    /// Up-to-two-letter uppercase monogram (e.g. "Mai Anh" → "MA").
    static func initials(_ name: String) -> String {
        let letters = name.split(separator: " ").prefix(2).compactMap { $0.first }
        return String(letters).uppercased()
    }

    /// Stable tint chosen from a fixed palette by hashing the name.
    static func tint(_ name: String) -> Color {
        let palette: [Color] = [.blue, .green, .orange, .purple, .pink, .teal, .indigo, .red]
        let hash = name.unicodeScalars.reduce(0) { $0 &+ Int($1.value) }
        return palette[hash % palette.count]
    }
}
