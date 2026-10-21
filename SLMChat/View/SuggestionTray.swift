import SwiftUI

/// Smart-reply suggestions above the input bar of a 1:1 chat. Deliberately
/// not offered in the group chat.
///
/// One row, always: chips scroll sideways rather than wrap, so the tray never
/// grows taller and pushes the conversation up. Each chip is a single line;
/// hovering shows the full text. Clicking a chip puts it in the composer to
/// edit (the existing behaviour); the paper plane that appears on hover sends
/// it as is. ⌘1–⌘3 pick a chip from the keyboard.
struct SuggestionTray: View {
    let suggestions: [String]
    let isLoading: Bool
    /// Refresh is offered only when a new generation could start now.
    let canRefresh: Bool
    let onUse: (String) -> Void
    let onSend: (String) -> Void
    let onRefresh: () -> Void
    let onDismiss: () -> Void

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: "sparkles")
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(Theme.accent)
                .help(isLoading ? "Suggesting replies…"
                                : "Suggested replies · click to edit, ⌘1–\(min(suggestions.count, 9)) to pick")
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    if isLoading {
                        SkeletonChips()
                    } else {
                        ForEach(Array(suggestions.prefix(9).enumerated()), id: \.element) { i, s in
                            ReplyChip(text: s, index: i, onUse: { onUse(s) }, onSend: { onSend(s) })
                        }
                    }
                }
                .padding(.vertical, 1)   // room for the chip border
            }
            // Fade the edge so a chip running off the right reads as "scroll".
            .mask(LinearGradient(stops: [.init(color: .black, location: 0.94),
                                         .init(color: .clear, location: 1)],
                                 startPoint: .leading, endPoint: .trailing))
            if !isLoading {
                TrayIconButton(systemImage: "arrow.clockwise", help: "New suggestions",
                               action: onRefresh)
                    .disabled(!canRefresh)
            }
            TrayIconButton(systemImage: "xmark", help: "Dismiss suggestions", action: onDismiss)
        }
        .padding(.horizontal, 24)
        .padding(.top, 8).padding(.bottom, 2)
        .frame(maxWidth: .infinity, alignment: .leading)
        .transition(.opacity)
    }
}

/// One suggestion: number badge, one line of text, send-now on hover.
private struct ReplyChip: View {
    let text: String
    let index: Int
    let onUse: () -> Void
    let onSend: () -> Void
    @State private var hovering = false

    var body: some View {
        HStack(spacing: 8) {
            Button(action: onUse) {
                HStack(alignment: .firstTextBaseline, spacing: 7) {
                    Text("\(index + 1)")
                        .font(.system(size: 10, weight: .bold, design: .rounded))
                        .foregroundStyle(Theme.accent)
                        .frame(width: 16, height: 16)
                        .background(Theme.accent.opacity(0.15), in: Circle())
                    Text(text)
                        .font(.system(size: 13))
                        .lineLimit(1)
                        .truncationMode(.tail)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .keyboardShortcut(KeyEquivalent(Character("\(index + 1)")), modifiers: .command)
            .help(text)

            if hovering {
                Button(action: onSend) {
                    Image(systemName: "paperplane.fill")
                        .font(.system(size: 11))
                        .foregroundStyle(.white)
                        .frame(width: 22, height: 22)
                        .background(Theme.accent, in: Circle())
                }
                .buttonStyle(.plain)
                .help("Send now")
                .transition(.scale.combined(with: .opacity))
            }
        }
        .padding(.leading, 8).padding(.trailing, hovering ? 5 : 12)
        .padding(.vertical, 6)
        .frame(maxWidth: 380, alignment: .leading)
        .background(Theme.accent.opacity(hovering ? 0.16 : 0.08), in: RoundedRectangle(cornerRadius: 16))
        .overlay(RoundedRectangle(cornerRadius: 16)
            .stroke(Theme.accent.opacity(hovering ? 0.6 : 0.3)))
        .onHover { h in withAnimation(.easeOut(duration: 0.12)) { hovering = h } }
    }
}

/// Placeholder chips while the model is generating.
private struct SkeletonChips: View {
    @State private var pulse = false

    var body: some View {
        HStack(spacing: 8) {
            ForEach([150.0, 210.0, 120.0], id: \.self) { w in
                RoundedRectangle(cornerRadius: 16)
                    .fill(Theme.accent.opacity(pulse ? 0.16 : 0.06))
                    .frame(width: w, height: 30)
            }
            Text("Suggesting replies…")
                .font(.system(size: 11.5))
                .foregroundStyle(Theme.textSecondary)
        }
        // Scoped to this view, so the endless pulse can't leak into the
        // chips that replace it.
        .animation(.easeInOut(duration: 0.8).repeatForever(autoreverses: true), value: pulse)
        .onAppear { pulse = true }
    }
}

private struct TrayIconButton: View {
    let systemImage: String
    let help: String
    let action: () -> Void
    @State private var hovering = false
    @Environment(\.isEnabled) private var isEnabled

    var body: some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(Theme.textSecondary.opacity(isEnabled ? 1 : 0.4))
                .frame(width: 22, height: 22)
                .background(hovering && isEnabled ? Theme.rowHover : .clear, in: Circle())
                .contentShape(Circle())
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
        .help(help)
    }
}
