import SwiftUI
#if canImport(AppKit)
import AppKit
#endif

/// Desktop palette and the small building blocks shared by the main window,
/// the chat panes and the input bars. Dark values follow the One Team desktop
/// design; light values are the same structure inverted.
enum Theme {
    static let accent        = Color(red: 0.49, green: 0.44, blue: 0.96)   // #7D70F5
    static let rail          = dynamic(light: 0xE6E6EC, dark: 0x16171B)
    static let sidebar       = dynamic(light: 0xF3F3F6, dark: 0x1E1F24)
    static let chat          = dynamic(light: 0xFFFFFF, dark: 0x0B0C10)
    static let rowSelected   = dynamic(light: 0xE1E1E9, dark: 0x2E2F35)
    static let rowHover      = dynamic(light: 0xEAEAF0, dark: 0x26272C)
    static let field         = dynamic(light: 0xEDEDF2, dark: 0x2A2B30)
    static let fieldBorder   = dynamic(light: 0xD6D6DE, dark: 0x3A3B41)
    static let bubbleIn      = dynamic(light: 0xEEEEF3, dark: 0x26272D)
    static let banner        = dynamic(light: 0xF1F0FE, dark: 0x1C1B2E)
    static let textSecondary = dynamic(light: 0x6E6E78, dark: 0x9A9BA3)

    /// Corner radius of the sidebar / chat panels inside the window.
    static let panelRadius: CGFloat = 14

    private static func dynamic(light: UInt32, dark: UInt32) -> Color {
        #if canImport(AppKit)
        Color(nsColor: NSColor(name: nil) { appearance in
            let isDark = appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
            return NSColor(hex: isDark ? dark : light)
        })
        #else
        Color(UIColor { $0.userInterfaceStyle == .dark ? UIColor(hex: dark) : UIColor(hex: light) })
        #endif
    }
}

#if canImport(AppKit)
private extension NSColor {
    convenience init(hex: UInt32) {
        self.init(srgbRed: CGFloat((hex >> 16) & 0xFF) / 255,
                  green: CGFloat((hex >> 8) & 0xFF) / 255,
                  blue: CGFloat(hex & 0xFF) / 255, alpha: 1)
    }
}
#else
private extension UIColor {
    convenience init(hex: UInt32) {
        self.init(red: CGFloat((hex >> 16) & 0xFF) / 255,
                  green: CGFloat((hex >> 8) & 0xFF) / 255,
                  blue: CGFloat(hex & 0xFF) / 255, alpha: 1)
    }
}
#endif

// MARK: - Building blocks

/// Monogram avatar used across the sidebar, headers and info panel.
struct AvatarCircle: View {
    let name: String
    var size: CGFloat = 28
    var systemImage: String? = nil
    var tint: Color? = nil

    var body: some View {
        ZStack {
            Circle().fill((tint ?? Avatar.tint(name)).gradient)
            if let systemImage {
                Image(systemName: systemImage)
                    .font(.system(size: size * 0.42, weight: .semibold))
            } else {
                Text(Avatar.initials(name))
                    .font(.system(size: size * 0.38, weight: .semibold))
            }
        }
        .foregroundStyle(.white)
        .frame(width: size, height: size)
    }
}

/// Large, light header icon (phone / search / panel style in the design).
/// `active` draws it in the accent colour, for toggles like the info panel.
struct HeaderIconButton: View {
    let systemImage: String
    let help: String
    var active = false
    let action: () -> Void
    @State private var hovering = false

    var body: some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .font(.system(size: 17, weight: .regular))
                .foregroundStyle(active ? Theme.accent : .primary)
                .frame(width: 34, height: 34)
                .background(hovering ? Theme.rowHover : .clear,
                            in: RoundedRectangle(cornerRadius: 8))
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
        .help(help)
    }
}

/// The header strip above a chat: avatar, title, subtitle, trailing actions.
struct ChatHeader<AvatarView: View, Actions: View>: View {
    let title: String
    var subtitle: String? = nil
    var onTitleTap: (() -> Void)? = nil
    @ViewBuilder let avatar: () -> AvatarView
    @ViewBuilder let actions: () -> Actions

    var body: some View {
        HStack(spacing: 12) {
            Button { onTitleTap?() } label: {
                HStack(spacing: 12) {
                    avatar()
                    VStack(alignment: .leading, spacing: 1) {
                        Text(title).font(.system(size: 16, weight: .semibold)).lineLimit(1)
                        if let subtitle {
                            Text(subtitle).font(.system(size: 11.5))
                                .foregroundStyle(Theme.textSecondary).lineLimit(1)
                        }
                    }
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .disabled(onTitleTap == nil)
            Spacer(minLength: 12)
            HStack(spacing: 6) { actions() }
        }
        .padding(.horizontal, 20)
        .frame(height: 64)
    }
}

/// "26 Aug" style separator between days in a thread.
struct DaySeparator: View {
    let date: Date

    var body: some View {
        Text(Self.label(for: date))
            .font(.system(size: 12, weight: .medium))
            .foregroundStyle(Theme.textSecondary)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 10)
    }

    static func label(for date: Date) -> String {
        let cal = Calendar.current
        if cal.isDateInToday(date) { return "Today" }
        if cal.isDateInYesterday(date) { return "Yesterday" }
        let sameYear = cal.isDate(date, equalTo: .now, toGranularity: .year)
        return date.formatted(sameYear ? .dateTime.day().month(.abbreviated)
                                       : .dateTime.day().month(.abbreviated).year())
    }
}

/// Day separators and message times in a thread.
enum ThreadLayout {
    /// A new day starts at `index`.
    static func startsDay(_ messages: [ChatMessage], _ index: Int) -> Bool {
        index == 0 || !Calendar.current.isDate(messages[index].date,
                                               inSameDayAs: messages[index - 1].date)
    }

    /// Send time shown beside a hovered message: "14:37" today, "26 Aug, 11:22" before.
    static func hoverTime(_ date: Date) -> String {
        let time = date.formatted(date: .omitted, time: .shortened)
        return Calendar.current.isDateInToday(date) ? time
            : "\(DaySeparator.label(for: date)), \(time)"
    }
}

/// The centred notice at the top of every thread (the design's E2E line). It
/// states what is actually true of this app: AI features run locally; the chat
/// partner's own replies come from a remote API.
struct OnDeviceNotice: View {
    var body: some View {
        VStack(spacing: 4) {
            HStack(spacing: 6) {
                Image(systemName: "cpu")
                Text("Translation, suggestions and safety checks run \(Text("on this Mac").fontWeight(.semibold))")
            }
            Text("They never send your messages anywhere for AI processing.")
        }
        .font(.system(size: 12))
        .foregroundStyle(Theme.textSecondary)
        .multilineTextAlignment(.center)
        .frame(maxWidth: .infinity)
        .padding(.top, 24).padding(.bottom, 8)
    }
}

/// Rounded "card" treatment for the banners that sit above a thread.
extension View {
    func bannerCard(_ tint: Color = Theme.accent) -> some View {
        self
            .padding(.horizontal, 14).padding(.vertical, 10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(tint.opacity(0.13), in: RoundedRectangle(cornerRadius: 12))
            .padding(.horizontal, 20).padding(.top, 8)
    }
}

/// Pill-shaped suggestion chip.
struct SuggestionChip: View {
    let text: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(text)
                .font(.system(size: 13))
                .lineLimit(1)
                .padding(.horizontal, 12).padding(.vertical, 6)
                .overlay(Capsule().stroke(Theme.accent.opacity(0.6)))
                .contentShape(Capsule())
        }
        .buttonStyle(.plain)
    }
}

extension View {
    /// Header-sized icon for a plain `Button` whose label switches to a spinner.
    func headerButtonStyle() -> some View {
        self
            .font(.system(size: 17))
            .frame(width: 34, height: 34)
            .contentShape(Rectangle())
            .buttonStyle(.plain)
    }

    /// Same, for a `Menu` (no chevron, no bezel).
    func headerMenuStyle() -> some View {
        self
            .menuStyle(.borderlessButton)
            .menuIndicator(.hidden)
            .font(.system(size: 17))
            .frame(width: 34, height: 34)
            .fixedSize()
    }
}
