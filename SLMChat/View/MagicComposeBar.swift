import SwiftUI

/// The Magic Compose surface: one row of instruction chips, and a status line
/// above it when there is something to report.
///
/// Tapping a chip writes its result **straight into the chat input box** — there
/// is no preview and no confirmation step. Undo is what makes that safe, so the
/// bar stays open after a rewrite lands rather than dismissing itself.
///
/// Every chip rewrites the ORIGINAL draft, captured when the bar opened, not
/// whatever the last chip produced. So switching Shorten → Professional gives a
/// professional version of what the user wrote, not a professional version of the
/// shortened one — see `Conversation.composeDraft`.
///
/// Chips are the only way in; there is no free-typed instruction, because every
/// string here is one the finetune was trained on (`Prompts.rewriteChips` /
/// `smartReplyChips`).
///
/// State lives on the `Conversation` (owned by the store), so a generation that is
/// still running survives scrolling and the keyboard cycling.
struct MagicComposeBar: View {
    @ObservedObject var conversation: Conversation
    /// Only for the language-pack install offered on the missing-pack path —
    /// nothing else here reads app state.
    @ObservedObject var store: ConversationStore
    let isReady: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            statusLine
            HStack(spacing: 8) {
                chipRow
                // Outside the scroll view so it can't scroll away — it is the
                // only way out of the bar besides Undo.
                Button { conversation.endCompose() } label: {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Close rewrite")
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(Color.accentColor.opacity(0.08))
        .overlay(alignment: .top) { Divider() }
        .transition(.move(edge: .bottom).combined(with: .opacity))
    }

    // MARK: - Status

    /// One line, and only when there is something to say. `.menu` renders nothing
    /// at all: the chips speak for themselves, and the draft they act on is
    /// already visible in the input box right below.
    @ViewBuilder
    private var statusLine: some View {
        switch conversation.composeState {
        case .idle, .menu:
            EmptyView()
        case .generating:
            HStack(spacing: 8) {
                ProgressView().controlSize(.small)
                Text(hasDraft ? "Rewriting\u{2026}" : "Writing a reply\u{2026}")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        case .applied:
            HStack(spacing: 6) {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundStyle(.green)
                Text("Replaced").font(.caption.weight(.medium))
                Spacer(minLength: 8)
                Button("Undo") { conversation.undoCompose() }
                    .font(.caption)
            }
        case .refused(let reason):
            // The design calls for a toast. Rendered inline instead: the message
            // is about the chip the user just tapped, and it sits directly above
            // that chip here, where a toast would float away from it.
            notice(icon: "exclamationmark.shield",
                   tint: .orange,
                   text: Self.refusalMessage(reason))
        case .failed(let message):
            HStack(spacing: 6) {
                notice(icon: conversation.missingPack == nil
                       ? "exclamationmark.triangle" : "arrow.down.circle",
                       tint: .secondary, text: message)
                // The install lives HERE for the same reason it lives in the
                // suggestion row: Settings offers a Get button for the NATIVE
                // language's pack alone, so a Vietnamese-native user asked to
                // install the Spanish pack has no other route to it.
                if let pack = conversation.missingPack {
                    packInstallButton(pack)
                }
            }
        }
    }

    /// Whether there is a draft to rewrite — the one input that decides which chip
    /// set and which prompt shape this bar uses.
    ///
    /// Read from the conversation rather than the input field, because the
    /// conversation holds the draft as CAPTURED when the bar opened, and that is
    /// the text the prompt carries. Once a rewrite lands, the field holds the
    /// RESULT while this still reports on the original — which is what keeps the
    /// chip set from flipping to smart-reply mid-session.
    private var hasDraft: Bool {
        !conversation.composeDraft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    /// What the model declined to write. The reason tag is a machine token
    /// (`HARMFUL_INSTRUCTION`), so it is mapped to copy rather than shown raw.
    static func refusalMessage(_ reason: String?) -> String {
        switch reason {
        case "HARMFUL_INSTRUCTION":
            return "Content violates safety standards."
        case .some(let other) where !other.isEmpty:
            // An unknown tag still means the model refused. Say so plainly rather
            // than dropping to a generic error the user would retry forever.
            let phrase = other.lowercased().replacingOccurrences(of: "_", with: " ")
            return "That rewrite was declined (\(phrase))."
        default:
            return "That rewrite was declined."
        }
    }

    /// Get / progress / retry for the pack blocking this rewrite. Mirrors the
    /// suggestion row's button, minus the states that can't occur here.
    @ViewBuilder
    private func packInstallButton(_ pack: LoRAAdapter) -> some View {
        switch store.loraStore.state(of: pack) {
        case .downloading(let progress):
            if let progress {
                ProgressView(value: progress).progressViewStyle(.linear).frame(width: 60)
            } else {
                ProgressView().controlSize(.small)
            }
        case .installed:
            // The download finished while this message was on screen. Re-running
            // the chip now resolves to the pack rather than the block.
            Button("Retry") { conversation.retryCompose() }
                .font(.caption)
        case .failed, .notInstalled:
            Button("Get") { store.loraStore.download(pack) }
                .font(.caption)
        }
    }

    private func notice(icon: String, tint: Color, text: String) -> some View {
        HStack(spacing: 6) {
            Image(systemName: icon).foregroundStyle(tint)
            Text(text).font(.caption)
            Spacer(minLength: 0)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: - Chips

    private var chipRow: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(conversation.composeChips) { chip in
                    let active = conversation.composeInstruction?.id == chip.id
                    Button { conversation.compose(chip) } label: {
                        Label(chip.label, systemImage: chip.icon)
                            .font(.caption.weight(active ? .semibold : .regular))
                            .lineLimit(1)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 6)
                            .background(active ? Color.accentColor.opacity(0.30)
                                               : Color.accentColor.opacity(0.14))
                            .clipShape(Capsule())
                    }
                    .buttonStyle(.plain)
                    .disabled(!isReady || conversation.isComposeGenerating)
                }
            }
            .padding(.vertical, 1)   // room for the capsule edges to draw
        }
    }
}
