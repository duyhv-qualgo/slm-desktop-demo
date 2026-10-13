import SwiftUI

/// One chat bubble with a per-message Translate action. Observes the message so
/// streamed text and lazily-filled translations update live.
struct MessageBubble: View {
    @ObservedObject var message: ChatMessage
    /// The user's native language (Settings). A message already in this language
    /// needs no translation, so the Translate control is hidden for it.
    var nativeLanguage: TranslateLanguage
    /// Fired when the user taps Translate. The view model runs the SLM and fills
    /// `message.translation` / toggles `showTranslation`.
    var onTranslate: () -> Void
    /// Fired to drop a cached translation and regenerate it (Retranslate).
    var onRetranslate: () -> Void = {}
    /// Fired when the user picks "Summarize" from a long message's long-press menu.
    /// The view model runs the chunked summarizer and fills `message.longSummary` /
    /// toggles `showLongSummary`. Defaults to a no-op so screens that don't wire it
    /// (nothing to summarize) still compile.
    var onSummarize: () -> Void = {}
    /// Fallback name for the person being reported, used in a 1:1 thread where
    /// messages carry no `authorName`. Groups don't need it.
    var contactName: String? = nil
    /// Fired when the user submits the report sheet, with the reason they chose
    /// and any note. Defaults to a no-op so previews and screens that don't handle
    /// reports still compile.
    var onReport: (ReportReason, String) -> Void = { _, _ in }

    /// Lazily-loaded attached image (bytes live in `ImageStore`, off the message).
    @State private var loadedImage: PlatformImage?

    /// Pointer is over this message — reveals the secondary inline action.
    @State private var hovering = false

    /// Whether the report sheet is up.
    @State private var reporting = false

    /// Whether a cropped long message is expanded to its full text. Transient UI
    /// state — a long bubble starts collapsed to `collapsedLineLimit` lines.
    @State private var expanded = false
    /// How many lines a long message shows before it's cropped with a "Show more".
    private static let collapsedLineLimit = 6

    private var isMine: Bool { message.sender == .you }
    /// Only serious severities hide the content — see `ChatMessage.isSensitive`,
    /// which the inbox preview shares so the two can't disagree.
    private var isSevere: Bool { message.isSensitive }
    /// Flagged content is hidden until the user taps to reveal it.
    private var blurred: Bool { isSevere && !message.revealUnsafe }
    /// Any non-Clean classification (Caution included) — drives the class·severity
    /// label under the bubble, whether or not the body is blurred.
    private var isFlagged: Bool { message.safety?.isUnsafe == true }

    /// Whether to offer Report.
    ///
    /// **Any message from someone else.** It used to require Critical, and that
    /// was wrong in a way the radicalisation scenario makes concrete: the
    /// classifier catches 2 of its 8 lines and misses the recruitment and the
    /// purge-organising, so a Critical-only gate refused a report on exactly the
    /// six a person would want escalated. A report is not the classifier's
    /// judgement — it is a human saying "someone should look at this" — so making
    /// it conditional on the classifier agreeing gets the dependency backwards.
    ///
    /// `isMine` stays excluded: a report is an assertion about another person's
    /// conduct, and reporting yourself submits nothing anyone can act on.
    private var canReport: Bool { !isMine }

    /// A message long enough to offer the per-message "Summarize" action (see
    /// `onSummarize`). Gated by a rough token estimate so short texts never show it.
    private var isLong: Bool { Prompts.isLongMessage(message.text) }

    /// Menu title reflects the summary's cached state: generate it, or re-open it.
    private var summarizeMenuTitle: String {
        message.longSummary == nil ? "Summarize" : "View summary"
    }

    /// "Class · Severity" (e.g. "Child safety · Critical") for the flag labels.
    private var flagLabel: String {
        let cls = message.safety?.displayName ?? "Flagged"
        if let sev = message.severity, sev != .safe {
            return "\(cls) · \(sev.displayName)"
        }
        return cls
    }

    /// Tint for the flag, keyed on how serious the severity is.
    private var severityColor: Color {
        switch message.severity {
        case .critical:  return .red
        case .violation: return .orange
        case .caution:   return .yellow
        default:         return .orange
        }
    }

    /// True when the bubble is predominantly in the user's native language, so
    /// there's little worth translating and the control is suppressed.
    ///
    /// Identity first, ratio second. `vietnameseRatio` alone can't answer this
    /// question for the languages the app now supports: it is a Vietnamese-vs-
    /// English word heuristic, and it abstains (nil) on Chinese, Russian and
    /// Arabic — which this predicate reads as "undetermined, treat as native" and
    /// so HIDES Translate on exactly the messages that most need it. (It used to
    /// show, but only by accident: before non-Latin scripts learned to abstain,
    /// a Chinese sentence scored 0.0 and passed for English.)
    ///
    /// So: if the language is identified and isn't the native one, offer Translate.
    /// The ratio still decides the remaining case — text identified AS native —
    /// because it grades code-switching (≥75% native → hide), which a single
    /// language label can't: a half-English Vietnamese message identifies as one
    /// language but is worth translating. Undetermined text (empty, still
    /// streaming, or filler-only) counts as native, so the button doesn't flash.
    private var isNativeLanguage: Bool {
        if let detected = message.detectedLanguage, detected != nativeLanguage {
            return false
        }
        guard let vi = message.vietnameseRatio else { return true }
        let nativeRatio = nativeLanguage == .vietnamese ? vi : 1 - vi
        return nativeRatio >= 0.75
    }

    var body: some View {
        HStack {
            if isMine { Spacer(minLength: 120) }
            VStack(alignment: isMine ? .trailing : .leading, spacing: 4) {
                if let name = message.authorName, !isMine {
                    Text(name)
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(Avatar.tint(name))
                }
                // Translate sits inline on the bubble's outer side — right of an
                // incoming bubble, left of your own — bottom-aligned with it.
                HStack(alignment: .bottom, spacing: 6) {
                    if isMine { hoverAccessories }
                    bubble
                        // The sheet sets `message.reported` on submit, not on open —
                        // opening the form and backing out must not count as reporting.
                        .sheet(isPresented: $reporting) {
                            ReportSheet(message: message,
                                        author: message.authorName ?? contactName ?? "this contact",
                                        onSubmit: onReport)
                        }
                    if !isMine { hoverAccessories }
                }
                .onHover { hovering = $0 }
            }
            // Long lines wrap at a readable width on a wide window.
            .frame(maxWidth: 560, alignment: isMine ? .trailing : .leading)
            if !isMine { Spacer(minLength: 120) }
        }
    }

    /// The bubble, with a long-press context menu attached only when there's text
    /// to act on (Copy / Summarize). Attachment-only bubbles get no menu.
    @ViewBuilder
    private var bubble: some View {
        // Attach the menu when there is anything to act on. Text-only used to be
        // the condition, which left an image-only bubble — a very likely thing to
        // want to report — with no menu at all.
        if message.text.isEmpty && !canReport {
            bubbleContent
        } else {
            bubbleContent.contextMenu { bubbleMenu }
        }
    }

    /// Long-press actions: always Copy the text; Summarize only for long messages.
    @ViewBuilder
    private var bubbleMenu: some View {
        if !message.text.isEmpty {
            Button {
                Clipboard.copy(message.text)
            } label: {
                Label("Copy", systemImage: "doc.on.doc")
            }
        }
        if isLong {
            Button(action: onSummarize) {
                Label(summarizeMenuTitle, systemImage: "text.append")
            }
        }
        if canReport {
            // `.destructive` so it reads as consequential and sorts to the bottom,
            // away from Copy — this is the one action here that tells another
            // person something.
            Divider()
            Button(role: .destructive) {
                reporting = true
            } label: {
                Label(message.reported ? "Reported" : "Report", systemImage: "flag")
            }
            .disabled(message.reported)
        }
    }

    private var bubbleContent: some View {
        VStack(alignment: .leading, spacing: 6) {
            if message.hasImage { attachedImage }
            if message.hasFile { attachedFile }

            // Hide the text row for an attachment with no caption (and the "…"
            // streaming placeholder only applies to plain text bubbles).
            if (!message.hasImage && !message.hasFile) || !message.text.isEmpty {
            ZStack {
                Text(message.text.isEmpty ? "…" : message.text)
                    .font(.system(size: 14))
                    .foregroundStyle(isMine ? .white : .primary)
                    // A long message is cropped to a few lines so it doesn't fill
                    // the thread; the user expands it or reads the summary instead.
                    .lineLimit(isLong && !expanded ? Self.collapsedLineLimit : nil)
                    .blur(radius: blurred ? 7 : 0)
                    // A blur is a VISUAL effect only: without this, VoiceOver
                    // reads the hidden text aloud verbatim and the guardrail does
                    // nothing for anyone using assistive technology. The reveal
                    // button below stays reachable and carries the class and
                    // severity, so the content is still available deliberately.
                    .accessibilityHidden(blurred)
                if blurred {
                    Button { message.revealUnsafe = true } label: {
                        Label("\(flagLabel) — tap to reveal",
                              systemImage: "eye.slash.fill")
                            .font(.caption2.weight(.medium))
                            .foregroundStyle(isMine ? .white : .primary)
                    }
                    .buttonStyle(.plain)
                }
            }
            // Show more / less for a cropped long message.
            if isLong && !blurred {
                Button { withAnimation(.easeInOut(duration: 0.15)) { expanded.toggle() } } label: {
                    Text(expanded ? "Show less" : "Show more")
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(isMine ? Color.white.opacity(0.9) : Theme.accent)
                }
                .buttonStyle(.plain)
            }
            }

            // Show the class · severity for any flagged message that isn't blurred.
            // When it's blurred, the reveal overlay already carries this label.
            if isFlagged && !blurred {
                Label("Flagged: \(flagLabel)",
                      systemImage: "exclamationmark.triangle.fill")
                    .font(.caption2)
                    .foregroundStyle(isMine ? Color.white.opacity(0.9) : severityColor)
            }

            if message.reported { reportedMarker }

            if message.showTranslation, let t = message.translation {
                Divider().overlay(isMine ? Color.white.opacity(0.4) : Color.gray.opacity(0.4))
                Text(t)
                    .font(.callout)
                    .foregroundStyle(isMine ? Color.white.opacity(0.9) : .secondary)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(isMine ? Theme.accent : Theme.bubbleIn)
        .clipShape(RoundedRectangle(cornerRadius: 14))
        // Don't let selection reveal blurred content. For long messages we prefer
        // the long-press menu (Copy / Summarize), so selection is off there — it
        // otherwise competes with the context-menu long-press gesture.
        .textSelectable(!blurred && !isLong)
    }

    /// Confirmation that this message was already reported.
    ///
    /// Shown in the bubble rather than only in the menu, so the state is visible
    /// without a long press — otherwise the only way to find out whether you had
    /// already reported something is to open the menu and read the label.
    private var reportedMarker: some View {
        Label("Reported", systemImage: "checkmark.shield.fill")
            .font(.caption2.weight(.medium))
            .foregroundStyle(isMine ? Color.white.opacity(0.9) : .secondary)
    }

    /// The attached image. Tagging still runs on-device in the background (so the
    /// photo stays searchable offline) — the predicted tags are just not shown as a
    /// caption under the bubble.
    @ViewBuilder
    private var attachedImage: some View {
        Group {
            if let ui = loadedImage {
                Image(platformImage: ui)
                    .resizable()
                    .scaledToFit()
                    .frame(maxWidth: 240, maxHeight: 260)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
            } else {
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color.gray.opacity(0.2))
                    .frame(width: 180, height: 180)
                    .overlay(ProgressView())
            }
        }
        .task(id: message.imageID) { await loadImage() }
    }

    /// A document attachment row: icon, filename, and human-readable size.
    private var attachedFile: some View {
        HStack(spacing: 10) {
            Image(systemName: "doc.fill")
                .font(.title2)
                .foregroundStyle(isMine ? Color.white : Color.accentColor)
                .frame(width: 32)
            VStack(alignment: .leading, spacing: 2) {
                Text(message.fileName ?? "File")
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(isMine ? .white : .primary)
                    .lineLimit(1)
                if let size = message.fileSize {
                    Text(ByteCountFormatter.string(fromByteCount: Int64(size), countStyle: .file))
                        .font(.caption2)
                        .foregroundStyle(isMine ? Color.white.opacity(0.8) : .secondary)
                }
            }
        }
        .frame(maxWidth: 240, alignment: .leading)
    }

    /// Load the attached image off the main thread (bytes come from `ImageStore`).
    private func loadImage() async {
        guard loadedImage == nil, let id = message.imageID else { return }
        let img = await Task.detached(priority: .utility) {
            ImageStore.data(for: id).flatMap(PlatformImage.init(data:))
        }.value
        loadedImage = img
    }

    /// Send time + translate, revealed while the pointer is over the message.
    /// Always in the layout (opacity, not insertion) so the row never shifts;
    /// a running translation keeps its spinner visible.
    private var hoverAccessories: some View {
        HStack(spacing: 4) {
            if isMine { timeLabel }
            translationControls
            if !isMine { timeLabel }
        }
        .opacity(hovering || message.isTranslating ? 1 : 0)
        .allowsHitTesting(hovering)
        .animation(.easeOut(duration: 0.12), value: hovering)
    }

    private var timeLabel: some View {
        Text(ThreadLayout.hoverTime(message.date))
            .font(.system(size: 11))
            .foregroundStyle(Theme.textSecondary)
            .fixedSize()
            .padding(.bottom, 6)
    }

    @ViewBuilder
    private var translationControls: some View {
        // A message already in the user's native language needs no translation.
        // Hide the control — unless one was already generated (e.g. before the
        // native language changed), so the user can still toggle it off.
        if isNativeLanguage && message.translation == nil {
            EmptyView()
        } else {
            HStack(spacing: 2) {
                if message.isTranslating {
                    ProgressView().controlSize(.small)
                        .frame(width: 26, height: 26)
                        .help("Translating…")
                } else {
                    // One control. Showing a translation always runs a fresh one
                    // (the cached text is never re-shown); while one is visible,
                    // the same button hides it.
                    InlineIconButton(systemImage: "character.bubble",
                                     help: message.showTranslation ? "Hide translation" : "Translate",
                                     active: message.showTranslation) {
                        if message.showTranslation {
                            message.showTranslation = false
                        } else {
                            onRetranslate()
                        }
                    }
                }
            }
        }
    }
}

private extension View {
    /// Conditionally enable text selection while keeping a single result type.
    @ViewBuilder
    func textSelectable(_ enabled: Bool) -> some View {
        if enabled { self.textSelection(.enabled) }
        else { self.textSelection(.disabled) }
    }
}

/// Small round icon button that sits beside a bubble (translate, retranslate).
private struct InlineIconButton: View {
    let systemImage: String
    let help: String
    var active = false
    let action: () -> Void
    @State private var hovering = false

    var body: some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(active ? Theme.accent : Theme.textSecondary)
                .frame(width: 26, height: 26)
                .background(hovering || active ? Theme.rowHover : .clear, in: Circle())
                .contentShape(Circle())
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
        .help(help)
    }
}
