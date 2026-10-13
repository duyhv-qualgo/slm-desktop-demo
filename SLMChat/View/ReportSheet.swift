import SwiftUI

/// The report form for a message.
///
/// A sheet rather than an immediate submit, because a one-tap report is a worse
/// product in both directions: the user learns nothing about what they just did,
/// and the report arrives with no reason attached, which is the field that makes
/// it triageable at all.
struct ReportSheet: View {
    @ObservedObject var message: ChatMessage
    /// Who is being reported — the contact, or the group member who wrote it.
    var author: String
    /// Called with the reason and note when the user submits.
    var onSubmit: (ReportReason, String) -> Void

    @Environment(\.dismiss) private var dismiss

    /// Pre-selected from the classifier's verdict so the common case is confirm
    /// and send. It is a SUGGESTION — the user can change it, and when they do,
    /// that disagreement is the most useful field in the report.
    @State private var reason: ReportReason
    @State private var note: String = ""
    /// Whether the user has chosen to look at content the THREAD was hiding.
    /// Opening the form must not do what the blur was there to prevent.
    @State private var revealed = false

    init(message: ChatMessage,
         author: String,
         onSubmit: @escaping (ReportReason, String) -> Void) {
        self.message = message
        self.author = author
        self.onSubmit = onSubmit
        _reason = State(initialValue: .suggested(for: message.safety))
    }

    /// True when the guardrail flagged this at all. Reporting no longer requires
    /// it — a message the classifier cleared is still reportable, and those are
    /// the reports most worth reading.
    private var wasFlagged: Bool { message.safety?.isUnsafe == true }

    private var verdictLabel: String {
        guard wasFlagged else { return "Not flagged by safety checks" }
        let cls = message.safety?.displayName ?? "Flagged"
        let sev = message.severity?.displayName ?? ""
        return sev.isEmpty ? cls : "\(cls) · \(sev)"
    }

    /// True when the user picked something other than what the model said. Only
    /// meaningful when there was a verdict to disagree with.
    private var disagrees: Bool {
        wasFlagged && ReportReason.suggested(for: message.safety) != reason
    }

    var body: some View {
        NavigationStack {
            Form {
                messageSection
                reasonSection
                noteSection
                disclosureSection
            }
            .navigationTitle("Report message")
            .inlineNavigationTitle()
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Submit") {
                        message.reported = true
                        onSubmit(reason, note)
                        dismiss()
                    }
                    .fontWeight(.semibold)
                }
            }
        }
        .presentationDetents([.medium, .large])
    }

    /// What is being reported. Blurred content keeps its own reveal here, so a
    /// user can file a report about a message they chose not to read.
    private var messageSection: some View {
        Section {
            HStack(spacing: 8) {
                Image(systemName: wasFlagged ? "exclamationmark.triangle.fill"
                                             : "checkmark.shield")
                    .foregroundStyle(wasFlagged ? .red : .secondary)
                VStack(alignment: .leading, spacing: 2) {
                    Text(author).font(.subheadline.weight(.semibold))
                    Text(verdictLabel).font(.caption).foregroundStyle(.secondary)
                }
            }
            // Hidden only if the thread was hiding it. Re-hiding a message the
            // user was already reading is theatre, and it costs them the context
            // they need to pick a reason.
            if revealed || !message.isSensitive {
                Text(message.text)
                    .font(.callout)
                    .foregroundStyle(.secondary)
            } else {
                Button("Show message") { revealed = true }
                    .font(.callout)
            }
        } header: {
            Text("Reporting")
        }
    }

    /// The dropdown. `.menu` rather than a pushed list so the whole form stays on
    /// one screen — a report abandoned halfway through a navigation stack is a
    /// report that doesn't get filed.
    private var reasonSection: some View {
        Section {
            Picker("Reason", selection: $reason) {
                ForEach(ReportReason.allCases) { r in
                    Text(r.title).tag(r)
                }
            }
            .pickerStyle(.menu)
        } header: {
            Text("Why are you reporting this?")
        } footer: {
            if disagrees {
                // Surfaced rather than silently recorded. The user corrected the
                // classifier, and saying so is both honest and the thing that
                // makes the correction feel worth making.
                Label("Thanks — you've picked something different from what we "
                    + "detected. That helps us improve.",
                      systemImage: "sparkles")
            } else if wasFlagged {
                Text("Pre-filled from what was detected on your device. Change it "
                   + "if it's wrong.")
            } else {
                // The most valuable report there is: the guardrail found nothing
                // and a person disagreed. Say so, so it doesn't feel futile.
                Label("Our safety checks didn't flag this one. Reports like yours "
                    + "are how we catch what they miss.",
                      systemImage: "sparkles")
            }
        }
    }

    /// Caps the note as it is typed.
    ///
    /// A binding rather than `onChange`, which would need iOS 17 for the two-value
    /// form and lets an over-long value exist for one render either way. `submit`
    /// truncates again — this is the courtesy, that is the rule.
    private var cappedNote: Binding<String> {
        Binding(get: { note },
                set: { note = String($0.prefix(GuardReport.maxNoteLength)) })
    }

    private var noteSection: some View {
        Section {
            TextField("Add anything that helps (optional)",
                      text: cappedNote, axis: .vertical)
                .lineLimit(2...5)
        } footer: {
            Text("\(note.count)/\(GuardReport.maxNoteLength)")
                .monospacedDigit()
        }
    }

    /// What actually leaves the device.
    ///
    /// Stated plainly because this app is end-to-end encrypted and a report is the
    /// one thing that crosses the wire. A user who can't tell what a report
    /// discloses will either not file one or file one believing something untrue;
    /// both are worse than saying it.
    private var disclosureSection: some View {
        Section {
            Label("The reason, the category, and who sent it",
                  systemImage: "checkmark.circle")
            Label("Not your other messages, and not this conversation",
                  systemImage: "xmark.circle")
        } header: {
            Text("What gets sent")
        } footer: {
            Text("Reports are reviewed by a person. Nothing is sent automatically "
               + "— safety checks run on your device and stay there.")
        }
        .font(.callout)
    }
}
