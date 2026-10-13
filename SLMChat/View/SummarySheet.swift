import SwiftUI

/// Presents the on-device summary of unread messages.
struct SummarySheet: View {
    let summary: String
    /// True while the summary is still streaming in — shows a spinner until the
    /// first token arrives (during model prefill), instead of "(no summary)".
    var isStreaming: Bool = false
    /// Which chunk the summarizer is on, when a long backlog is being folded.
    var progress: String? = nil
    /// Display name of the on-device model that produced the summary (user-chosen).
    var modelName: String = "the on-device model"
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    Label("Summary of unread messages", systemImage: "text.append")
                        .font(.headline)
                    if summary.isEmpty && isStreaming {
                        HStack(spacing: 8) {
                            ProgressView()
                            Text(progress ?? "Summarizing…").foregroundStyle(.secondary)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                    } else {
                    Text(summary.isEmpty ? "(no summary)" : summary)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    Text("Generated on-device by \(modelName).")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding()
            }
            .navigationTitle("Summary")
            .inlineNavigationTitle()
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
        .presentationDetents([.medium])
    }
}

/// Presents the on-device summary of ONE long message, in the same sheet style as
/// the whole-conversation summary. Observes the message so it flips from a
/// "Summarizing…" spinner to the summary the moment the chunked summarizer fills it.
struct MessageSummarySheet: View {
    @ObservedObject var message: ChatMessage
    /// Display name of the on-device model that produced the summary (user-chosen).
    var modelName: String = "the on-device model"
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    Label("Message summary", systemImage: "text.append")
                        .font(.headline)
                    if message.isSummarizingLong && message.longSummary == nil {
                        HStack(spacing: 8) {
                            ProgressView()
                            // Naming the step matters here: a multi-chunk message
                            // is several model calls and can run far longer than a
                            // single one, so an unlabelled spinner reads as a hang.
                            Text(message.summaryProgress ?? "Summarizing…")
                                .foregroundStyle(.secondary)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                    } else {
                        Text((message.longSummary?.isEmpty == false) ? message.longSummary! : "(no summary)")
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    Text("Generated on-device by \(modelName).")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding()
            }
            .navigationTitle("Summary")
            .inlineNavigationTitle()
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
        .presentationDetents([.medium])
    }
}
