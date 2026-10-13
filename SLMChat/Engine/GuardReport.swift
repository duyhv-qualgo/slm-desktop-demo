import Foundation

/// Why the recipient says they are reporting.
///
/// User-facing wording, mapped to the classifier's taxonomy — but the mapping is
/// one-way and advisory. The classifier's label pre-selects a reason; the reason
/// the user actually submits is theirs, and where the two disagree, that
/// disagreement is the most useful thing in the report: it is unlabelled
/// evaluation data arriving from real traffic.
///
/// **`political` is deliberately absent**, and must stay absent. Vietnam's rule
/// set carries 114 political patterns and the UK carries zero precisely because
/// political criticism is protected expression (UK_COMPLIANCE.md §3). Offering
/// "political content" as a reportable reason would hand every user a button for
/// silencing dissent, which no amount of downstream review makes safe.
enum ReportReason: String, CaseIterable, Identifiable {
    case harassment
    case violence
    case suicide
    case hate
    case sexual
    case child
    case scam
    case illegal
    case spam
    case other

    var id: String { rawValue }

    var title: String {
        switch self {
        case .harassment:   return "Bullying or harassment"
        case .violence:     return "Violence or threats"
        case .suicide:     return "Suicide or self-harm"
        case .hate:         return "Hate speech"
        case .sexual:       return "Nudity or sexual content"
        case .child:  return "Child safety"
        case .scam:    return "Scam or fraud"
        case .illegal: return "Illegal goods or drugs"
        case .spam:         return "Spam"
        case .other:        return "Something else"
        }
    }

    /// The reason to pre-select for a given verdict, so the common case is one tap.
    ///
    /// Falls back to `.other` rather than guessing — including for `political`,
    /// which has no reason to map to and must not acquire one.
    static func suggested(for label: SafetyLabel?) -> ReportReason {
        switch label {
        case .harassment:   return .harassment
        case .violence:     return .violence
        case .suicide:     return .suicide
        case .hate:         return .hate
        case .sexual:       return .sexual
        case .child:  return .child
        case .scam:    return .scam
        case .illegal: return .illegal
        default:            return .other
        }
    }
}

/// A recipient's report of a message the guardrail flagged Critical.
///
/// This is the one place in the design where a verdict is allowed to leave the
/// device, and the shape of it is the whole argument. Everything else the
/// guardrail produces stays local: sender-side classification is advisory to the
/// sender, and the classifier's verdicts are never telemetry. What crosses the
/// wire is a report a HUMAN chose to send about a message they received.
///
/// That distinction is what separates this from client-side scanning. A system
/// that uploaded every Critical verdict automatically would be exactly the
/// capability Apple proposed in August 2021 and abandoned in December 2022, and
/// exactly what OSA s.121 contemplates — see UK_COMPLIANCE.md §5. The guardrail
/// can raise the prompt; only the recipient can send it.
///
/// **This is a sample app with no server**, so `submit` logs. The fields are the
/// ones a real submission would need, and the gaps are noted below rather than
/// left for someone to discover after wiring a backend to it.
enum GuardReport {

    /// What a report carries. Deliberately NOT the message text.
    struct Payload {
        /// What the guardrail said, or **nil when it said nothing** — the message
        /// was clean, or never screened.
        ///
        /// Optional because the reports on unflagged messages are the valuable
        /// ones. Measured on the radicalisation scenario, the classifier catches
        /// 2 lines of 8 and misses the recruitment and the purge-organising; a
        /// reporting flow that only accepted its own flags would refuse exactly
        /// the six a human needed to escalate.
        let contentClass: SafetyLabel?
        let severity: Severity?
        /// What the REPORTER said it was. Distinct from `contentClass`, which is
        /// what the model said — see `reasonMatchesVerdict`.
        let reason: ReportReason
        /// The reporter's own words, trimmed and length-capped. Empty when they
        /// added nothing.
        let note: String
        /// Who sent the reported message: the contact, or a group member's name.
        let reportedAuthor: String
        /// Where it happened, for the reporter's own record.
        let context: String
        /// Whether a rule or the classifier raised it — a rule hit is reproducible
        /// from the pattern, a model verdict is not, and a reviewer needs to know
        /// which kind of claim they are looking at.
        let source: SafetySource
        let at: Date

        /// Whether the reporter agreed with the classifier, or nil if there was
        /// no verdict to agree with.
        ///
        /// The single most valuable field for improving a pack: a disagreement is
        /// a labelled example of the model being wrong, volunteered by someone who
        /// read the message, and it arrives from real traffic rather than a test
        /// set. `nil` is the strongest case of all — the model said nothing and a
        /// human still thought it was worth reporting.
        var reasonMatchesVerdict: Bool? {
            guard let contentClass, contentClass.isUnsafe else { return nil }
            return ReportReason.suggested(for: contentClass) == reason
        }

        /// True when the guardrail missed this entirely. The set worth reviewing
        /// first, both for moderation and for the next pack.
        var isMissedByGuardrail: Bool {
            contentClass == nil || contentClass?.isUnsafe == false
        }
    }

    /// Longest note kept. A report is a signal, not a support ticket; unbounded
    /// free text is also the easiest way for message content to end up somewhere
    /// this design promised it wouldn't.
    static let maxNoteLength = 300

    /// Record a report. Returns the payload so callers can test what would be sent.
    ///
    /// ⚠️ Missing before this could face a real backend, all of which are product
    /// decisions rather than plumbing:
    ///
    ///  - **Franking.** Without a sender-attributable MAC from the delivery
    ///    service, a report is an unverifiable claim: nothing stops a reporter
    ///    fabricating a message that was never sent. Message franking is the
    ///    standard construction and it has to exist in the transport, not here.
    ///  - **The content itself.** A reviewer cannot adjudicate a class name. Real
    ///    systems attach the reported ciphertext plus a key so the report opens
    ///    exactly that message and nothing else — which is a considered erosion of
    ///    E2EE for one message the recipient chose to disclose, not a side effect.
    ///  - **Rate limiting and abuse handling**, since coordinated reporting is the
    ///    obvious attack on any reputation signal built from this.
    @discardableResult
    static func submit(_ message: ChatMessage,
                       reason: ReportReason,
                       note: String = "",
                       author: String,
                       context: String,
                       source: SafetySource = .model) -> Payload? {
        // No severity gate. Reporting used to require Critical, on the reasoning
        // that Caution and Violation are evidence rather than proof and a button
        // there trains people to report ambiguity. That reasoning was about the
        // CLASSIFIER's confidence, and a report is not the classifier's to make —
        // it is a person saying "a human should look at this". Gating it on the
        // model's opinion means the model's misses are unreportable, which is
        // backwards: those are the ones most needing a human.
        //
        // The remaining rule is `MessageBubble.canReport` — not your own message.
        let contentClass = message.safety
        let severity = message.severity

        let trimmed = String(note.trimmingCharacters(in: .whitespacesAndNewlines)
                                 .prefix(maxNoteLength))
        let payload = Payload(contentClass: contentClass, severity: severity,
                              reason: reason, note: trimmed,
                              reportedAuthor: author, context: context,
                              source: source, at: Date())

        // No message text, here or anywhere downstream. The log is the one surface
        // that reliably escapes the device — into sysdiagnose, into crash
        // reporters — and printing flagged content into it would leak exactly what
        // the guardrail just decided to withhold.
        // The note is counted, not printed, for the same reason the text isn't:
        // it is the reporter's free text about a message and can quote it.
        let verdictText = contentClass.map {
            "\($0.displayName)/\(severity?.displayName ?? "-")"
        } ?? "none"
        let agreement: String
        switch payload.reasonMatchesVerdict {
        case .some(true):  agreement = "reporter agrees"
        case .some(false): agreement = "reporter DISAGREES"
        // The guardrail found nothing and a human reported it anyway. Called out
        // because this is the set worth reviewing first.
        case .none:        agreement = "GUARDRAIL MISS"
        }
        NSLog("[GuardReport] reason=%@ verdict=%@ (%@, %@) by %@ in %@ — "
            + "note %d chars — queued for submission",
              reason.rawValue, verdictText, source.logDescription, agreement,
              author, context, trimmed.count)
        return payload
    }
}
