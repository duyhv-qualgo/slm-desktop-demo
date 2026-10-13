import Foundation

/// Who authored a message. The demo simulates a 1:1 chat: `.you` is the device
/// owner, `.friend` is the on-device SLM persona.
enum Sender: String, Codable {
    case you
    case friend

    /// How the sender is labelled inside prompts handed to the model. The chat
    /// partner is "Friend" so the persona preamble and history read naturally.
    var promptLabel: String {
        switch self {
        case .you:    return "You"
        case .friend: return "Friend"
        }
    }
}

/// One chat bubble. A reference type so the view model can mutate translation
/// state in place (fill `translation`, flip `showTranslation`) without rebuilding
/// the whole array.
final class ChatMessage: Identifiable, ObservableObject {
    let id = UUID()
    let sender: Sender
    let date: Date

    @Published var text: String {
        didSet { _viRatioValid = false; _languageValid = false }   // text changed (streaming) → re-detect
    }
    /// Marked read once the user has "seen" it (opened summary / replied). Only
    /// `.friend` messages start unread; `.you` messages are read by definition.
    @Published var isRead: Bool
    /// Cached translation of `text`, filled lazily by the Translate action.
    @Published var translation: String?
    /// Whether the translated text is currently revealed under the original.
    @Published var showTranslation: Bool = false
    /// True while a translation request for this bubble is in flight.
    @Published var isTranslating: Bool = false

    /// Cached summary of a long `text`, produced on demand by the per-message
    /// "Summarize" long-press action (chunked map-reduce) and shown in a sheet.
    /// `nil` until requested; persisted so a long message is only ever summarized once.
    @Published var longSummary: String?
    /// True while a long-message summary request for this bubble is in flight.
    @Published var isSummarizingLong: Bool = false
    /// What that request is currently doing, for the sheet to show while there is
    /// no text yet — a long message is a map-reduce over several model calls and
    /// can run tens of seconds. Transient: derived state, never persisted, cleared
    /// when the summary lands.
    @Published var summaryProgress: String?

    /// Guardrail safety class, filled lazily by the background classifier. `nil`
    /// = not classified yet; stored as the raw `SafetyLabel` value so it persists.
    @Published var safetyLabel: Int?
    /// Guardrail severity level (head 2), stored as the raw `Severity` value.
    /// `nil` until classified. Read together with `safetyLabel`.
    @Published var severityLevel: Int?
    /// In a group chat, the name of the bot that authored this message. `nil` for
    /// 1:1 threads and for the user's own messages.
    var authorName: String?

    /// Identifier of an attached image in `ImageStore`, or `nil` for a text-only
    /// message. The bytes live on disk; this is just the reference.
    let imageID: String?
    /// Tags predicted for the attached image by the on-device image-tagging model,
    /// sorted by descending score. `nil` = not tagged yet; `[]` = tagged, no tags.
    /// Persisted so the image is only ever tagged once and stays searchable offline.
    @Published var imageTags: [String]?

    /// True when this bubble carries an image attachment.
    var hasImage: Bool { imageID != nil }

    /// Identifier of an attached document in `FileStore`, or `nil`. Distinct from
    /// `imageID`: images are "Media", documents are "Files".
    let fileID: String?
    /// Original display name of the attached document (e.g. "report.pdf").
    let fileName: String?
    /// Size of the attached document in bytes, for display.
    let fileSize: Int?

    /// True when this bubble carries a (non-image) document attachment.
    var hasFile: Bool { fileID != nil }
    /// Whether the user has tapped to reveal a message whose body is blurred
    /// because it was flagged unsafe. Transient (not persisted).
    @Published var revealUnsafe: Bool = false

    /// Whether the recipient has reported this message.
    ///
    /// **Persisted**, unlike `revealUnsafe`. A report is an assertion the user
    /// made about someone else's conduct, and forgetting it across a relaunch
    /// would put the button back and invite the same message being reported
    /// twice — which, in the design this feeds (§4), inflates the very counter
    /// enforcement would be based on.
    ///
    /// Recipient-attested by construction: it is only ever set by a tap. The
    /// classifier can raise the prompt but can never set this itself, which is
    /// what keeps sender-side telemetry out of the reputation signal.
    @Published var reported: Bool = false

    /// The classified safety label, if any.
    var safety: SafetyLabel? { safetyLabel.flatMap(SafetyLabel.init(rawValue:)) }

    /// The classified severity, if any.
    var severity: Severity? { severityLevel.flatMap(Severity.init(rawValue:)) }

    /// Whether this message's CONTENT should be withheld from view.
    ///
    /// **Critical only — which since v16 means RULE hits only.** The expression is
    /// unchanged, but what it selects is not: the v16 checkpoint has no severity
    /// head, so a model verdict carries `severity == nil` and can no longer reach
    /// `.critical`. Blurring is now driven entirely by hand-authored patterns in
    /// `guardrails/regions/*.yaml`, each of which carries the severity a person
    /// assigned it. The model still classifies, still labels the bubble, and still
    /// makes the message reportable — it just no longer hides anything.
    ///
    /// That is a deliberate consequence of `Severity` becoming rules-only rather
    /// than an accident, and it cuts both ways:
    ///
    ///   + It retires the false positive that prompted the change. "Cảm ơn cậu
    ///     nhiều nhé… Mình sẽ cố gắng nghỉ ngơi một chút." — gratitude and a plan
    ///     to rest — was Self-harm/Critical 0.879 under v15 and blurred. Hiding
    ///     someone's reach-out from the friend who might answer is the inverse of
    ///     what Ofcom's children's codes ask for. It cannot happen now: no model
    ///     verdict blurs.
    ///
    ///   − A harm the model catches and no pattern covers is labelled but stays
    ///     readable. The exposure is smallest exactly where it matters most —
    ///     `child` is the one category with patterns in every rule set, including
    ///     the universal fallback every region loads — and largest for languages
    ///     with a classifier but no authored rules (ar, ru, zh, and es/fr beyond
    ///     the 3 universal patterns). Closing it means writing patterns, which is
    ///     the honest fix; reconstructing severity from the class would only move
    ///     the guess somewhere less visible.
    ///
    /// Defined here, once, because it is asked in more than one place: the chat
    /// bubble blurs on it and the inbox row substitutes its preview on it. Two
    /// copies of this rule would drift, and the drift is silent — a message
    /// hidden inside the thread but printed in full in the conversation list.
    var isSensitive: Bool { severity == .critical }

    /// Apply a guardrail verdict to this bubble (label + severity together).
    ///
    /// `severityLevel` stays nil when the verdict has no severity, which is every
    /// model verdict since v16. Storing `.safe`'s raw value instead would assert
    /// the content had been assessed and found harmless — the opposite of "nobody
    /// assessed it" — and would render a "Safe" chip on a flagged message.
    func apply(_ result: SafetyResult) {
        safetyLabel = result.label.rawValue
        severityLevel = result.severity?.rawValue
    }

    // Cached language detection so the Translate control isn't re-detected on
    // every SwiftUI redraw (and token-by-token while a reply streams). Invalidated
    // in `text`'s didSet. Derived state — not `@Published` and not persisted.
    // Read on the main thread only (the UI); background features call
    // `Prompts.vietnameseRatio` directly, so there's no cross-thread access.
    private var _viRatio: Double?
    private var _viRatioValid = false
    private var _language: TranslateLanguage?
    private var _languageValid = false

    /// Vietnamese share of this bubble's words (0…1), or nil when undetermined —
    /// see `Prompts.vietnameseRatio`. Cached; recomputed only after `text` changes.
    var vietnameseRatio: Double? {
        if !_viRatioValid {
            _viRatio = Prompts.vietnameseRatio(text)
            _viRatioValid = true
        }
        return _viRatio
    }

    /// The language this bubble is written in, or nil when it can't be told —
    /// see `Prompts.sourceLanguage`. Cached for the same reason as the ratio, and
    /// more urgently: this one runs `NLLanguageRecognizer`, which is far too
    /// expensive to repeat per SwiftUI redraw and per streamed token.
    var detectedLanguage: TranslateLanguage? {
        if !_languageValid {
            _language = Prompts.sourceLanguage(of: text)
            _languageValid = true
        }
        return _language
    }

    init(sender: Sender,
         text: String,
         date: Date = Date(),
         isRead: Bool = true,
         authorName: String? = nil,
         imageID: String? = nil,
         imageTags: [String]? = nil,
         longSummary: String? = nil,
         fileID: String? = nil,
         fileName: String? = nil,
         fileSize: Int? = nil) {
        self.sender = sender
        self.text = text
        self.date = date
        self.authorName = authorName
        self.imageID = imageID
        self.imageTags = imageTags
        self.longSummary = longSummary
        self.fileID = fileID
        self.fileName = fileName
        self.fileSize = fileSize
        // A friend message is unread until acknowledged; your own is always read.
        self.isRead = sender == .you ? true : isRead
    }

    // MARK: - Persistence

    /// Flat, Codable snapshot of this bubble for on-disk storage. Transient UI
    /// flags (`showTranslation`, `isTranslating`) are intentionally dropped — only
    /// the durable content and read/translation state is saved.
    var snapshot: StoredMessage {
        StoredMessage(sender: sender, date: date, text: text,
                      isRead: isRead, translation: translation,
                      safetyLabel: safetyLabel, severityLevel: severityLevel,
                      authorName: authorName,
                      imageID: imageID, imageTags: imageTags,
                      longSummary: longSummary,
                      fileID: fileID, fileName: fileName, fileSize: fileSize,
                      reported: reported)
    }

    /// Rebuild a bubble from its stored snapshot.
    convenience init(_ stored: StoredMessage) {
        self.init(sender: stored.sender, text: stored.text,
                  date: stored.date, isRead: stored.isRead,
                  authorName: stored.authorName,
                  imageID: stored.imageID, imageTags: stored.imageTags,
                  longSummary: stored.longSummary,
                  fileID: stored.fileID, fileName: stored.fileName, fileSize: stored.fileSize)
        self.translation = stored.translation
        self.reported = stored.reported ?? false
        self.safetyLabel = stored.safetyLabel
        self.severityLevel = stored.severityLevel
    }
}

/// Codable value type mirroring a `ChatMessage`. Kept separate from the reference
/// type so persistence never wrestles with `@Published`/Combine.
struct StoredMessage: Codable {
    var sender: Sender
    var date: Date
    var text: String
    var isRead: Bool
    var translation: String?
    /// Raw `SafetyLabel` value; `nil` if not yet classified. Optional-decoded so
    /// older saved files without this key still load.
    var safetyLabel: Int?
    /// Raw `Severity` value; `nil` if not yet classified. Optional-decoded so
    /// older saved files without this key still load.
    var severityLevel: Int?
    /// Group-chat author name; `nil` for 1:1 messages. Optional-decoded for
    /// backward compatibility with older saves.
    var authorName: String?
    /// `ImageStore` id of an attached image; `nil` for text-only messages.
    /// Optional-decoded so older saves without this key still load.
    var imageID: String?
    /// Cached image tags (descending score); `nil` until the image is tagged.
    /// Optional-decoded for backward compatibility.
    var imageTags: [String]?
    /// Cached per-message long summary; `nil` until the user summarizes it.
    /// Optional-decoded for backward compatibility.
    var longSummary: String?
    /// `FileStore` id of an attached document; `nil` for non-file messages.
    var fileID: String?
    /// Original display name of the attached document.
    var fileName: String?
    /// Size of the attached document in bytes.
    var fileSize: Int?
    /// Whether the recipient reported this message. Optional-decoded so saves
    /// written before reporting existed still load (nil reads as "not reported").
    var reported: Bool?
}
