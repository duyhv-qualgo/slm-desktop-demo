import Foundation

/// What the on-device features need from one chat message: the SLM prompts
/// (`Prompts`), guardrail scheduling (`MessageScreener`) and reporting
/// (`GuardReport`).
///
/// Those take this protocol rather than the demo's `ChatMessage`, so a host
/// messaging app conforms its own message type and brings none of the demo's
/// model layer with it.
protocol SLMMessage {
    var id: UUID { get }
    /// The message body. Empty for a photo or file sent without a caption.
    var text: String { get }
    /// Sent by the device owner. Rendered as `Prompts.userName` in prompts.
    var isFromUser: Bool { get }
    /// Who sent it in a group chat. nil in a 1:1 (the contact is implied) and
    /// for the owner's own messages.
    var authorName: String? { get }
    /// The guardrail's verdict, or nil until the message has been screened.
    var safety: SafetyLabel? { get }
    var severity: Severity? { get }
}
