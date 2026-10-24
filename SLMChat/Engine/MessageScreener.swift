import Foundation

/// `MessageScreener`'s tuning and priorities. Outside the generic class so
/// callers can name them without a message type.
enum MessageScreening {

    /// How far either side of the visible row to screen ahead.
    ///
    /// BIDIRECTIONAL, which older-only was not. A chat that opens at the last-read
    /// position has the unread messages BELOW the anchor, so the reader scrolls
    /// DOWN into them — look-ahead that only ran older would race away from the
    /// direction of travel and every new row would arrive unscreened. Running both
    /// ways costs about twice the look-ahead work and is correct whether the
    /// thread opens at the bottom or mid-history.
    static let lookAhead = 10

    /// Most messages allowed to wait at once.
    ///
    /// Roughly two screens either side of the viewport. Past this the queue is
    /// holding work for rows far outside it, which is precisely the work that
    /// delays what the reader is actually looking at. Sized well above a single
    /// viewport's demand so ordinary scrolling never sheds — only a flick does.
    static let maxPending = 48

    enum Priority {
        /// A message that just arrived while the user is here. Runs at once,
        /// ahead of the queue, and persists immediately.
        case immediate
        /// Backlog reached by scrolling. Queued, ordered by distance from the
        /// viewport, and persisted with coalescing.
        case background
    }
}

/// Schedules guardrail screening for a thread's messages.
///
/// Extracted because the naive version — one detached `Task` per message, run in
/// whatever order they were created — has two defects that only appear under
/// load, and both were measured:
///
///  1. **Wrong order.** A 200-row flick queued 210 messages and finished 17 of
///     them. The rest drained FIFO, so the message on screen waited behind ~200
///     the reader had already passed and would never return to.
///  2. **No shedding.** Every row scrolled past stayed queued forever, even
///     though nobody would look at it again.
///
/// This replaces that with a single drain loop over a pending set, always taking
/// the message NEAREST THE VIEWPORT next, and dropping the farthest work when the
/// queue grows past `MessageScreening.maxPending`. Shedding is safe because it is not a decision
/// not to screen: a shed message is forgotten entirely, so scrolling back to it
/// re-requests it through `onAppear` like any other row.
///
/// Shared by `Conversation` and `GroupChat` — the scheduling is identical and is
/// too easy to get subtly wrong twice.
@MainActor
final class MessageScreener<Message: SLMMessage> {

    typealias Priority = MessageScreening.Priority

    /// Messages waiting, keyed by id so a re-request can't double-enter.
    private var pending: [UUID: (message: Message, index: Int)] = [:]

    /// Ids already queued, running, or done — the dedupe gate. A shed message is
    /// REMOVED from this, which is what lets `onAppear` ask for it again.
    private var claimed: Set<UUID> = []

    /// Latest row the viewport reported. Priority is distance from here.
    private var anchor = 0

    private var draining = false

    /// Classifications actually run. Test seam: duplicate work is invisible in
    /// message state, because the second pass computes the same verdict.
    private(set) var runs = 0

    /// Messages shed to keep the queue bounded. Not a loss of coverage — they are
    /// re-requested if the reader returns — but a spike here means the reader is
    /// outrunning the screener, which is the number a flood control would watch.
    private(set) var shed = 0

    var pendingCount: Int { pending.count }

    private let screen: (Message, Priority) async -> Void

    init(screen: @escaping (Message, Priority) async -> Void) {
        self.screen = screen
    }

    /// Screen `message` now, ahead of anything queued.
    func screenNow(_ message: Message) async {
        guard needsScreening(message), !claimed.contains(message.id) else { return }
        claimed.insert(message.id)
        pending.removeValue(forKey: message.id)   // it may have been queued already
        runs += 1
        await screen(message, .immediate)
    }

    /// A row scrolled into view: queue it and its neighbours, nearest first.
    ///
    /// Cheap to call from `onAppear` on every row — anything already claimed
    /// returns immediately, which is the common case once look-ahead is ahead of
    /// the scroll.
    func viewportDidShow(index: Int, in messages: [Message]) {
        guard messages.indices.contains(index) else { return }
        anchor = index

        enqueue(messages[index], at: index)
        // Outward from the anchor, so the nearest neighbours are queued first
        // even before priority ordering gets a say.
        for d in 1...MessageScreening.lookAhead {
            if messages.indices.contains(index - d) { enqueue(messages[index - d], at: index - d) }
            if messages.indices.contains(index + d) { enqueue(messages[index + d], at: index + d) }
        }
        shedIfNeeded()
        drain()
    }

    /// Forget everything queued — the thread changed underneath us.
    func reset() {
        pending.removeAll()
        claimed.removeAll()
    }

    // MARK: - Internals

    private func needsScreening(_ message: Message) -> Bool {
        message.safety == nil && !message.text.isEmpty
    }

    private func enqueue(_ message: Message, at index: Int) {
        guard needsScreening(message), !claimed.contains(message.id) else { return }
        claimed.insert(message.id)
        pending[message.id] = (message, index)
    }

    /// Drop the work furthest from the viewport until the queue fits.
    private func shedIfNeeded() {
        guard pending.count > MessageScreening.maxPending else { return }
        let ordered = pending.sorted { distance($0.value.index) > distance($1.value.index) }
        for (id, _) in ordered.prefix(pending.count - MessageScreening.maxPending) {
            pending.removeValue(forKey: id)
            claimed.remove(id)          // so a return visit can ask again
            shed += 1
        }
    }

    private func distance(_ index: Int) -> Int { abs(index - anchor) }

    /// One drain loop, not one task per message. Serial anyway (the engine has a
    /// serial queue underneath), and this is what makes re-ordering possible: the
    /// next message is chosen when the previous one finishes, so a scroll that
    /// happens mid-drain changes what runs next.
    private func drain() {
        guard !draining else { return }
        draining = true
        Task { [weak self] in
            while let next = self?.takeNearest() {
                self?.runs += 1
                await self?.screen(next, .background)
            }
            self?.draining = false
        }
    }

    private func takeNearest() -> Message? {
        guard let (id, entry) = pending.min(by: { distance($0.value.index) < distance($1.value.index) })
        else { return nil }
        pending.removeValue(forKey: id)
        // A message can be screened between queueing and running (the live path,
        // or a duplicate row) — skip it rather than paying for it twice.
        guard needsScreening(entry.message) else { return takeNearest() }
        return entry.message
    }
}
