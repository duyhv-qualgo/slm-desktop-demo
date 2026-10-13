import Foundation

/// Periodic memory-footprint logging, for attributing a growth to what caused it.
///
/// Written after a run reached >3 GB, which no single measured consumer explains:
/// the whole guardrail is ~10 MB (both language packs 4.7, rule sets 0.8–1.8,
/// screening 4 000 messages 3.8, 20 000 verdicts 0.00), and the SLM context is
/// fixed at `n_ctx` 3072 / `n_ubatch` 128. A number that large is either many
/// model loads that were never released, or something growing per-frame or
/// per-message that nothing measured.
///
/// Rather than guess, this samples `phys_footprint` — the same accounting iOS
/// uses for its per-app limit, so it is the number that predicts a jetsam kill —
/// and prints a line whenever it moves by more than `stepMB`. Marks let you name
/// the moment something happened, so the log reads as a story:
///
///     [MEM] 152.3 MB  (+0.0)  · launch
///     [MEM] 448.9 MB  (+296.6) · after: load task model
///     [MEM] 461.2 MB  (+12.3) · after: guard warm-up
///     [MEM] 1876.4 MB (+1415.2) · after: benchmark run 3
///                                ^ there it is
///
/// Reporting only on movement keeps a quiet app quiet: an idle minute produces
/// no output at all.
enum MemoryTracker {

    /// Log when the footprint moves by at least this much since the last line.
    /// Small enough to catch a steady climb, large enough that allocator noise
    /// doesn't fill the log.
    static var stepMB: Double = 25

    /// How often to sample. Cheap — one `task_info` call.
    static var interval: TimeInterval = 1.0

    private static var timer: DispatchSourceTimer?
    private static var lastReported: Double = 0
    private static var baseline: Double = 0
    private static let queue = DispatchQueue(label: "com.example.slmchat.memtrack",
                                             qos: .utility)

    static var currentMB: Double {
        Double(ProcessMetrics.memoryFootprint() ?? 0) / 1_048_576
    }

    /// Begin sampling. Safe to call twice; the second call is ignored.
    ///
    /// Not started automatically — this is a diagnostic, and a timer that runs in
    /// every build is the kind of thing that quietly becomes load-bearing. Call it
    /// from `SLMChatApp` while chasing something, or from a test.
    static func start(label: String = "start") {
        queue.sync {
            guard timer == nil else { return }
            baseline = currentMB
            lastReported = baseline
            NSLog("[MEM] %.1f MB  (baseline) · %@", baseline, label)

            let t = DispatchSource.makeTimerSource(queue: queue)
            t.schedule(deadline: .now() + interval, repeating: interval)
            t.setEventHandler {
                let now = currentMB
                guard abs(now - lastReported) >= stepMB else { return }
                let delta = now - lastReported
                lastReported = now
                NSLog("[MEM] %.1f MB  (%+.1f, %+.1f since baseline)",
                      now, delta, now - baseline)
            }
            t.resume()
            timer = t
        }
    }

    static func stop() {
        queue.sync {
            timer?.cancel()
            timer = nil
        }
    }

    /// Name a moment. Always prints, whether or not the footprint moved — the
    /// point of a mark is to bracket an operation so the NEXT delta is attributable.
    static func mark(_ label: String) {
        let now = currentMB
        queue.async {
            let delta = now - lastReported
            lastReported = now
            NSLog("[MEM] %.1f MB  (%+.1f) · %@", now, delta, label)
        }
    }

    /// Bracket one operation and report what it cost.
    ///
    /// Note the footprint is sampled immediately after `body` returns, so deferred
    /// work (a background load, an autorelease pool that hasn't drained) lands on
    /// a later sample rather than here. That is usually what you want: it
    /// distinguishes "this call allocated" from "this call started something that
    /// allocated".
    @discardableResult
    static func measuring<T>(_ label: String, _ body: () async throws -> T)
        async rethrows -> T {
        let before = currentMB
        let result = try await body()
        let after = currentMB
        NSLog("[MEM] %.1f MB  (%+.1f) · %@", after, after - before, label)
        queue.async { lastReported = after }
        return result
    }
}
