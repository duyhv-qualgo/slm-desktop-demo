import Foundation

/// A turn in the conversation history handed to the model, as the C bridge
/// expects it: `["role": "user"|"model", "text": "…"]`.
typealias Turn = [String: String]

/// Identifies which app feature a `SLMEngine.complete`/`stream` call serves.
/// Tags the `[SLMPerf]` log line (via `description`, unchanged from the raw
/// strings the codebase used before — existing log filters keep matching).
///
/// This exists instead of a bare `String` so the valid label set is
/// compiler-checked. Add a case here — not a new string literal at the call
/// site — when a new feature needs one.
enum SLMFeatureLabel: Equatable, CustomStringConvertible {
    /// Default for `-complete`, e.g. a one-shot call not naming a feature.
    case complete
    /// Default for `-stream`, e.g. a one-shot call not naming a feature.
    case stream
    /// Auto-reply suggestion chips (1:1 or group).
    case suggest
    /// Whole-conversation or unread-window summary.
    case summarize
    /// One chunk of a long-message map-reduce summary. `phase` is a free-form
    /// log tag ("MAP", "REDUCE#1", …) — see `ChatService.summarizeChunks`.
    case summarizeChunk(phase: String)
    /// Message translation.
    case translate
    /// Translating a search query for on-device image-tag lookup.
    case translateQuery
    /// Calendar/reminder event extraction.
    case event
    /// Magic Compose: rewriting the user's draft, or generating a reply from a
    /// compose chip.
    case compose

    var description: String {
        switch self {
        case .complete:                    return "complete"
        case .stream:                      return "stream"
        case .suggest:                     return "suggest"
        case .summarize:                   return "summarize"
        case .summarizeChunk(let phase):   return "summarize-\(phase.lowercased())"
        case .translate:                   return "translate"
        case .translateQuery:              return "translate-query"
        case .event:                       return "event"
        case .compose:                     return "compose"
        }
    }
}

/// User-selectable compute backend, chosen in Settings. `.auto` is the shipping
/// default; the explicit cases exist so a device can be measured both ways
/// without a rebuild — the CPU/GPU prefill gap is the whole latency question for
/// auto-suggest.
///
/// Note `.gpu` is a REQUEST, not a guarantee: `ChatLM` forces CPU on the
/// simulator (no Metal compute device) and when llama.cpp reports no gpu-offload
/// support. The `[SLMPerf]` log lines report the backend actually in use.
enum SLMBackendOption: String, CaseIterable, Identifiable {
    case auto, gpu, cpu

    var id: String { rawValue }

    var label: String {
        switch self {
        case .auto: return "Auto"
        case .gpu:  return "GPU"
        case .cpu:  return "CPU"
        }
    }
}

/// Which llama.cpp backend the app's engines load on.
///
/// Metal is the default because PREFILL — not generation — dominates a turn: a
/// ~480-token auto-suggest prompt costs seconds on CPU while the ~50-token
/// answer costs far less, and prefill is a batched matmul, the workload a GPU is
/// best at. (Decode is memory-bound and gains little either way.)
///
/// The catch: ggml's Metal backend calls `abort()` on init failure instead of
/// returning an error, so a device where Metal init fails would crash on every
/// launch, with the SLM features dead and no way to recover from inside the app.
/// `preferred()` guards that with a probe flag: it's set just before a GPU load
/// and cleared once the load returns. Finding it still set at the next launch
/// means the previous attempt never returned — the process died inside Metal
/// init — so the app falls back to CPU permanently instead of crash-looping.
enum SLMBackend {
    static let gpu = "gpu"
    static let cpu = "cpu"

    /// Set while a GPU load is in flight; survives the abort that would clear it.
    private static let probeKey = "slm.gpuLoadInFlight"

    /// True when a previous GPU load aborted, so the app is pinned to CPU. Drives
    /// the Settings sheet's explanatory footer.
    static var gpuProbeTripped: Bool {
        UserDefaults.standard.bool(forKey: probeKey)
    }

    /// Backend for the app's own engines, honouring the user's Settings choice.
    ///
    /// `.auto` means GPU unless the crash guard tripped. An EXPLICIT `.gpu`
    /// overrides the guard — the guard exists to stop an unattended crash loop,
    /// not to overrule someone deliberately retrying it from Settings.
    static func preferred(_ choice: SLMBackendOption = .auto) -> String {
        switch choice {
        case .cpu: return cpu
        case .gpu: return gpu
        case .auto:
            if gpuProbeTripped {
                NSLog("[SLMPerf] previous GPU load never returned (Metal init aborted) → using CPU")
                return cpu
            }
            return gpu
        }
    }

    /// Clear the tripped guard so GPU is attempted again — for a Settings-level
    /// "retry GPU" affordance, or after shipping a fix.
    static func resetGPUProbe() {
        UserDefaults.standard.removeObject(forKey: probeKey)
    }

    /// Run `load` with the probe flag raised when it targets the GPU, so a Metal
    /// abort inside it is still visible on the next launch.
    static func guardingLoad<T>(backend: String, _ load: () -> T) -> T {
        guard backend == gpu else { return load() }
        let defaults = UserDefaults.standard
        defaults.set(true, forKey: probeKey)
        // Force the write to disk NOW: an abort inside load() gives no chance to
        // flush later, and an unflushed flag guards nothing.
        defaults.synchronize()
        let result = load()
        defaults.removeObject(forKey: probeKey)
        return result
    }
}

/// Serialized async wrapper around one `ChatLM` (llama.cpp / GGUF model).
/// Every generation runs on one private serial queue, so requests to this engine
/// queue and run one at a time.
///
/// The app runs ONE of these — the task finetune behind translate / suggest /
/// summarize / event / guardrail. There was a second for the chat partner's own
/// replies; those come from `ChatAPI` now, and that engine is gone.
///
/// `ChatLM` renders a real system prompt + role-tagged history + the latest user
/// turn through the GGUF model's embedded chat template, so the model sees proper
/// multi-turn structure. Single-turn tasks pass `system: nil, history: []`.
final class SLMEngine {

    /// Emit a `[SLMPerf]` line per generation (prefill/decode split, token counts,
    /// tok/s, latency). On by default — the lines are one per model call, and the
    /// auto-suggest latency budget is measured from them. Set to false to silence.
    static var logPerf = true

    private let namespace: String
    private let queue: DispatchQueue
    private var bridge: ChatLM?
    /// Bundle basename of the currently loaded checkpoint, for the perf log.
    private(set) var loadedModelName: String = "-"

    /// Backend the loaded model is actually running on (`"gpu"` / `"cpu"`), or nil
    /// if nothing is loaded. Not necessarily the backend that was requested.
    var activeBackend: String? { bridge?.activeBackend }

    /// Path of the LoRA adapter actually applied, or nil when running on bare
    /// weights. An adapter that failed to load leaves this nil while the engine
    /// still works, so read it rather than assuming `warmUp(adapterPath:)` took.
    var activeLoRAPath: String? { bridge?.activeLoRAPath }

    init(namespace: String) {
        self.namespace = namespace
        self.queue = DispatchQueue(label: "com.example.slmchat.engine.\(namespace)",
                                   qos: .userInitiated)
    }

    /// Load the model once. `backend` defaults to `SLMBackend.preferred()` — Metal
    /// unless a previous GPU load aborted (see `SLMBackend`). `ChatLM` still
    /// forces CPU on the simulator, which has no Metal compute device, so a
    /// `"gpu"` request there is silently served on CPU.
    ///
    /// `adapterPath` is the engine's DEFAULT LoRA language pack: every call uses
    /// it unless it names another (`complete(adapterPath:)`). See `LoRACatalog`
    /// and the `loraPath:` note in ChatLM.h.
    func warmUp(modelName: String,
                backend: String = SLMBackend.preferred(),
                adapterPath: String? = nil) async -> (ok: Bool, error: String?) {
        guard let modelPath = ResourceLookup.path(modelName, ext: "gguf") else {
            return (false, "\(modelName).gguf not in bundle — drop it into Resources/.")
        }
        return await withCheckedContinuation { cont in
            queue.async { [weak self] in
                let b = ChatLM()
                let t0 = Date()
                let ok = SLMBackend.guardingLoad(backend: backend) {
                    b.load(modelPath: modelPath,
                           backend: backend,
                           loraPath: adapterPath)
                }
                if Self.logPerf {
                    // Report the adapter that was ASKED FOR alongside the one that
                    // took: a pack that fails to parse degrades to bare weights
                    // silently, and that difference is invisible in the output.
                    let lora: String
                    switch (adapterPath, b.activeLoRAPath) {
                    case (nil, _):            lora = "none"
                    case (_?, let got?):      lora = (got as NSString).lastPathComponent
                    case (let want?, nil):    lora = "FAILED(\((want as NSString).lastPathComponent))"
                    }
                    NSLog("[SLMPerf] load %@/%@ backend=%@ (asked %@) lora=%@ → %@ in %.3fs",
                          self?.namespace ?? "?", modelName, b.activeBackend, backend,
                          lora, ok ? "ok" : "FAILED", -t0.timeIntervalSinceNow)
                }
                if ok {
                    self?.bridge = b
                    self?.loadedModelName = modelName
                    cont.resume(returning: (true, nil))
                } else {
                    cont.resume(returning: (false, b.lastError))
                }
            }
        }
    }

    /// Release the loaded model, freeing its memory mapping. `warmUp` keeps the
    /// current model alive until a new one loads (so a failed switch can't kill
    /// the features), which means both are mapped at once during a load —
    /// `ChatService.releaseSLMs` calls this first to avoid that. Runs on the
    /// engine's queue, so in-flight work finishes.
    func unload() async {
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            queue.async { [weak self] in
                self?.bridge = nil
                cont.resume()
            }
        }
    }

    /// Token count of `text` under the loaded model's vocabulary, or nil when no
    /// model is loaded.
    ///
    /// Deliberately synchronous and off the engine queue. Tokenizing only READS the
    /// vocabulary — it touches neither the context nor the KV cache — so it is safe
    /// to call concurrently with a generation, and it has to be: chunking a long
    /// message means hundreds of these calls, and hopping the serial queue for each
    /// would either deadlock (when called from a callback) or serialise behind an
    /// in-flight generation.
    func tokenCount(of text: String) -> Int? {
        guard let bridge else { return nil }
        return Int(bridge.tokenCount(of: text))
    }

    /// Load a LoRA pack into the engine ahead of the call that needs it, so the
    /// one-off `.gguf` parse doesn't land inside a latency-budgeted generation.
    ///
    /// Fire-and-forget: it runs on the engine's serial queue, so it can't overlap
    /// a generation, and if a generation gets there first that call simply pays
    /// the cost itself — the result is identical either way, only the timing
    /// differs. Nothing depends on this having completed.
    func prewarmAdapter(path: String?) {
        guard let path else { return }
        queue.async { [weak self] in self?.bridge?.prewarmAdapter(path: path) }
    }

    /// Stop the generation running right now — a Stop button, a closed chat, or a
    /// newer request making this one useless. See `ChatLM.cancel`.
    ///
    /// Deliberately NOT on the engine queue: that queue is blocked inside the very
    /// generation being cancelled, so a queued cancel would only run after it
    /// finished. The cancelled call still returns normally (its text so far,
    /// `stop=cancelled` in `[SLMPerf]`); calls queued behind it are not affected.
    func cancelCurrentGeneration() {
        bridge?.cancel()
    }

    /// Blocking generation, serialized and off-main. Returns the trimmed reply.
    ///
    /// `label` names the feature this call serves and only tags the
    /// `[SLMPerf]` line (see `SLMFeatureLabel`); it has no other effect on
    /// generation.
    func complete(system: String?,
                  history: [Turn],
                  user: String,
                  maxTokens: Int32,
                  temperature: Float,
                  label: SLMFeatureLabel = .complete,
                  adapterPath: String? = nil) async -> String? {
        await withCheckedContinuation { cont in
            queue.async { [weak self] in
                guard let self, let bridge = self.bridge else {
                    cont.resume(returning: nil); return
                }
                let queued = Date()
                let out = bridge.generate(system: system,
                                          history: history,
                                          user: user,
                                          maxTokens: maxTokens,
                                          temperature: temperature,
                                          adapterPath: adapterPath)
                self.logPerf(label: label.description, bridge: bridge,
                             callSeconds: -queued.timeIntervalSinceNow,
                             maxTokens: maxTokens, temperature: temperature)
                if out == nil { NSLog("[SLMEngine] generate → nil, lastError=%@", bridge.lastError) }
                cont.resume(returning: out?.trimmingCharacters(in: .whitespacesAndNewlines))
            }
        }
    }

    /// Streamed generation for the chatbot's typing effect. `onChunk` fires per
    /// token on the engine's thread (NOT main) — marshal to main before touching
    /// UI. Returns only when generation completes.
    func stream(system: String?,
                history: [Turn],
                user: String,
                maxTokens: Int32,
                temperature: Float,
                label: SLMFeatureLabel = .stream,
                onChunk: @escaping (_ chunk: String, _ done: Bool) -> Void) async {
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            queue.async { [weak self] in
                guard let self, let bridge = self.bridge else {
                    onChunk("", true); cont.resume(); return
                }
                let queued = Date()
                bridge.stream(system: system,
                              history: history,
                              user: user,
                              maxTokens: maxTokens,
                              temperature: temperature,
                              onChunk: onChunk)
                self.logPerf(label: label.description, bridge: bridge,
                             callSeconds: -queued.timeIntervalSinceNow,
                             maxTokens: maxTokens, temperature: temperature)
                cont.resume()
            }
        }
    }

    // MARK: - Perf logging

    /// Emit one `[SLMPerf]` line for the generation that just finished on
    /// `bridge`, e.g.
    ///
    /// ```
    /// [SLMPerf] task/suggest qwen3(gpu) | ttft 0.951s | decode 33.5 tok/s |
    /// TOTAL 2.627s | stop=eos | cap=160 temp=0.80
    /// ```
    ///
    /// `callSeconds` is measured in Swift and includes the bridge call itself, so
    /// it should track `TOTAL` closely — a gap means time went to the `onChunk`
    /// consumer. Queue wait (another feature generating first) is NOT in either
    /// number; measure that at the feature level with `SLMTurnTimer`.
    private func logPerf(label: String, bridge: ChatLM, callSeconds: Double,
                         maxTokens: Int32, temperature: Float) {
        guard Self.logPerf else { return }
        guard let m = bridge.lastTurnMetrics else {
            NSLog("[SLMPerf] %@/%@ %@ | no metrics (call %.3fs) err=%@",
                  namespace, label, loadedModelName, callSeconds, bridge.lastError)
            return
        }
        NSLog("[SLMPerf] %@/%@ %@(%@) | ttft %.3fs | decode %.1f tok/s | TOTAL %.3fs | "
              + "stop=%@ | cap=%d temp=%.2f%@",
              namespace, label, loadedModelName, bridge.activeBackend,
              m.timeToFirstTokenSeconds, m.decodeTokensPerSecond, m.totalSeconds,
              m.stopReason, maxTokens, temperature,
              callSeconds - m.totalSeconds > 0.05
                  ? String(format: " | callback +%.3fs", callSeconds - m.totalSeconds) : "")
    }
}

/// Stopwatch for one user-visible turn — the number a latency budget is written
/// against (e.g. "auto-suggest must land under 3s"). A turn can span several
/// model calls (retries, map-reduce chunks) plus parsing and any queue wait
/// behind another feature, none of which a single `[SLMPerf]` generation line
/// shows.
///
/// ```swift
/// let turn = SLMTurnTimer("auto-suggest")
/// …
/// turn.finish("attempts=\(n) chips=\(suggestions.count)")
/// ```
struct SLMTurnTimer {
    private let label: String
    private let start = Date()
    /// Extra latency the user already waited before the turn began (e.g. the
    /// auto-suggest debounce), reported alongside so the end-to-end wait is
    /// visible next to the model's share of it.
    private let preDelay: Double

    init(_ label: String, preDelay: Double = 0) {
        self.label = label
        self.preDelay = preDelay
    }

    var elapsed: Double { -start.timeIntervalSinceNow }

    /// Log the turn's total. `budget` flags the line with OVER/ok so a regression
    /// is visible without doing arithmetic in the console.
    func finish(_ detail: String = "", budget: Double = 3.0) {
        guard SLMEngine.logPerf else { return }
        let t = elapsed
        let e2e = t + preDelay
        NSLog("[SLMPerf] TURN %@ | model %.3fs%@ | e2e %.3fs | %@ (budget %.1fs) %@",
              label, t,
              preDelay > 0 ? String(format: " + wait %.3fs", preDelay) : "",
              e2e, e2e <= budget ? "ok" : "OVER", budget, detail)
    }
}
