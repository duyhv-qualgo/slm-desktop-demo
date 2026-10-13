import Foundation
#if canImport(UIKit)
import UIKit
#endif

/// Inference backend the LiteRT-LM engine is loaded on. The raw value is what
/// `ChatLM.load(modelPath:backend:cacheDir:)` expects.
enum BenchmarkBackend: String, CaseIterable, Identifiable {
    case cpu
    case gpu

    var id: String { rawValue }

    /// On Apple platforms LiteRT-LM's GPU delegate is Metal, so that's what the
    /// label says — "GPU" alone is ambiguous next to the Neural Engine.
    var displayName: String {
        switch self {
        case .cpu: return "CPU"
        case .gpu: return "GPU (Metal)"
        }
    }

    /// Metal isn't usable under the iOS simulator for LiteRT-LM's delegate, and
    /// not every model/device pair compiles on the GPU path either — so a GPU run
    /// is allowed to fail and is reported rather than aborting the sweep.
    var isLikelyAvailable: Bool {
        #if targetEnvironment(simulator)
        return self == .cpu
        #else
        return true
        #endif
    }
}

extension ChatLMActivationType {
    /// Short label for logs, the CSV and the results list. `auto` resolves to
    /// F32 on GPU and the model's default on CPU, so it's reported as-configured
    /// rather than guessing what the runtime picked.
    var label: String {
        switch self {
        case .F32:  return "f32"
        case .F16:  return "f16"
        case .I16:  return "i16"
        case .I8:   return "i8"
        default:    return "auto"
        }
    }
}

/// Which question a benchmark pass is answering. The two are deliberately kept
/// apart: they are not comparable to each other, only within themselves.
enum BenchmarkMode: String, CaseIterable, Identifiable {
    /// Fair speed comparison. Sampling is pinned as deterministically as the
    /// runtime allows and throughput is measured over a FIXED decode window, so
    /// the number means the same thing for every model and backend.
    case benchmarkFixed = "benchmark-fixed"
    /// Realistic UX. The app's real generation settings, EOS respected, so
    /// natural output length and end-to-end latency are what a user would see.
    case production = "production"

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .benchmarkFixed: return "Benchmark (fixed)"
        case .production:     return "Production"
        }
    }

    var explanation: String {
        switch self {
        case .benchmarkFixed:
            return "Deterministic settings, throughput measured over a fixed "
                 + "decode window. Comparable across models and backends."
        case .production:
            return "The app's real sampling with EOS respected. Shows natural "
                 + "output length and realistic latency, not a fair speed race."
        }
    }
}

/// Decoding settings for one benchmark pass, kept entirely separate from the
/// app's production generation settings in `ChatService`.
struct BenchmarkConfig {
    let mode: BenchmarkMode
    /// Output cap handed to the runtime, and — when `forcesDecodeLength` — the
    /// exact number of tokens that will be decoded.
    let maxTokens: Int32
    let temperature: Float
    /// Whether to pin the decode length via LiteRT-LM's
    /// `set_num_decode_tokens`, which suppresses early stopping on EOS.
    let forcesDecodeLength: Bool

    /// Load options for the engine this config runs on. Benchmark
    /// instrumentation is on for both modes — that's how exact token counts are
    /// obtained — but only the fixed mode pins the decode length.
    var engineOptions: SLMEngine.BenchmarkOptions {
        SLMEngine.BenchmarkOptions(forcedDecodeTokens: forcesDecodeLength ? maxTokens : Int32(0))
    }

    /// Fair-comparison pass. `forcedDecodeTokens` makes the runtime decode
    /// exactly `maxTokens` tokens and ignore EOS, so every model does identical
    /// work. 512 in + 512 forced out is why benchmark loads get a 2048-token
    /// single-sequence context (`kBenchmarkCtx` in ChatLM.mm) instead of the
    /// app's 1024-token slots — the two must both fit in one sequence.
    ///
    /// One caveat on determinism: temperature is 0.1, not 0. Temperature 0
    /// selects LiteRT-LM's greedy sampler, which fails in this build —
    /// generation returns immediately with no tokens, the same reason
    /// `ChatService.guardClassifyTemp` is pinned at 0.1. 0.1 is near-
    /// deterministic but is not true argmax decoding.
    static let fixed = BenchmarkConfig(mode: .benchmarkFixed,
                                       maxTokens: 512,
                                       temperature: 0.1,
                                       forcesDecodeLength: true)

    /// Realistic pass: mid-range sampling with EOS respected, so the run ends
    /// where a real generation would. The 512 cap is a ceiling, not a target —
    /// EOS should be what stops it, and `stopReason` says which did. 0.5 sits between the app's most deterministic
    /// task (event, 0.1) and its most varied (suggest, 0.8); it is a benchmark
    /// constant, not a mirror of any one feature.
    static let production = BenchmarkConfig(mode: .production,
                                            maxTokens: 512,
                                            temperature: 0.5,
                                            forcesDecodeLength: false)

    static func forMode(_ mode: BenchmarkMode) -> BenchmarkConfig {
        switch mode {
        case .benchmarkFixed: return .fixed
        case .production:     return .production
        }
    }
}

/// One inference run's latency measurement.
///
/// Timestamps come from `DispatchTime` (monotonic clock) so the numbers can't be
/// skewed by a wall-clock adjustment mid-run:
///
///  - `t0`      — captured immediately before calling into the engine
///  - `t1`      — first streamed token
///  - `tLast`   — last streamed token
struct BenchmarkRun {
    let modelName: String
    let device: String
    let backend: BenchmarkBackend
    let mode: BenchmarkMode
    /// 1-based index within this model+backend+mode series. Run 1 is the warm-up.
    let runIndex: Int
    let isWarmUp: Bool

    /// Time To First Token measured by the runtime (prefill + first decode).
    let ttftMS: Double
    /// Decode throughput reported by the runtime, in real tokens/sec.
    let decodeTPS: Double
    /// End-to-end wall-clock, `tLast - t0`, measured from Swift. This is the one
    /// figure that legitimately comes from the caller's side: it's what the app
    /// would actually wait, including bridge overhead the runtime doesn't count.
    let totalLatencyMS: Double
    /// EXACT number of tokens decoded, from the runtime. Not a chunk count.
    let totalTokens: Int
    /// Tokens requested. With `forcesDecodeLength`, `totalTokens` should equal it.
    let targetTokens: Int
    /// Streaming callbacks received. Kept because it is NOT the token count —
    /// the runtime batches several tokens per chunk — and the ratio is worth
    /// seeing rather than silently assuming.
    let chunkCount: Int
    /// Whether this run's length was pinned by the runtime.
    let lengthWasForced: Bool
    /// Activation precision the engine actually loaded with. Runs at different
    /// precisions are not directly comparable.
    let activation: ChatLMActivationType
    /// Process-wide CPU and memory cost of this run. Process-wide because that's
    /// what the OS can tell us — see `ProcessMetrics`.
    let usage: ResourceUsage
    /// Footprint measured with no benchmark model loaded, just before this
    /// series' model was mapped. Everything memory-related is reported against
    /// this, so the figure is the model's own cost rather than the app's total.
    let baselineFootprintBytes: UInt64
    /// Footprint immediately after the model loaded, before any generation.
    let loadedFootprintBytes: UInt64
    /// Thermal state either side of the run. Anything other than `nominal` means
    /// the CPU was being throttled, so the speed figures reflect thermal policy
    /// as much as the model — the run is kept but marked.
    let startThermal: ProcessInfo.ThermalState
    let endThermal: ProcessInfo.ThermalState

    var wasThrottled: Bool { startThermal != .nominal || endThermal != .nominal }

    /// RAM attributable to holding this model's weights.
    var modelLoadMB: Double {
        (Double(loadedFootprintBytes) - Double(baselineFootprintBytes)) / 1_048_576
    }
    /// Peak RAM attributable to this model, weights plus the KV cache and
    /// activations that generation adds on top.
    var modelPeakMB: Double {
        (Double(usage.peakFootprintBytes) - Double(baselineFootprintBytes)) / 1_048_576
    }

    var prefillTokens: Int
    var prefillTPS: Double

    /// Whether the run decoded all the way to the requested count.
    var reachedTarget: Bool { totalTokens >= targetTokens }

    /// In fixed mode every run must hit the target exactly; anything else means
    /// the forced-length setting didn't take, and the run can't be compared.
    var isAnomalous: Bool { lengthWasForced && !reachedTarget }

    /// The console line format the harness prints for every run.
    var consoleLine: String {
        String(format: "%@, %@, %@, %@/%@, %.1f, %.2f, %.1f, %d/%d (%d chunks), cpu %.0f%%, ram %.0fMB",
               mode.rawValue, modelName, device, backend.rawValue, activation.label,
               ttftMS, decodeTPS, totalLatencyMS,
               totalTokens, targetTokens, chunkCount,
               usage.cpuPercent, usage.peakFootprintMB)
    }
}

/// Mean and sample standard deviation of one metric across a series.
struct BenchmarkStat {
    let mean: Double
    let stdDev: Double

    /// `nil` mean for an empty series; stdDev is 0 for a single sample (no
    /// spread to measure), not undefined.
    init?(_ values: [Double]) {
        guard !values.isEmpty else { return nil }
        let n = Double(values.count)
        // Held locally as well as stored: referring to `mean` inside the reduce
        // below would capture a not-yet-initialized property.
        let average = values.reduce(0, +) / n
        mean = average
        guard values.count > 1 else { stdDev = 0; return }
        // Sample stddev (n-1): these runs are a sample of the device's behaviour,
        // not the whole population.
        let variance = values.reduce(0) { $0 + ($1 - average) * ($1 - average) } / (n - 1)
        stdDev = variance.squareRoot()
    }

    func formatted(_ unit: String, decimals: Int = 1) -> String {
        String(format: "%.\(decimals)f ± %.\(decimals)f %@", mean, stdDev, unit)
    }
}

/// Aggregate over one model × backend × mode series (warm-up and anomalies
/// excluded).
struct BenchmarkSummary: Identifiable {
    let modelName: String
    let device: String
    let backend: BenchmarkBackend
    let mode: BenchmarkMode
    /// Runs actually aggregated.
    let sampleCount: Int
    /// Runs excluded for not hitting the forced token count, on top of the
    /// warm-up. Should be 0 — see `BenchmarkRun.isAnomalous`.
    let anomalyCount: Int

    let ttft: BenchmarkStat
    let decodeTPS: BenchmarkStat
    let totalLatency: BenchmarkStat
    let cpuPercent: BenchmarkStat
    /// Total process footprint — includes whatever else the app has resident.
    let peakMemoryMB: BenchmarkStat
    /// Peak footprint minus the pre-load baseline: this model's own cost, and
    /// the only memory figure that's fair to compare across models.
    let modelPeakMB: BenchmarkStat
    /// RAM the weights alone occupy, before generation.
    let modelLoadMB: Double
    /// Whether the app's own SLMs were unloaded for this sweep. When false, the
    /// baseline includes them and a model the app also has loaded shares pages
    /// with it, understating its cost.
    let wasIsolated: Bool
    /// Runs that ran while the device was thermally throttled. Counted, not
    /// excluded — on a sustained sweep it's often every run, and dropping them
    /// would leave nothing. It does mean the speed figures are a floor.
    let throttledRunCount: Int
    let meanTokens: Double
    let targetTokens: Int
    /// Mean tokens per streamed chunk, so the batching factor is visible.
    let meanTokensPerChunk: Double

    var id: String { "\(mode.rawValue)-\(modelName)-\(backend.rawValue)" }
}

/// Collects per-token arrival timestamps. The engine's streaming callback fires
/// on a background thread while the awaiting caller sits on the main actor, so
/// the two sides are separated by a lock. Both accessors are synchronous by
/// design — `NSLock` is unavailable from async contexts.
private final class TokenTimeline {
    private let lock = NSLock()
    private var times: [UInt64] = []

    func mark(_ nanos: UInt64) {
        lock.lock()
        times.append(nanos)
        lock.unlock()
    }

    var timestamps: [UInt64] {
        lock.lock()
        defer { lock.unlock() }
        return times
    }
}

/// Measures generation latency of the LiteRT-LM SLMs.
///
/// Loads each model into its OWN `SLMEngine` (one per backend) rather than
/// borrowing `ChatService.taskEngine`, so running a benchmark never swaps the
/// model the app's features are using, and never touches the app's production
/// generation settings.
///
/// One caveat on token counting: the LiteRT-LM C API hands back decoded text
/// chunks, not token ids. This counts one token per non-empty chunk, which holds
/// for the current runtime (it emits one callback per decoded token) but is an
/// approximation, not a tokenizer-exact count.
@MainActor
final class BenchmarkHarness: ObservableObject {

    /// Fixed prompt for every run, model and mode: a ~512-token passage plus an
    /// open-ended instruction, so models decode as far as they can before EOS.
    ///
    /// The length is the point. The previous prompt was 31 tokens (40 after the
    /// chat template), which measured prefill badly in two ways. Throughput is
    /// still climbing at that length — qwen3-0.6b-v6 measured 227 tok/s at ~25
    /// tokens against 372 tok/s at ~990, because a short prompt divides the fixed
    /// per-call cost over very few tokens. And it was unstable: the same 23-token
    /// case measured 293, 834, 893 and 1327 tok/s across four consecutive runs on
    /// one machine, a 4.5x spread on identical input. Neither is a defect in the
    /// engine; both make a 40-token prompt the wrong instrument for comparing
    /// models or backends.
    ///
    /// Real prose, not repeated filler: repetition is unusually easy to predict
    /// and unusually cache-friendly, which flatters both phases.
    ///
    /// Changing this text invalidates comparison against previously recorded runs
    /// — token count is part of the measurement, so treat it as a versioned
    /// constant rather than editable copy.
    static let fixedPrompt = """
    Read the following passage, then write a detailed technical explanation that \
    expands on every stage it describes.

    Modern smartphone manufacturing begins long before assembly, with the \
    refinement of raw materials. Silicon is purified to semiconductor grade, \
    drawn into monocrystalline ingots, and sliced into wafers a fraction of a \
    millimetre thick. Those wafers pass through hundreds of photolithography, \
    etching, deposition and implantation steps, each one adding or removing \
    material with tolerances measured in nanometres. A single defective step \
    ruins the die, so wafers are inspected continuously and yields are tracked \
    per lot. Completed dies are tested at wafer level, cut apart, and packaged \
    into the chips that will become the application processor, the modem, the \
    memory and the power management circuitry.

    In parallel, the display is built up from a glass or polymer substrate, \
    coated with thin-film transistors, and bonded to a touch digitiser and a \
    protective cover lens. Camera modules pair image sensors with stacks of \
    moulded plastic lenses aligned to micron precision, then add voice-coil \
    motors for autofocus and optical stabilisation. Batteries are wound or \
    stacked from coated electrode foils, filled with electrolyte, sealed, and \
    then cycled repeatedly to verify capacity and internal resistance before \
    any of them are allowed near a phone.

    Final assembly brings these subassemblies together on a line that mixes \
    robotic placement with manual inspection. Surface-mount machines populate \
    the main board with thousands of components, reflow ovens solder them in a \
    controlled thermal profile, and automated optical inspection checks every \
    joint. The board is then married to the display, the battery, the cameras \
    and the enclosure, with adhesives and gaskets applied to meet ingress \
    protection ratings. Each unit is flashed with firmware, calibrated, and put \
    through functional tests covering radios, sensors, audio, charging and \
    thermals. Units that fail are diverted for rework or teardown analysis, and \
    the failure data feeds back into process control upstream.

    Beyond the line itself, production depends on a supply chain that must stay \
    synchronised across continents. Components are qualified with second sources \
    wherever possible, because a single fire, flood or export restriction can \
    idle an assembly plant within days. Finished units are packed with \
    accessories and documentation, palletised, and routed through customs to \
    regional distribution centres, where firmware may be updated again before \
    retail. Returns and warranty repairs flow back along the same path, and the \
    components recovered from them are graded, refurbished or recycled.

    Explain each of these stages in depth, including why the tolerances matter, \
    what typically goes wrong, and how manufacturers detect and correct it.
    """

    /// Runs per model per backend per mode, including the warm-up.
    static let runsPerSeries = 10

    /// Seconds to idle between series so the device can shed heat.
    ///
    /// Back-to-back series bias a sweep: sustained inference pushes the SoC out
    /// of its `nominal` thermal state, and everything measured after that point
    /// is throttled. A model benchmarked last then looks slower than one
    /// benchmarked first purely because of running order. Pausing lets each
    /// series start from a comparable thermal state — at the cost of a much
    /// longer sweep, so it's opt-in.
    static let cooldownOptions: [Int] = [0, 30, 60, 120]

    /// Seconds to idle BETWEEN the runs of a single series (not just between
    /// series). Ten generations back-to-back heat the SoC into throttling, which
    /// collapses every model toward the same throttled-clock TPS — so a per-run
    /// pause keeps each of the 10 rounds starting from a comparable thermal state.
    static let runCooldownOptions: [Int] = [0, 15, 30]

    /// Every completed run, in execution order.
    @Published private(set) var runs: [BenchmarkRun] = []
    /// One entry per model × backend × mode series that produced usable runs.
    @Published private(set) var summaries: [BenchmarkSummary] = []
    /// Non-nil while a benchmark is in flight.
    @Published private(set) var progress: String?
    /// Load failures, unmeasurable runs and short-of-target runs, accumulated
    /// rather than overwritten so nothing scrolls out of sight mid-sweep.
    @Published private(set) var issues: [String] = []
    /// Battery and thermal readings for the whole sweep. Nil until one finishes.
    /// Sweep-level rather than per-run — see `EnergyReading`.
    @Published private(set) var energy: EnergyReading?

    /// Optional idle between the runs WITHIN a series (0 = off). Set from the UI
    /// before a sweep; `runSeries` reads it, so it needs no plumbing through
    /// `run(...)`. See `runCooldownOptions` for why it helps.
    @Published var perRunCooldownSeconds: Int = 0

    private let energyMonitor = EnergyMonitor()
    /// Whether the current sweep unloaded the app's own models. Recorded on every
    /// summary so a CSV can't be read as isolated when it wasn't.
    private var isIsolated = false

    /// Where the CSV lands. Surfaced in the UI so it can be found in the Files
    /// app / pulled off the device with the container download.
    let csvURL: URL = FileManager.default
        .urls(for: .documentDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("slm_benchmark.csv")

    /// The engine for the series currently running. Exactly one model is held at
    /// a time: these checkpoints are 285–613 MB, and the app itself already has
    /// its own models resident, so caching an engine per backend+mode is enough
    /// to exhaust the address space on a 4 GB device.
    private var engine: SLMEngine?
    private let device = BenchmarkHarness.deviceIdentifier()

    // MARK: - Running

    /// Sweep every model × backend × mode: load, run `runsPerSeries` generations
    /// back-to-back, discard run 1 as warm-up, and aggregate the rest. Every run
    /// is logged and appended to the CSV as it completes, so a sweep interrupted
    /// partway still leaves its measurements on disk.
    ///
    /// A backend that won't load for a given model is recorded in `issues` and
    /// skipped, so one unsupported pairing doesn't cost the whole sweep.
    /// `isolating` is the app's own service. When supplied, its SLM engines are
    /// unloaded for the duration of the sweep and reloaded afterwards, which is
    /// what makes the RAM figures comparable across models — otherwise whichever
    /// checkpoint the app already has mapped shares pages with the benchmark
    /// copy and looks far cheaper than the rest.
    ///
    /// `cooldownSeconds` idles between series so each starts from a comparable
    /// thermal state — see `cooldownOptions`.
    func run(models: [SLMModelOption],
             backends: [BenchmarkBackend],
             modes: [BenchmarkMode],
             cooldownSeconds: Int = 0,
             isolating service: ChatService? = nil) async {
        issues = []
        energy = nil
        // The bridge can only report "engine_create returned null"; the actual
        // reason (unsupported op, out of memory, bad backend) is only in
        // LiteRT-LM's own log. INFO is enough to see it without drowning the
        // console in per-token spam.
        SLMEngine.setRuntimeLogLevel(2)
        defer { SLMEngine.setRuntimeLogLevel(3) }

        let available = models.filter(\.isAvailable)
        guard !available.isEmpty else {
            issues = ["No models bundled — add a .gguf to Resources/."]
            return
        }
        guard !backends.isEmpty, !modes.isEmpty else {
            issues = ["Select at least one backend and one mode."]
            return
        }

        // Unload the app's models BEFORE the first baseline is taken, so the
        // baseline reflects an app with no SLMs resident.
        isIsolated = service != nil
        if let service {
            progress = "Unloading the app's models…"
            await service.releaseSLMs()
        }

        energyMonitor.start()

        // Mode is the outermost loop so a full comparable pass finishes before the
        // production pass starts — an interrupted sweep then still has one
        // complete, self-consistent set of numbers.
        var isFirstSeries = true
        for mode in modes {
            let config = BenchmarkConfig.forMode(mode)
            for model in available {
                for backend in backends {
                    // Cool down BETWEEN series, not before the first — there's
                    // nothing to shed yet, and it would just delay the sweep.
                    if !isFirstSeries { await cooldown(seconds: cooldownSeconds) }
                    isFirstSeries = false
                    await runSeries(model: model, backend: backend, config: config)
                }
            }
        }
        // Don't sit on 600 MB after the sweep — the app still has its own models
        // to keep resident.
        if let engine { await engine.unload() }
        engine = nil

        // Restore the app's models before anything else — the sweep is over, and
        // leaving the app's features dead is far worse than a delayed summary.
        if let service {
            progress = "Reloading the app's models…"
            if await service.restoreSLMs() == false {
                issues.append("FAILED to reload the app's models after the sweep — "
                            + "translate/suggest/summarize will not work until the "
                            + "app is relaunched.")
            }
        }

        let reading = energyMonitor.stop()
        energy = reading
        if reading.wasCharging {
            issues.append("Device was plugged in during the sweep — battery drain "
                        + "is not measurable. Unplug and rerun for an energy figure.")
        }
        NSLog("[Bench] ENERGY drain %@ over %.0fs | thermal %@ → %@",
              reading.drainPercent.map { String(format: "%.1f%%", $0) } ?? "n/a",
              reading.elapsedSeconds,
              reading.startThermalState.label, reading.endThermalState.label)
        progress = nil
    }

    /// One model × backend × mode series.
    private func runSeries(model: SLMModelOption,
                           backend: BenchmarkBackend,
                           config: BenchmarkConfig) async {
        let series = "\(config.mode.displayName) · \(model.displayName) · \(backend.displayName)"
        progress = "\(series) · loading…"

        // Frees the previous series' model before mapping this one — see
        // `freshEngine`.
        let engine = await freshEngine(backend: backend, mode: config.mode)

        // Baseline with NO benchmark model mapped. Everything memory-related is
        // reported against this, so the numbers are the model's own cost rather
        // than the app's running total.
        let baseline = ProcessMetrics.memoryFootprint() ?? 0

        // GPU loads default to F32 activations (a Gemma requirement), which costs
        // roughly twice the activation memory. If the default fails on GPU, retry
        // at F16 rather than dropping the model. Which precision won is recorded
        // on every run — F16 and F32 are not a like-for-like comparison.
        var activation: ChatLMActivationType = .auto
        MemoryTracker.mark("benchmark: before load \(model.resourceName)")
        var load = await engine.warmUp(modelName: model.resourceName,
                                       backend: backend.rawValue,
                                       benchmark: config.engineOptions,
                                       activation: activation)
        if !load.ok, backend == .gpu {
            NSLog("[Bench] %@: F32 load failed (%@) — retrying at F16",
                  series, load.error ?? "unknown")
            await engine.unload()
            activation = .F16
            load = await engine.warmUp(modelName: model.resourceName,
                                       backend: backend.rawValue,
                                       benchmark: config.engineOptions,
                                       activation: activation)
            if load.ok {
                issues.append("\(series): would not load with F32 activations; "
                            + "measured at F16 instead — NOT directly comparable "
                            + "to F32 results.")
            }
        }
        guard load.ok else {
            // The bridge only sees a null engine, so the reason is in the
            // LiteRT-LM log the sweep just enabled. A "TF_LITE_PREFILL_DECODE
            // not found" there usually means the section failed to mmap for lack
            // of memory, not that the model is malformed.
            let note = "\(series): \(load.error ?? "load failed") — see the "
                     + "LiteRT-LM log above for the underlying reason."
            NSLog("[Bench] %@", note)
            issues.append(note)
            self.engine = nil
            return
        }

        let loaded = ProcessMetrics.memoryFootprint() ?? baseline

        var measured: [BenchmarkRun] = []
        var anomalies = 0

        for index in 1...Self.runsPerSeries {
            // Optional per-run cooldown so back-to-back generations don't heat the
            // SoC into a throttled floor (which collapses TPS across models). Skip
            // before the first run — nothing to shed yet.
            if index > 1 { await cooldown(seconds: perRunCooldownSeconds) }
            progress = "\(series) · run \(index)/\(Self.runsPerSeries)"
            let isWarmUp = index == 1
            guard let run = await measureOne(engine: engine,
                                             modelName: model.displayName,
                                             backend: backend,
                                             activation: activation,
                                             config: config,
                                             baseline: baseline,
                                             loaded: loaded,
                                             runIndex: index,
                                             isWarmUp: isWarmUp) else {
                issues.append("\(series): run \(index) produced no tokens.")
                continue
            }

            NSLog("[Bench] %@%@", run.consoleLine, isWarmUp ? "  (warm-up, discarded)" : "")
            appendCSV(run)
            runs.append(run)
            guard !isWarmUp else { continue }

            // With the decode length pinned, a run that misses the target means
            // the forced-length setting didn't take. That's a broken measurement,
            // not a slow one, so it's surfaced and excluded rather than averaged.
            if run.isAnomalous {
                anomalies += 1
                issues.append("\(series): run \(index) decoded \(run.totalTokens)/"
                            + "\(run.targetTokens) tokens despite the length being "
                            + "forced — EXCLUDED from the aggregate.")
                continue
            }
            measured.append(run)
        }

        guard let summary = Self.summarize(measured,
                                           modelName: model.displayName,
                                           device: device,
                                           backend: backend,
                                           config: config,
                                           anomalyCount: anomalies,
                                           isolated: isIsolated) else { return }

        NSLog("[Bench] AGG %@ | ttft %@ | decode %@ | total %@ | tokens %.1f/%d | n=%d",
              summary.id,
              summary.ttft.formatted("ms"),
              summary.decodeTPS.formatted("tok/s", decimals: 2),
              summary.totalLatency.formatted("ms"),
              summary.meanTokens, summary.targetTokens, summary.sampleCount)
        summaries.append(summary)
    }

    /// Idle so the device can shed heat, counting down in the progress line so
    /// the pause doesn't look like a hang.
    private func cooldown(seconds: Int) async {
        guard seconds > 0 else { return }
        for remaining in stride(from: seconds, to: 0, by: -1) {
            let thermal = ProcessInfo.processInfo.thermalState.label
            progress = "Cooling down — \(remaining)s (thermal: \(thermal))"
            try? await Task.sleep(nanoseconds: 1_000_000_000)
        }
    }

    /// Release whatever model is currently loaded and hand back a fresh engine.
    /// Unloading first is what makes a large model loadable at all: the previous
    /// model's mapping has to be gone before the next one is mapped.
    private func freshEngine(backend: BenchmarkBackend, mode: BenchmarkMode) async -> SLMEngine {
        if let engine { await engine.unload() }
        engine = nil
        let next = SLMEngine(namespace: "bench-\(backend.rawValue)-\(mode.rawValue)")
        engine = next
        return next
    }

    /// One measured generation. TTFT, token counts and decode throughput come
    /// from LiteRT-LM's own instrumentation rather than from callback timing —
    /// the streaming callback batches multiple tokens per chunk, so counting
    /// callbacks would understate the token count and the throughput. Total
    /// latency stays wall-clock, since that's what the caller actually waits.
    ///
    /// Returns nil if the runtime reported no metrics (engine not loaded in
    /// benchmark mode, or generation failed).
    private func measureOne(engine: SLMEngine,
                            modelName: String,
                            backend: BenchmarkBackend,
                            activation: ChatLMActivationType,
                            config: BenchmarkConfig,
                            baseline: UInt64,
                            loaded: UInt64,
                            runIndex: Int,
                            isWarmUp: Bool) async -> BenchmarkRun? {
        let timeline = TokenTimeline()
        let sampler = ResourceSampler()
        let startThermal = ProcessInfo.processInfo.thermalState

        sampler.start()
        let t0 = DispatchTime.now().uptimeNanoseconds
        let info = await engine.streamMeasured(system: nil,
                                               history: [],
                                               user: Self.fixedPrompt,
                                               maxTokens: config.maxTokens,
                                               temperature: config.temperature) { chunk, done in
            // The final callback carries an empty chunk purely to signal
            // completion — it isn't content, so it must not be counted.
            guard !done, !chunk.isEmpty else { return }
            timeline.mark(DispatchTime.now().uptimeNanoseconds)
        }
        let tEnd = DispatchTime.now().uptimeNanoseconds
        let usage = sampler.stop()

        guard let info else { return nil }

        return BenchmarkRun(modelName: modelName,
                            device: device,
                            backend: backend,
                            mode: config.mode,
                            runIndex: runIndex,
                            isWarmUp: isWarmUp,
                            ttftMS: info.timeToFirstTokenSeconds * 1000,
                            decodeTPS: info.decodeTokensPerSecond,
                            totalLatencyMS: Double(tEnd - t0) / 1_000_000,
                            totalTokens: Int(info.decodeTokenCount),
                            targetTokens: Int(config.maxTokens),
                            chunkCount: timeline.timestamps.count,
                            lengthWasForced: config.forcesDecodeLength,
                            activation: activation,
                            usage: usage,
                            baselineFootprintBytes: baseline,
                            loadedFootprintBytes: loaded,
                            startThermal: startThermal,
                            endThermal: ProcessInfo.processInfo.thermalState,
                            prefillTokens: Int(info.prefillTokenCount),
                            prefillTPS: info.prefillTokensPerSecond)
    }

    private static func summarize(_ runs: [BenchmarkRun],
                                  modelName: String,
                                  device: String,
                                  backend: BenchmarkBackend,
                                  config: BenchmarkConfig,
                                  anomalyCount: Int,
                                  isolated: Bool) -> BenchmarkSummary? {
        guard let ttft = BenchmarkStat(runs.map(\.ttftMS)),
              let decode = BenchmarkStat(runs.map(\.decodeTPS)),
              let total = BenchmarkStat(runs.map(\.totalLatencyMS)),
              let cpu = BenchmarkStat(runs.map(\.usage.cpuPercent)),
              let memory = BenchmarkStat(runs.map(\.usage.peakFootprintMB)),
              let modelPeak = BenchmarkStat(runs.map(\.modelPeakMB)) else { return nil }

        let n = Double(runs.count)
        let chunks = runs.reduce(0) { $0 + $1.chunkCount }
        let tokens = runs.reduce(0) { $0 + $1.totalTokens }

        return BenchmarkSummary(
            modelName: modelName,
            device: device,
            backend: backend,
            mode: config.mode,
            sampleCount: runs.count,
            anomalyCount: anomalyCount,
            ttft: ttft,
            decodeTPS: decode,
            totalLatency: total,
            cpuPercent: cpu,
            peakMemoryMB: memory,
            modelPeakMB: modelPeak,
            modelLoadMB: runs.first?.modelLoadMB ?? 0,
            wasIsolated: isolated,
            throttledRunCount: runs.filter(\.wasThrottled).count,
            meanTokens: Double(tokens) / n,
            targetTokens: Int(config.maxTokens),
            meanTokensPerChunk: chunks > 0 ? Double(tokens) / Double(chunks) : 0)
    }

    // MARK: - CSV

    private static let csvHeader = [
        "timestamp", "mode", "model_name", "device", "backend", "activation",
        "run_index", "is_warmup", "is_anomalous", "reached_target", "length_forced",
        "ttft_ms", "decode_tps", "total_latency_ms",
        "total_tokens", "target_tokens", "chunk_count",
        "prefill_tokens", "prefill_tps",
        "cpu_percent", "cpu_seconds",
        "peak_memory_mb", "memory_growth_mb", "thermal_start", "thermal_end",
        "baseline_memory_mb", "model_load_mb", "model_peak_mb", "isolated",
    ].joined(separator: ",") + "\n"

    /// Append one row, creating the file with headers on first use. Failures are
    /// logged, never thrown — a bad write must not abort a sweep that's already
    /// printing to the console.
    private func appendCSV(_ run: BenchmarkRun) {
        let fields: [String] = [
            ISO8601DateFormatter().string(from: Date()),
            run.mode.rawValue,
            Self.csvField(run.modelName),
            Self.csvField(run.device),
            run.backend.rawValue,
            run.activation.label,
            String(run.runIndex),
            run.isWarmUp ? "true" : "false",
            run.isAnomalous ? "true" : "false",
            run.reachedTarget ? "true" : "false",
            run.lengthWasForced ? "true" : "false",
            String(format: "%.1f", run.ttftMS),
            String(format: "%.2f", run.decodeTPS),
            String(format: "%.1f", run.totalLatencyMS),
            String(run.totalTokens),
            String(run.targetTokens),
            String(run.chunkCount),
            String(run.prefillTokens),
            String(format: "%.2f", run.prefillTPS),
            String(format: "%.1f", run.usage.cpuPercent),
            String(format: "%.3f", run.usage.cpuSeconds),
            String(format: "%.1f", run.usage.peakFootprintMB),
            String(format: "%.1f", run.usage.footprintGrowthMB),
            run.startThermal.label,
            run.endThermal.label,
            String(format: "%.1f", Double(run.baselineFootprintBytes) / 1_048_576),
            String(format: "%.1f", run.modelLoadMB),
            String(format: "%.1f", run.modelPeakMB),
            isIsolated ? "true" : "false",
        ]
        let row = fields.joined(separator: ",") + "\n"

        do {
            if FileManager.default.fileExists(atPath: csvURL.path) {
                let handle = try FileHandle(forWritingTo: csvURL)
                defer { try? handle.close() }
                try handle.seekToEnd()
                try handle.write(contentsOf: Data(row.utf8))
            } else {
                try Data((Self.csvHeader + row).utf8).write(to: csvURL)
            }
        } catch {
            NSLog("[Bench] CSV append failed: %@", error.localizedDescription)
        }
    }

    /// Quote a field only if it needs it, so the common case stays readable.
    private static func csvField(_ value: String) -> String {
        guard value.contains(where: { $0 == "," || $0 == "\"" || $0 == "\n" }) else { return value }
        return "\"" + value.replacingOccurrences(of: "\"", with: "\"\"") + "\""
    }

    // MARK: - Device

    /// Hardware model identifier (e.g. `iPhone11,2`) plus OS version, so a CSV
    /// pooled from several devices stays attributable. On the simulator
    /// `hw.machine` reports the host arch, so the simulated device is used instead.
    private static func deviceIdentifier() -> String {
        var machine = "unknown"
        var size = 0
        if sysctlbyname("hw.machine", nil, &size, nil, 0) == 0, size > 0 {
            var buffer = [CChar](repeating: 0, count: size)
            if sysctlbyname("hw.machine", &buffer, &size, nil, 0) == 0 {
                machine = String(cString: buffer)
            }
        }
        #if targetEnvironment(simulator)
        let simulated = ProcessInfo.processInfo
            .environment["SIMULATOR_MODEL_IDENTIFIER"] ?? machine
        machine = "\(simulated)-sim"
        #endif
        #if canImport(UIKit)
        return "\(machine)/iOS\(UIDevice.current.systemVersion)"
        #else
        return machine
        #endif
    }
}
