import SwiftUI

/// Developer screen (Settings → Benchmark) that drives `BenchmarkHarness` over
/// the bundled SLMs and shows the numbers as they land. The same rows go to the
/// console and to the CSV in Documents/ regardless of this screen — it's just
/// the trigger and a live read-out.
struct BenchmarkView: View {
    /// The app's shared service, so the sweep can unload its models to get a
    /// clean memory baseline. Optional so the screen still works standalone.
    var service: ChatService?

    @StateObject private var harness = BenchmarkHarness()

    /// Unloading the app's own SLMs is the default: without it the RAM figures
    /// can't be compared across models.
    @State private var isolateAppModels = true

    /// Which models to sweep, by id. Starts empty and is filled on appear, so a
    /// model added to Resources/ later still shows up.
    @State private var selectedModelIDs: Set<String> = []

    /// Idle between series so each starts from a comparable thermal state.
    @State private var cooldownSeconds = 0

    /// Backends to sweep. Defaults to CPU alone rather than everything runnable:
    /// each added backend doubles the sweep, and the extra sustained load is what
    /// pushes the device into throttling.
    @State private var selectedBackends: Set<BenchmarkBackend> = [.cpu]

    /// Only the comparable pass by default; the production pass doubles the
    /// sweep time and answers a different question.
    @State private var selectedModes: Set<BenchmarkMode> = [.benchmarkFixed]

    /// Every model that could be swept.
    private var availableModels: [SLMModelOption] {
        DesktopModels.taskModels.filter(\.isAvailable)
    }

    /// The models actually selected for this sweep.
    private var models: [SLMModelOption] {
        availableModels.filter { selectedModelIDs.contains($0.id) }
    }

    /// Stable sweep order regardless of tap order.
    private var backends: [BenchmarkBackend] {
        BenchmarkBackend.allCases.filter { selectedBackends.contains($0) }
    }
    private var modes: [BenchmarkMode] {
        BenchmarkMode.allCases.filter { selectedModes.contains($0) }
    }

    private var setupDescription: String {
        let series = models.count * backends.count * modes.count
        let runs = BenchmarkHarness.runsPerSeries
        return "\(models.count) model(s) × \(backends.count) backend(s) × \(modes.count) "
             + "mode(s) = \(series) series of \(runs) runs each (\(series * runs) total). "
             + "The first run of every series is discarded as warm-up. Each backend gets "
             + "its own engine, so this never touches the model or the settings the app's "
             + "features are using."
    }

    var body: some View {
        List {
            modelSection
            modeSection
            backendSection
            cooldownSection
            runSection
            if let energy = harness.energy { energySection(energy) }
            if !harness.summaries.isEmpty { summarySection }
            if !harness.issues.isEmpty { issuesSection }
            if !harness.runs.isEmpty { runsSection }
            csvSection
        }
        .navigationTitle("Benchmark")
        .inlineNavigationTitle()
        .onAppear {
            // Default to one model rather than all: a full sweep heats the device
            // enough to throttle its own later series.
            if selectedModelIDs.isEmpty, let first = availableModels.first {
                selectedModelIDs = [first.id]
            }
        }
    }

    private var modelSection: some View {
        Section {
            ForEach(availableModels) { model in
                Button { toggle(model) } label: {
                    checkRow(title: model.displayName,
                             subtitle: model.subtitle,
                             isOn: selectedModelIDs.contains(model.id))
                }
                .disabled(harness.progress != nil)
            }
        } header: {
            Text("Models")
        } footer: {
            Text("Benchmark one model at a time for the cleanest numbers. Sweeping "
               + "everything back-to-back heats the device, and once it throttles, "
               + "whatever runs last is measured slower than it really is.")
        }
    }

    // MARK: - Sections

    private var modeSection: some View {
        Section {
            ForEach(BenchmarkMode.allCases) { mode in
                Button { toggle(mode) } label: {
                    checkRow(title: mode.displayName,
                             subtitle: mode.explanation,
                             isOn: selectedModes.contains(mode))
                }
                .disabled(harness.progress != nil)
            }
        } header: {
            Text("Mode")
        } footer: {
            Text("The two passes answer different questions and are not comparable to "
               + "each other. Rows are tagged with their mode in the CSV.")
        }
    }

    private var backendSection: some View {
        Section {
            ForEach(BenchmarkBackend.allCases) { backend in
                Button { toggle(backend) } label: {
                    checkRow(title: backend.displayName,
                             subtitle: backend.isLikelyAvailable
                                 ? nil : "Not available in the simulator",
                             isOn: selectedBackends.contains(backend))
                }
                .disabled(harness.progress != nil)
            }
        } header: {
            Text("Backend")
        } footer: {
            Text("Selecting both runs each model on CPU and on Metal so the two can be "
               + "compared directly. A backend a model can't load on is reported and skipped.")
        }
    }

    /// Two cooldown controls, each with a plain-language caption, plus a shared
    /// footer explaining WHY they exist. A "run" is one of the 10 repeats of a
    /// test; a "series" is one whole model × backend × mode.
    private var cooldownSection: some View {
        Section {
            cooldownRow(title: "Between runs",
                        caption: "Pause between each of the 10 repeats of the same test.",
                        options: BenchmarkHarness.runCooldownOptions,
                        selection: $harness.perRunCooldownSeconds)

            cooldownRow(title: "Between series",
                        caption: "Pause when moving to the next model, backend, or mode.",
                        options: BenchmarkHarness.cooldownOptions,
                        selection: $cooldownSeconds)
        } header: {
            Text("Cooldown")
        } footer: {
            Text("Running generations back-to-back heats the chip until it throttles, "
               + "which makes different models report nearly the same speed. Adding a "
               + "pause lets it cool so the numbers stay comparable. Start with "
               + "15–30s between runs; leave both Off for the quickest (but hottest) sweep.")
        }
    }

    /// One labelled segmented cooldown picker with a caption underneath.
    @ViewBuilder
    private func cooldownRow(title: String, caption: String,
                             options: [Int], selection: Binding<Int>) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title).font(.subheadline.weight(.medium))
            Picker(title, selection: selection) {
                ForEach(options, id: \.self) { seconds in
                    Text(seconds == 0 ? "Off" : "\(seconds)s").tag(seconds)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .disabled(harness.progress != nil)
            Text(caption).font(.caption).foregroundStyle(.secondary)
        }
        .padding(.vertical, 2)
    }

    private var runSection: some View {
        Section {
            if service != nil {
                Button { isolateAppModels.toggle() } label: {
                    checkRow(title: "Unload the app's models first",
                             subtitle: "Required for comparable RAM figures",
                             isOn: isolateAppModels)
                }
                .disabled(harness.progress != nil)
            }

            Button {
                Task {
                    await harness.run(models: models,
                                      backends: backends,
                                      modes: modes,
                                      cooldownSeconds: cooldownSeconds,
                                      isolating: isolateAppModels ? service : nil)
                }
            } label: {
                HStack {
                    Text(harness.progress == nil ? "Run benchmark" : "Running…")
                    Spacer()
                    if harness.progress != nil { ProgressView().controlSize(.small) }
                }
            }
            .disabled(harness.progress != nil || models.isEmpty
                      || backends.isEmpty || modes.isEmpty)

            if let progress = harness.progress {
                Text(progress).font(.caption).foregroundStyle(.secondary)
            }
        } footer: {
            Text(setupDescription + (isolateAppModels && service != nil
                 ? " The app's translate/suggest/summarize models are unloaded for the "
                 + "duration and reloaded afterwards — those features won't work until "
                 + "the sweep finishes."
                 : ""))
        }
    }

    /// Battery is reported for the sweep as a whole, not per run — see
    /// `EnergyReading` for why a per-run column would be noise.
    private func energySection(_ energy: EnergyReading) -> some View {
        Section {
            metric("Duration", String(format: "%.0f s", energy.elapsedSeconds))
            if let drain = energy.drainPercent {
                metric("Battery drain", String(format: "%.1f %%", drain))
                if let rate = energy.drainPercentPerHour {
                    metric("Rate", String(format: "%.1f %%/hr", rate))
                }
            } else {
                Text(energy.wasCharging
                     ? "Device was plugged in — drain not measurable."
                     : "Battery level unavailable on this device.")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
            metric("Thermal", "\(energy.startThermalState.label) → \(energy.endThermalState.label)")
        } header: {
            Text("Energy (whole sweep)")
        } footer: {
            Text("iOS reports battery level in 1% steps, so a single run can't move it "
               + "measurably — this covers the entire sweep. Short sweeps will still "
               + "read 0%. Thermal state matters too: once it leaves nominal the CPU is "
               + "being throttled and the speed numbers drop.")
        }
    }

    private var summarySection: some View {
        Section("Results (mean ± sd, warm-up excluded)") {
            ForEach(harness.summaries) { summary in
                VStack(alignment: .leading, spacing: 3) {
                    Text("\(summary.modelName) · \(summary.backend.displayName)")
                        .font(.subheadline.weight(.semibold))
                    Text("\(summary.mode.displayName) · n=\(summary.sampleCount)")
                        .font(.caption2)
                        .foregroundStyle(.secondary)

                    metric("TTFT", summary.ttft.formatted("ms", decimals: 0))
                    metric("Decode", summary.decodeTPS.formatted("tok/s", decimals: 2))
                    metric("Total", summary.totalLatency.formatted("ms", decimals: 0))
                    metric("Tokens", String(format: "%.1f / %d",
                                            summary.meanTokens, summary.targetTokens))
                    metric("Tokens per chunk",
                           String(format: "%.2f", summary.meanTokensPerChunk))
                    metric("CPU", summary.cpuPercent.formatted("%", decimals: 0))
                    metric("Model RAM (weights)",
                           String(format: "%.0f MB", summary.modelLoadMB))
                    metric("Model RAM (peak)",
                           summary.modelPeakMB.formatted("MB", decimals: 0))
                    metric("Process RAM (peak)",
                           summary.peakMemoryMB.formatted("MB", decimals: 0))

                    if summary.throttledRunCount > 0 {
                        Text("\(summary.throttledRunCount) of \(summary.sampleCount) run(s) "
                           + "ran while thermally throttled — treat these speeds as a "
                           + "floor, and rerun with a cooldown to compare fairly.")
                            .font(.caption2)
                            .foregroundStyle(.orange)
                    }
                    if !summary.wasIsolated {
                        Text("App models stayed loaded — RAM includes them, and a model "
                           + "the app also has open shares pages with it, so these are "
                           + "not comparable across models.")
                            .font(.caption2)
                            .foregroundStyle(.orange)
                    }

                    if summary.anomalyCount > 0 {
                        Text("\(summary.anomalyCount) run(s) excluded — forced length "
                           + "didn't hold.")
                            .font(.caption2)
                            .foregroundStyle(.red)
                    }
                }
                .padding(.vertical, 2)
            }
        }
    }

    private var issuesSection: some View {
        Section("Anomalies") {
            ForEach(Array(harness.issues.enumerated()), id: \.offset) { _, issue in
                Text(issue).font(.caption2).foregroundStyle(.orange)
            }
        }
    }

    private var runsSection: some View {
        Section("Runs") {
            ForEach(Array(harness.runs.enumerated()), id: \.offset) { _, run in
                VStack(alignment: .leading, spacing: 2) {
                    Text("\(run.modelName) · \(run.backend.displayName) · run \(run.runIndex)"
                       + (run.isWarmUp ? " (warm-up)" : ""))
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(run.isWarmUp ? .secondary : .primary)
                    Text(run.consoleLine)
                        .font(.caption2.monospaced())
                        .foregroundStyle(run.isAnomalous ? .red : .secondary)
                }
            }
        }
    }

    private var csvSection: some View {
        Section {
            Text(harness.csvURL.path)
                .font(.caption2.monospaced())
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
        } header: {
            Text("CSV")
        } footer: {
            Text("Every run is appended here, tagged with its mode. Pull it off the "
               + "device with Xcode → Devices → the app's container.")
        }
    }

    // MARK: - Helpers

    @ViewBuilder
    private func checkRow(title: String, subtitle: String?, isOn: Bool) -> some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text(title).foregroundStyle(.primary)
                if let subtitle {
                    Text(subtitle).font(.caption).foregroundStyle(.secondary)
                }
            }
            Spacer(minLength: 8)
            if isOn {
                Image(systemName: "checkmark")
                    .font(.body.weight(.semibold))
                    .foregroundStyle(.tint)
            }
        }
        .contentShape(Rectangle())
    }

    private func toggle(_ model: SLMModelOption) {
        if selectedModelIDs.contains(model.id) { selectedModelIDs.remove(model.id) }
        else { selectedModelIDs.insert(model.id) }
    }

    private func toggle(_ backend: BenchmarkBackend) {
        if selectedBackends.contains(backend) { selectedBackends.remove(backend) }
        else { selectedBackends.insert(backend) }
    }

    private func toggle(_ mode: BenchmarkMode) {
        if selectedModes.contains(mode) { selectedModes.remove(mode) }
        else { selectedModes.insert(mode) }
    }

    @ViewBuilder
    private func metric(_ name: String, _ value: String) -> some View {
        HStack {
            Text(name).font(.caption).foregroundStyle(.secondary)
            Spacer(minLength: 8)
            Text(value).font(.caption.monospaced())
        }
    }
}
