import Foundation
#if canImport(UIKit)
import UIKit
#endif

/// CPU and memory cost of one measured stretch of work.
struct ResourceUsage {
    /// CPU seconds consumed by the whole process (all threads, user + system).
    let cpuSeconds: Double
    /// Wall-clock seconds the measurement covered.
    let wallSeconds: Double
    /// Highest physical footprint seen while sampling.
    let peakFootprintBytes: UInt64
    /// Footprint at the start and end, so growth is visible separately from peak.
    let startFootprintBytes: UInt64
    let endFootprintBytes: UInt64

    /// Average CPU utilisation as a percentage. This is process-wide and can
    /// exceed 100%: 200% means two cores were busy for the whole stretch.
    /// Inference is multi-threaded on the CPU backend, so values above 100 are
    /// expected there and are the interesting part.
    var cpuPercent: Double { wallSeconds > 0 ? cpuSeconds / wallSeconds * 100 : 0 }

    var peakFootprintMB: Double { Double(peakFootprintBytes) / 1_048_576 }
    var footprintGrowthMB: Double {
        (Double(endFootprintBytes) - Double(startFootprintBytes)) / 1_048_576
    }
}

/// Reads the current process's CPU time and memory footprint from Mach.
///
/// These are **process-wide**, not per-engine: the numbers include the app's own
/// UI, its other loaded models, and anything else running. During a benchmark
/// the app is idle apart from inference, so the attribution is close, but it is
/// an upper bound on what generation itself costs — not an isolated figure.
enum ProcessMetrics {

    /// Total CPU time consumed by the process so far (user + system, live and
    /// terminated threads). Nil if the kernel call fails.
    static func cpuTime() -> Double? {
        // Live threads.
        var threadInfo = task_thread_times_info_data_t()
        var threadCount = mach_msg_type_number_t(
            MemoryLayout<task_thread_times_info_data_t>.size / MemoryLayout<natural_t>.size)
        let threadResult = withUnsafeMutablePointer(to: &threadInfo) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(threadCount)) {
                task_info(mach_task_self_, task_flavor_t(TASK_THREAD_TIMES_INFO), $0, &threadCount)
            }
        }
        guard threadResult == KERN_SUCCESS else { return nil }

        // Threads that have already exited — without these, a run that spins up
        // and tears down worker threads would under-report.
        var taskInfo = mach_task_basic_info_data_t()
        var taskCount = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info_data_t>.size
                                             / MemoryLayout<natural_t>.size)
        let taskResult = withUnsafeMutablePointer(to: &taskInfo) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(taskCount)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &taskCount)
            }
        }
        guard taskResult == KERN_SUCCESS else { return nil }

        return seconds(threadInfo.user_time) + seconds(threadInfo.system_time)
             + seconds(taskInfo.user_time)   + seconds(taskInfo.system_time)
    }

    /// Physical memory footprint in bytes — the same accounting iOS uses for its
    /// per-app memory limit, so it's the number that predicts a jetsam kill.
    /// `resident_size` would be the wrong choice: it omits compressed pages and
    /// counts some shared memory that doesn't bill to this app.
    static func memoryFootprint() -> UInt64? {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size
                                         / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return nil }
        return UInt64(info.phys_footprint)
    }

    private static func seconds(_ t: time_value_t) -> Double {
        Double(t.seconds) + Double(t.microseconds) / 1_000_000
    }
}

/// Samples memory on a background timer for the duration of one run, and takes
/// CPU time as a delta across it.
///
/// Memory needs polling because the peak matters and it isn't visible from the
/// endpoints — weight conversion and shader compilation spike well above the
/// steady state. CPU doesn't: it's a monotonic counter, so a start/end delta is
/// exact and cheaper than sampling.
final class ResourceSampler {

    /// 50 ms: fine enough to catch a load-time spike, coarse enough that the
    /// sampling itself doesn't show up in the CPU number it's measuring.
    private static let interval: DispatchTimeInterval = .milliseconds(50)

    private let queue = DispatchQueue(label: "com.example.slmchat.bench.sampler",
                                      qos: .utility)
    private var timer: DispatchSourceTimer?

    private let lock = NSLock()
    private var peak: UInt64 = 0

    private var startCPU: Double = 0
    private var startWall: UInt64 = 0
    private var startFootprint: UInt64 = 0

    func start() {
        startCPU = ProcessMetrics.cpuTime() ?? 0
        startWall = DispatchTime.now().uptimeNanoseconds
        startFootprint = ProcessMetrics.memoryFootprint() ?? 0

        lock.lock()
        peak = startFootprint
        lock.unlock()

        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now() + Self.interval, repeating: Self.interval)
        timer.setEventHandler { [weak self] in
            guard let self, let footprint = ProcessMetrics.memoryFootprint() else { return }
            self.lock.lock()
            if footprint > self.peak { self.peak = footprint }
            self.lock.unlock()
        }
        timer.resume()
        self.timer = timer
    }

    /// Stop sampling and return the usage for the stretch since `start()`.
    func stop() -> ResourceUsage {
        timer?.cancel()
        timer = nil

        let endCPU = ProcessMetrics.cpuTime() ?? startCPU
        let endWall = DispatchTime.now().uptimeNanoseconds
        let endFootprint = ProcessMetrics.memoryFootprint() ?? startFootprint

        lock.lock()
        let peakSeen = max(peak, endFootprint)
        lock.unlock()

        return ResourceUsage(
            cpuSeconds: max(0, endCPU - startCPU),
            wallSeconds: Double(endWall - startWall) / 1_000_000_000,
            peakFootprintBytes: peakSeen,
            startFootprintBytes: startFootprint,
            endFootprintBytes: endFootprint)
    }
}

/// Battery and thermal readings for a whole sweep.
///
/// Deliberately NOT per-run. `UIDevice.batteryLevel` reports in 1% steps, and a
/// single 256-token generation takes a few seconds — far too little to move a
/// phone battery by a whole percent. A per-run drain column would read 0.00 on
/// every row and 1% on an arbitrary one, which looks like data and isn't. Across
/// a full sweep of several minutes the delta becomes real, so that's where it's
/// reported.
struct EnergyReading {
    /// 0…1, or nil when the level is unavailable (simulator, or monitoring off).
    let startLevel: Float?
    let endLevel: Float?
    let elapsedSeconds: Double
    /// True if the device was plugged in at any point, which invalidates drain.
    let wasCharging: Bool
    let startThermalState: ProcessInfo.ThermalState
    let endThermalState: ProcessInfo.ThermalState

    /// Battery percentage points consumed. Nil when unmeasurable — including
    /// while charging, where the level can rise and a "drain" would be nonsense.
    var drainPercent: Double? {
        guard !wasCharging, let startLevel, let endLevel,
              startLevel >= 0, endLevel >= 0 else { return nil }
        return Double(startLevel - endLevel) * 100
    }

    /// Drain extrapolated to a rate. Only meaningful once the sweep is long
    /// enough to have moved the level at all — below ~1% of total drain this is
    /// dominated by the 1% reporting granularity, so it's reported alongside the
    /// raw delta rather than instead of it.
    var drainPercentPerHour: Double? {
        guard let drain = drainPercent, elapsedSeconds > 0 else { return nil }
        return drain / (elapsedSeconds / 3600)
    }
}

/// Tracks battery level and thermal state across a sweep.
@MainActor
final class EnergyMonitor {
    private var startLevel: Float?
    private var startThermal: ProcessInfo.ThermalState = .nominal
    private var startWall: UInt64 = 0
    private var sawCharging = false
    private var pollTimer: Timer?

    /// Battery monitoring has to be switched on before `batteryLevel` returns
    /// anything but -1.
    func start() {
        #if canImport(UIKit)
        UIDevice.current.isBatteryMonitoringEnabled = true
        let level = UIDevice.current.batteryLevel
        startLevel = level >= 0 ? level : nil
        sawCharging = UIDevice.current.batteryState != .unplugged
        // The state can change mid-sweep; a phone plugged in halfway through
        // still invalidates the drain figure, so poll rather than trusting the
        // reading taken at the start.
        pollTimer = Timer.scheduledTimer(withTimeInterval: 5, repeats: true) { _ in
            Task { @MainActor in
                if UIDevice.current.batteryState != .unplugged { self.sawCharging = true }
            }
        }
        #endif
        startThermal = ProcessInfo.processInfo.thermalState
        startWall = DispatchTime.now().uptimeNanoseconds
    }

    func stop() -> EnergyReading {
        pollTimer?.invalidate()
        pollTimer = nil

        var endLevel: Float?
        #if canImport(UIKit)
        let level = UIDevice.current.batteryLevel
        endLevel = level >= 0 ? level : nil
        if UIDevice.current.batteryState != .unplugged { sawCharging = true }
        UIDevice.current.isBatteryMonitoringEnabled = false
        #endif

        return EnergyReading(
            startLevel: startLevel,
            endLevel: endLevel,
            elapsedSeconds: Double(DispatchTime.now().uptimeNanoseconds - startWall) / 1_000_000_000,
            wasCharging: sawCharging,
            startThermalState: startThermal,
            endThermalState: ProcessInfo.processInfo.thermalState)
    }
}

extension ProcessInfo.ThermalState {
    var label: String {
        switch self {
        case .nominal:  return "nominal"
        case .fair:     return "fair"
        case .serious:  return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }
}
