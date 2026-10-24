import Foundation

/// Reads the current process's memory footprint from Mach.
///
/// This is **process-wide**, not per-engine: the number includes the app's own
/// UI, its other loaded models, and anything else running.
enum ProcessMetrics {

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
}
