import Foundation

/// Reads the process's physical memory footprint via `task_vm_info`.
/// `phys_footprint` is the value iOS Jetsam uses to decide kills, so it is the
/// right number to watch against the 1.5 GB PoC ceiling.
enum MemoryMonitor {

    /// Current physical footprint in bytes, or nil if the syscall fails.
    static func physicalFootprintBytes() -> UInt64? {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)

        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { intPointer in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), intPointer, &count)
            }
        }

        guard result == KERN_SUCCESS else { return nil }
        return UInt64(info.phys_footprint)
    }

    static func physicalFootprintMB() -> Double? {
        guard let bytes = physicalFootprintBytes() else { return nil }
        return Double(bytes) / (1024 * 1024)
    }
}
