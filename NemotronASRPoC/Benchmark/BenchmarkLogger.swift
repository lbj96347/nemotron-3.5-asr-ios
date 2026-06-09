import Foundation

/// Immutable snapshot of the metrics shown in the on-screen benchmark panel.
struct BenchmarkSnapshot: Equatable {
    var modelLoadMS: Double
    var firstPartialLatencyMS: Double
    var chunkP50MS: Double
    var chunkP90MS: Double
    var chunkP99MS: Double
    var averageChunkMS: Double
    var realTimeFactor: Double
    var peakMemoryMB: Double
    var currentMemoryMB: Double
    var recordingSeconds: Double
    var thermalState: String

    static let empty = BenchmarkSnapshot(
        modelLoadMS: 0, firstPartialLatencyMS: 0,
        chunkP50MS: 0, chunkP90MS: 0, chunkP99MS: 0, averageChunkMS: 0,
        realTimeFactor: 0, peakMemoryMB: 0, currentMemoryMB: 0,
        recordingSeconds: 0, thermalState: "—"
    )
}

/// Owns benchmark bookkeeping for a session and produces `BenchmarkSnapshot`s
/// plus a CSV export. Drive from a single actor/queue.
final class BenchmarkLogger {
    private let latency = LatencyTracker()
    private(set) var modelLoadMS: Double = 0
    private(set) var firstPartialLatencyMS: Double = 0
    private(set) var peakMemoryMB: Double = 0
    private var recordingStart: CFTimeInterval?
    private var sawFirstPartial = false

    func reset() {
        latency.reset()
        firstPartialLatencyMS = 0
        peakMemoryMB = 0
        recordingStart = nil
        sawFirstPartial = false
    }

    func recordModelLoad(seconds: Double) {
        modelLoadMS = seconds * 1000
    }

    func markRecordingStart(at time: CFTimeInterval) {
        recordingStart = time
        sawFirstPartial = false
    }

    /// Call when the first partial transcript of the session is emitted.
    func markFirstPartial(at time: CFTimeInterval) {
        guard !sawFirstPartial, let start = recordingStart else { return }
        firstPartialLatencyMS = (time - start) * 1000
        sawFirstPartial = true
    }

    func recordChunk(inferenceSeconds: Double, audioSeconds: Double) {
        latency.record(inferenceSeconds: inferenceSeconds, audioSeconds: audioSeconds)
    }

    /// Sample memory; keeps the running peak. Call periodically.
    func sampleMemory() {
        if let mb = MemoryMonitor.physicalFootprintMB() {
            peakMemoryMB = max(peakMemoryMB, mb)
        }
    }

    func snapshot(now: CFTimeInterval) -> BenchmarkSnapshot {
        let elapsed = recordingStart.map { now - $0 } ?? 0
        return BenchmarkSnapshot(
            modelLoadMS: modelLoadMS,
            firstPartialLatencyMS: firstPartialLatencyMS,
            chunkP50MS: latency.percentile(50),
            chunkP90MS: latency.percentile(90),
            chunkP99MS: latency.percentile(99),
            averageChunkMS: latency.averageMS,
            realTimeFactor: latency.realTimeFactor,
            peakMemoryMB: peakMemoryMB,
            currentMemoryMB: MemoryMonitor.physicalFootprintMB() ?? 0,
            recordingSeconds: elapsed,
            thermalState: Self.thermalStateDescription()
        )
    }

    static func thermalStateDescription() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal:  return "Nominal"
        case .fair:     return "Fair"
        case .serious:  return "Serious"
        case .critical: return "Critical"
        @unknown default: return "Unknown"
        }
    }

    /// CSV row export for offline comparison tables.
    func csvHeader() -> String {
        "model_load_ms,first_partial_ms,chunk_p50_ms,chunk_p90_ms,chunk_p99_ms,avg_chunk_ms,rtf,peak_mem_mb,recording_s,thermal"
    }

    func csvRow(now: CFTimeInterval) -> String {
        let s = snapshot(now: now)
        return [
            s.modelLoadMS, s.firstPartialLatencyMS, s.chunkP50MS, s.chunkP90MS,
            s.chunkP99MS, s.averageChunkMS, s.realTimeFactor, s.peakMemoryMB, s.recordingSeconds
        ].map { String(format: "%.3f", $0) }.joined(separator: ",") + ",\(s.thermalState)"
    }
}
