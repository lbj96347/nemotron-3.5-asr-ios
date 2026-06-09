import Foundation
import QuartzCore

/// Accumulates per-chunk inference latencies and computes percentiles + RTF.
/// Not thread-safe; drive it from a single actor/queue.
final class LatencyTracker {
    private(set) var samplesMS: [Double] = []
    private(set) var totalAudioSeconds: Double = 0
    private(set) var totalInferenceSeconds: Double = 0

    func reset() {
        samplesMS.removeAll(keepingCapacity: true)
        totalAudioSeconds = 0
        totalInferenceSeconds = 0
    }

    /// Record one chunk: how long inference took and how much audio it covered.
    func record(inferenceSeconds: Double, audioSeconds: Double) {
        samplesMS.append(inferenceSeconds * 1000)
        totalInferenceSeconds += inferenceSeconds
        totalAudioSeconds += audioSeconds
    }

    /// Real-time factor over the whole session (< 1.0 means faster than real time).
    var realTimeFactor: Double {
        guard totalAudioSeconds > 0 else { return 0 }
        return totalInferenceSeconds / totalAudioSeconds
    }

    func percentile(_ p: Double) -> Double {
        guard !samplesMS.isEmpty else { return 0 }
        let sorted = samplesMS.sorted()
        let rank = max(0, min(Double(sorted.count - 1), p / 100 * Double(sorted.count - 1)))
        let lower = Int(rank.rounded(.down))
        let upper = Int(rank.rounded(.up))
        if lower == upper { return sorted[lower] }
        let weight = rank - Double(lower)
        return sorted[lower] * (1 - weight) + sorted[upper] * weight
    }

    var averageMS: Double {
        guard !samplesMS.isEmpty else { return 0 }
        return samplesMS.reduce(0, +) / Double(samplesMS.count)
    }

    /// Convenience to time a synchronous block.
    @discardableResult
    static func measure<T>(_ body: () throws -> T) rethrows -> (value: T, seconds: Double) {
        let start = CACurrentMediaTime()
        let value = try body()
        return (value, CACurrentMediaTime() - start)
    }
}
