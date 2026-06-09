import Foundation

/// Streaming tiers supported by the published Nemotron CoreML model.
/// Chunk windows are seconds-scale, NOT the 320 ms the original draft assumed.
enum StreamingTier: Int, CaseIterable, Identifiable {
    case ms560 = 560
    case ms1120 = 1120
    case ms2240 = 2240   // recommended default
    case ms4480 = 4480

    var id: Int { rawValue }
    var seconds: Double { Double(rawValue) / 1000.0 }
    var displayName: String { "\(String(format: "%.2f", seconds)) s" }

    /// Samples per chunk at 16 kHz.
    var samplesPerChunk: Int { Int(seconds * AudioResampler.targetSampleRate) }

    static let recommended: StreamingTier = .ms2240
}

/// Accumulates 16 kHz mono Float32 samples and emits fixed-size chunks aligned
/// to the selected streaming tier. Thread-confine to one queue/actor.
///
/// `overlapSamples` lets a small tail of the previous chunk prepend the next one
/// (lookahead context the streaming encoder may want). Default 0 until the model
/// metadata tells us the real lookahead in Phase 4.
final class AudioChunkBuffer {
    private(set) var tier: StreamingTier
    private var overlapSamples: Int
    private var pending: [Float] = []
    private var carryOver: [Float] = []

    init(tier: StreamingTier = .recommended, overlapSamples: Int = 0) {
        self.tier = tier
        self.overlapSamples = max(0, overlapSamples)
    }

    /// Change the tier; flushes pending samples so chunk sizes stay consistent.
    func setTier(_ tier: StreamingTier, overlapSamples: Int = 0) {
        self.tier = tier
        self.overlapSamples = max(0, overlapSamples)
        reset()
    }

    func reset() {
        pending.removeAll(keepingCapacity: true)
        carryOver.removeAll(keepingCapacity: true)
    }

    /// Append samples; returns any complete chunks that became available.
    func append(_ samples: [Float]) -> [[Float]] {
        pending.append(contentsOf: samples)
        var chunks: [[Float]] = []
        let target = tier.samplesPerChunk

        while pending.count >= target {
            let body = Array(pending.prefix(target))
            pending.removeFirst(target)

            var chunk = carryOver
            chunk.append(contentsOf: body)
            chunks.append(chunk)

            carryOver = overlapSamples > 0 ? Array(body.suffix(overlapSamples)) : []
        }
        return chunks
    }

    /// Emit whatever remains (zero-padded to one chunk) when recording stops.
    /// Returns nil if there is nothing buffered.
    func flushFinal() -> [Float]? {
        guard !pending.isEmpty else { return nil }
        var chunk = carryOver
        chunk.append(contentsOf: pending)
        let target = tier.samplesPerChunk + carryOver.count
        if chunk.count < target {
            chunk.append(contentsOf: repeatElement(0, count: target - chunk.count))
        }
        reset()
        return chunk
    }

    var bufferedSampleCount: Int { pending.count }
}
