import Foundation

/// Maintains the streaming mel carry-over the encoder requires.
///
/// The preprocessor emits 224 mel frames per 2240 ms chunk, but the encoder
/// input is `[1, 128, 233]` = `pre_encode_cache` (9) + 224. The 9-frame tail is
/// the caller's responsibility: zeros on the first chunk, then the last 9 frames
/// of the previous chunk's mel prepended to the current one.
///
/// Mel buffers are row-major `[1, features, frames]`, i.e. element (0, f, t) is
/// at index `f * frames + t`.
final class MelFrameCache {
    let features: Int          // 128
    let cacheFrames: Int       // 9 (pre_encode_cache)

    /// The retained tail, stored as `[features, cacheFrames]` row-major
    /// (element (f, t) at `f * cacheFrames + t`). Starts as zeros.
    private var tail: [Float]

    init(features: Int, cacheFrames: Int) {
        self.features = features
        self.cacheFrames = cacheFrames
        self.tail = [Float](repeating: 0, count: features * cacheFrames)
    }

    func reset() {
        tail = [Float](repeating: 0, count: features * cacheFrames)
    }

    /// Prepend the retained tail to `chunkMel` (`[features, chunkFrames]`
    /// row-major) and return `[features, cacheFrames + chunkFrames]` row-major.
    /// Also updates the tail to the last `cacheFrames` frames of `chunkMel`.
    func prepend(chunkMel: [Float], chunkFrames: Int) -> [Float] {
        precondition(chunkMel.count == features * chunkFrames,
                     "chunkMel count \(chunkMel.count) != \(features)*\(chunkFrames)")
        let totalFrames = cacheFrames + chunkFrames
        var out = [Float](repeating: 0, count: features * totalFrames)
        for f in 0..<features {
            let outRow = f * totalFrames
            let tailRow = f * cacheFrames
            // tail frames first
            for t in 0..<cacheFrames {
                out[outRow + t] = tail[tailRow + t]
            }
            // then this chunk's frames
            let chunkRow = f * chunkFrames
            for t in 0..<chunkFrames {
                out[outRow + cacheFrames + t] = chunkMel[chunkRow + t]
            }
        }
        updateTail(from: chunkMel, chunkFrames: chunkFrames)
        return out
    }

    /// Retain the last `cacheFrames` frames of `chunkMel` as the next tail.
    private func updateTail(from chunkMel: [Float], chunkFrames: Int) {
        var next = [Float](repeating: 0, count: features * cacheFrames)
        let start = max(0, chunkFrames - cacheFrames)
        let available = chunkFrames - start
        for f in 0..<features {
            let chunkRow = f * chunkFrames
            let tailRow = f * cacheFrames
            // right-align if the chunk had fewer than cacheFrames frames
            let pad = cacheFrames - available
            for t in 0..<available {
                next[tailRow + pad + t] = chunkMel[chunkRow + start + t]
            }
        }
        tail = next
    }
}
