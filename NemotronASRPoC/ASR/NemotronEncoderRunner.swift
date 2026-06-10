import CoreML
import Foundation

/// Owns the streaming encoder and its three pass-through caches, turning a
/// 2240 ms mel chunk into a list of per-frame encoder vectors.
///
/// Caches (`cache_channel`[1,24,42,1024], `cache_time`[1,24,1024,8],
/// `cache_len`[1]) are streaming state: initialised to zeros / len 0, and the
/// `*_out` of chunk N is fed as the `*_in` of chunk N+1. They are NEVER reset
/// within a single audio file. `prompt_id` conditions the encoder only.
final class NemotronEncoderRunner {
    private let runner: CoreMLModelRunner
    private let sig: ModelSignatures

    // Cache tensors carried across chunks.
    private var cacheChannel: MLMultiArray
    private var cacheTime: MLMultiArray
    private var cacheLen: MLMultiArray

    private let cacheChannelShape: [Int]
    private let cacheTimeShape: [Int]

    /// Log the encoder output's shape/strides/dtype once per session to confirm the
    /// on-device (ANE) memory layout the stride-safe reads must handle.
    private var loggedLayout = false

    init(runner: CoreMLModelRunner) throws {
        self.runner = runner
        self.sig = runner.signatures
        let cc = sig.metadata?.cache_channel_shape ?? [1, 24, 42, 1024]
        let ct = sig.metadata?.cache_time_shape ?? [1, 24, 1024, 8]
        self.cacheChannelShape = cc
        self.cacheTimeShape = ct
        self.cacheChannel = try CoreMLModelRunner.zeros(shape: cc)
        self.cacheTime = try CoreMLModelRunner.zeros(shape: ct)
        self.cacheLen = try CoreMLModelRunner.int32Array([0], shape: [1])
    }

    /// Reset streaming state — call only between separate files/sessions.
    func reset() throws {
        cacheChannel = try CoreMLModelRunner.zeros(shape: cacheChannelShape)
        cacheTime = try CoreMLModelRunner.zeros(shape: cacheTimeShape)
        cacheLen = try CoreMLModelRunner.int32Array([0], shape: [1])
    }

    /// Encode one mel window (`[1, 128, 233]` row-major flat) → list of valid
    /// encoder frames, each `[encoder_dim]` (1024). `totalFrames` is the fixed
    /// window size (233); `validFrames` is how many of those are real audio
    /// (< 233 only on a zero-padded final window) and is passed as `mel_length`.
    /// Threads caches forward and honours `encoded_length` on padded final chunks.
    func encode(melTotalFrames: [Float], totalFrames: Int, validFrames: Int, promptID: Int) throws -> [[Float]] {
        let cacheLenIn = CoreMLModelRunner.firstInt(cacheLen)
        // Wrap the prediction + frame extraction in an autorelease pool so the
        // encoder's MLMultiArray outputs don't accumulate across chunks (jetsam
        // guard). The threaded caches are reassigned to instance vars (strong refs)
        // and `frames` is a copied [[Float]], so all survive the drain.
        return try autoreleasepool {
            let features = sig.metadata?.mel_features ?? 128
            let melArray = try CoreMLModelRunner.floatArray(
                melTotalFrames, shape: [1, features, totalFrames])
            let melLen = try CoreMLModelRunner.int32Array([Int32(validFrames)], shape: [1])
            let promptArray = try CoreMLModelRunner.int32Array([Int32(promptID)], shape: [1])

            let inputs: [String: MLFeatureValue] = [
                "mel": MLFeatureValue(multiArray: melArray),
                "mel_length": MLFeatureValue(multiArray: melLen),
                "cache_channel": MLFeatureValue(multiArray: cacheChannel),
                "cache_time": MLFeatureValue(multiArray: cacheTime),
                "cache_len": MLFeatureValue(multiArray: cacheLen),
                "prompt_id": MLFeatureValue(multiArray: promptArray),
            ]

            let out = try runner.predict(module: "encoder", inputs: inputs)

            // Thread caches forward.
            if let cc = out.featureValue(for: "cache_channel_out")?.multiArrayValue { cacheChannel = cc }
            if let ct = out.featureValue(for: "cache_time_out")?.multiArrayValue { cacheTime = ct }
            if let cl = out.featureValue(for: "cache_len_out")?.multiArrayValue { cacheLen = cl }

            guard let encoded = out.featureValue(for: "encoded")?.multiArrayValue else {
                throw CoreMLModelRunner.RunnerError.missingOutput(module: "encoder", name: "encoded")
            }
            let shape = encoded.shape.map { $0.intValue }   // [1, 1024, 28]
            let frameCount = shape.count == 3 ? shape[2] : 0

            // encoded_length tells us how many frames are valid (final padded chunk
            // may produce fewer than `frameCount`).
            var outFrames = frameCount
            var encLenValue = -1
            if let encLen = out.featureValue(for: "encoded_length")?.multiArrayValue {
                encLenValue = CoreMLModelRunner.firstInt(encLen)
                if encLenValue > 0 { outFrames = min(frameCount, encLenValue) }
            }

            let cacheLenOut = CoreMLModelRunner.firstInt(cacheLen)
            Log.stream.debug(
                "encode: mel valid=\(validFrames)/\(totalFrames), cache_len \(cacheLenIn)→\(cacheLenOut), encoded=\(shape.description, privacy: .public), encoded_length=\(encLenValue) → \(outFrames) frames")

            if !loggedLayout {
                loggedLayout = true
                let strides = encoded.strides.map { $0.intValue }
                let contig = CoreMLModelRunner.isContiguous(shape: shape, strides: strides)
                Log.stream.info(
                    "ENCODER OUTPUT LAYOUT: shape=\(shape.description, privacy: .public), strides=\(strides.description, privacy: .public), dtype=\(encoded.dataType.rawValue), contiguous=\(contig)")
            }

            // Read the whole encoded tensor once, stride/dtype-safe, then slice frames
            // from the contiguous [1, dim, frameCount] row-major buffer.
            let flat = CoreMLModelRunner.contiguousFloats(encoded)
            let dim = shape.count == 3 ? shape[1] : (features == 0 ? 0 : flat.count / max(1, frameCount))
            var frames: [[Float]] = []
            frames.reserveCapacity(outFrames)
            for t in 0..<outFrames {
                var v = [Float](repeating: 0, count: dim)
                for d in 0..<dim { v[d] = flat[d * frameCount + t] }
                frames.append(v)
            }
            return frames
        }
    }
}
