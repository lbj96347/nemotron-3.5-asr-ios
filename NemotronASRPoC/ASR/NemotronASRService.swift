import CoreML
import Foundation
import QuartzCore

/// End-to-end offline transcription of an audio file through the full Nemotron
/// CoreML pipeline: load file → 16 kHz mono → 2240 ms chunks → preprocessor →
/// streaming encoder → RNN-T greedy decode → tokenizer.
///
/// This is the Phase 6 target consumed by `EndToEndTranscriptionTests`. The live
/// mic path (Phase 7) will reuse the same encoder/decoder/tokenizer pieces.
final class NemotronASRService {

    struct Result {
        let transcript: String
        let tokenIDs: [Int]
        let chunkCount: Int
        let audioSeconds: Double
        let inferenceSeconds: Double
        var promptID: Int = -1
        var realTimeFactor: Double { audioSeconds > 0 ? inferenceSeconds / audioSeconds : 0 }
    }

    enum ServiceError: Error, CustomStringConvertible {
        case modelsMissing(String)
        case signaturesMissing
        case empty(String)
        var description: String {
            switch self {
            case .modelsMissing(let r): return "models missing: \(r)"
            case .signaturesMissing: return "ModelSignatures.json missing"
            case .empty(let r): return r
            }
        }
    }

    private let variant: String
    private let tier: StreamingTier

    init(variant: String = "multilingual", tier: StreamingTier = .recommended) {
        self.variant = variant
        self.tier = tier
    }

    /// Transcribe up to `maxSeconds` of `url` in `language`.
    ///
    /// `modelsDirectoryOverride` / `signaturesOverride` let tests point at the
    /// repo's checked-in model dir + signatures (simulator shares the host FS),
    /// independent of app-bundle packaging. In the app, both are nil and the
    /// models are resolved from the bundle via `ModelLocator`.
    /// `onProgress` is called after each segment completes with (segment, total)
    /// for live UI feedback. `isCancelled`, if it returns true at a segment
    /// boundary, aborts the run by throwing `CancellationError`.
    func transcribeFile(
        url: URL,
        language: ASRLanguage,
        maxSeconds: Double? = 60,
        modelsDirectoryOverride: URL? = nil,
        signaturesOverride: ModelSignatures? = nil,
        onProgress: ((_ segment: Int, _ totalSegments: Int) -> Void)? = nil,
        isCancelled: (() -> Bool)? = nil
    ) throws -> Result {
        // 1. Locate models + signatures.
        let dir: URL
        if let override = modelsDirectoryOverride {
            dir = override
        } else {
            let locator = ModelLocator(variant: variant, tier: tier)
            guard case .available(let located) = locator.locate() else {
                if case .missing(let reason) = locator.locate() {
                    throw ServiceError.modelsMissing(reason)
                }
                throw ServiceError.modelsMissing("unknown")
            }
            dir = located
        }
        let sig: ModelSignatures
        if let override = signaturesOverride {
            sig = override
        } else {
            guard case .success(let loaded) = ModelSignatures.loadFromBundle() else {
                throw ServiceError.signaturesMissing
            }
            sig = loaded
        }

        // 2. Load CoreML modules.
        let runner = CoreMLModelRunner(directory: dir, signatures: sig)
        let loadStart = CACurrentMediaTime()
        try runner.load(modules: ["preprocessor", "encoder", "decoder_joint"])
        Log.model.info("All modules loaded in \(String(format: "%.2f", CACurrentMediaTime() - loadStart))s")

        // 3. Build pipeline components.
        let encoderRunner = try NemotronEncoderRunner(runner: runner)
        let decoder = try RNNTDecoder(runner: runner)
        let tokenizer = try Tokenizer.load(
            contentsOf: dir.appendingPathComponent("tokenizer.json"),
            blankID: sig.blankTokenID ?? 13087)

        let promptID = sig.promptID(forLanguageCode: language.rawValue)
            ?? sig.metadata?.default_prompt_id ?? 101

        let features = sig.metadata?.mel_features ?? 128
        let chunkFrames = sig.metadata?.chunk_mel_frames ?? 224
        let cacheFrames = sig.metadata?.pre_encode_cache ?? 9
        let windowFrames = sig.metadata?.total_mel_frames ?? (cacheFrames + chunkFrames)

        // 4. Load audio (≤ maxSeconds; nil = whole file → arbitrary length).
        let samples = try AudioFileLoader.load16kMono(url: url, maxSeconds: maxSeconds)
        guard !samples.isEmpty else { throw ServiceError.empty("no audio samples decoded") }
        let audioSeconds = Double(samples.count) / AudioResampler.targetSampleRate

        // The preprocessor accepts up to ~80 s per call, so process the clip in
        // SEGMENTS of whole 2240 ms chunks (35 × 35,840 = 1,254,400 samples ≈ 78.4 s,
        // under the 1,280,000-sample cap). Each segment is preprocessed once
        // (continuous mel, no per-chunk STFT edges); the 9-frame mel tail, the
        // encoder caches, and the RNN-T predictor state are carried ACROSS segment
        // seams so the whole file decodes as one continuous stream. A clip ≤ one
        // segment behaves exactly like a single-shot continuous pass.
        let samplesPerChunk = tier.samplesPerChunk                 // 35,840
        let chunksPerSegment = 35
        let segmentSamples = chunksPerSegment * samplesPerChunk
        let hop = samplesPerChunk / chunkFrames                    // 160

        // carriedTail = last `cacheFrames` real mel frames of the previous segment
        // (zeros before the first). Row-major [features, cacheFrames].
        var carriedTail = [Float](repeating: 0, count: features * cacheFrames)

        var allTokens: [Int] = []
        var totalChunks = 0
        let totalSegments = (samples.count + segmentSamples - 1) / segmentSamples
        let inferStart = CACurrentMediaTime()

        var segStart = 0
        var segIndex = 0
        while segStart < samples.count {
            if isCancelled?() == true { throw CancellationError() }
            let segEnd = min(segStart + segmentSamples, samples.count)
            let realLen = segEnd - segStart
            let chunksInSeg = (realLen + samplesPerChunk - 1) / samplesPerChunk
            let paddedLen = chunksInSeg * samplesPerChunk

            // Pad the (possibly final) segment up to a whole number of chunks so
            // the preprocessor yields exactly chunksInSeg*chunkFrames (+1 centering)
            // frames — no frame-count drift across segments.
            var segSamples = Array(samples[segStart..<segEnd])
            if paddedLen > realLen {
                segSamples.append(contentsOf: repeatElement(0, count: paddedLen - realLen))
            }

            let (segMel, segMelFrames) = try runPreprocessor(runner: runner, samples: segSamples, features: features)
            let realCols = chunksInSeg * chunkFrames               // frames we consume
            guard segMelFrames >= realCols else {
                throw ServiceError.empty("segment mel \(segMelFrames) < expected \(realCols)")
            }
            // Real (non-padded) mel frames in this segment, for the final partial chunk.
            let realMelFrames = realLen / hop

            // extendedMel = carriedTail ++ segMel[0..<realCols] → [features, cacheFrames+realCols].
            // Window k then occupies extendedMel columns [k*chunkFrames, +windowFrames).
            let extCols = cacheFrames + realCols
            var extendedMel = [Float](repeating: 0, count: features * extCols)
            for f in 0..<features {
                let dst = f * extCols
                for t in 0..<cacheFrames { extendedMel[dst + t] = carriedTail[f * cacheFrames + t] }
                let srcRow = f * segMelFrames
                for t in 0..<realCols { extendedMel[dst + cacheFrames + t] = segMel[srcRow + t] }
            }

            for k in 0..<chunksInSeg {
                let validNew = max(0, min(chunkFrames, realMelFrames - k * chunkFrames))
                let window = melWindow(
                    mel: extendedMel, melFrames: extCols, features: features,
                    chunkStart: k * chunkFrames + cacheFrames, chunkFrames: chunkFrames,
                    cacheFrames: cacheFrames, windowFrames: windowFrames)
                let frames = try encoderRunner.encode(
                    melTotalFrames: window, totalFrames: windowFrames,
                    validFrames: cacheFrames + validNew, promptID: promptID)
                let tokens = try decoder.decode(encoderFrames: frames)
                allTokens.append(contentsOf: tokens)
                totalChunks += 1
            }

            // Carry the last `cacheFrames` real mel frames into the next segment.
            for f in 0..<features {
                let srcRow = f * segMelFrames
                for t in 0..<cacheFrames {
                    carriedTail[f * cacheFrames + t] = segMel[srcRow + (realCols - cacheFrames + t)]
                }
            }

            segIndex += 1
            Log.asr.info("segment \(segIndex)/\(totalSegments): \(chunksInSeg) chunks → \(allTokens.count) tokens so far")
            onProgress?(segIndex, totalSegments)
            segStart += segmentSamples
        }
        let inferenceSeconds = CACurrentMediaTime() - inferStart

        let transcript = tokenizer.decode(allTokens)
        return Result(
            transcript: transcript,
            tokenIDs: allTokens,
            chunkCount: totalChunks,
            audioSeconds: audioSeconds,
            inferenceSeconds: inferenceSeconds,
            promptID: promptID)
    }

    // MARK: - Helpers

    /// Run the preprocessor on the full sample buffer → mel `[features, frames]`
    /// row-major flat + frame count.
    private func runPreprocessor(runner: CoreMLModelRunner, samples: [Float], features: Int) throws -> ([Float], Int) {
        let audio = try CoreMLModelRunner.floatArray(samples, shape: [1, samples.count])
        let audioLen = try CoreMLModelRunner.int32Array([Int32(samples.count)], shape: [1])
        let out = try runner.predict(module: "preprocessor", inputs: [
            "audio": MLFeatureValue(multiArray: audio),
            "audio_length": MLFeatureValue(multiArray: audioLen),
        ])
        guard let mel = out.featureValue(for: "mel")?.multiArrayValue else {
            throw CoreMLModelRunner.RunnerError.missingOutput(module: "preprocessor", name: "mel")
        }
        let shape = mel.shape.map { $0.intValue }   // [1, features, frames]
        let frames = shape.count == 3 ? shape[2] : 0
        let flat = CoreMLModelRunner.toFloats(mel)
        return (flat, frames)
    }

    /// Build one `[features, windowFrames]` row-major encoder window from the
    /// continuous mel: `cacheFrames` of real left context (zeros where the index
    /// is before 0) followed by up to `chunkFrames` new frames, zero-padded to
    /// `windowFrames` when the clip ends mid-window.
    private func melWindow(mel: [Float], melFrames: Int, features: Int,
                           chunkStart: Int, chunkFrames: Int,
                           cacheFrames: Int, windowFrames: Int) -> [Float] {
        var window = [Float](repeating: 0, count: features * windowFrames)
        // Source frame for window position p is: chunkStart - cacheFrames + p.
        for f in 0..<features {
            let srcRow = f * melFrames
            let dstRow = f * windowFrames
            for p in 0..<windowFrames {
                let src = chunkStart - cacheFrames + p
                if src >= 0 && src < melFrames {
                    window[dstRow + p] = mel[srcRow + src]
                }
            }
        }
        return window
    }
}
