import CoreML
import Foundation
import QuartzCore

/// Live, per-chunk streaming transcription (Phase 7). Reuses the same CoreML
/// pieces as the offline `NemotronASRService`, but consumes ONE tier-sized chunk
/// at a time instead of whole segments, threading streaming state across calls:
///
/// - `MelFrameCache` carries the 9-frame mel tail (preprocessor edge context),
/// - `NemotronEncoderRunner` threads `cache_channel/time/len`,
/// - `RNNTDecoder` carries the recurrent predictor state (`token`/`h`/`c`).
///
/// An `actor` so the FIFO consumer loop in `RecordingController` runs every
/// `ingest` off the main actor and strictly serialized — no chunk overlaps
/// another, and the heavy CoreML work never touches the UI thread.
actor StreamingTranscriber {
    private let runner: CoreMLModelRunner
    private let encoder: NemotronEncoderRunner
    private let decoder: RNNTDecoder
    private let tokenizer: Tokenizer
    private let melCache: MelFrameCache

    private let promptID: Int
    private let features: Int
    private let chunkFrames: Int
    private let cacheFrames: Int
    private let windowFrames: Int

    /// Wall-clock seconds the model load took (for the benchmark panel).
    nonisolated let loadSeconds: Double

    private var allTokens: [Int] = []

    private init(runner: CoreMLModelRunner, encoder: NemotronEncoderRunner,
                 decoder: RNNTDecoder, tokenizer: Tokenizer, melCache: MelFrameCache,
                 promptID: Int, features: Int, chunkFrames: Int,
                 cacheFrames: Int, windowFrames: Int, loadSeconds: Double) {
        self.runner = runner
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.melCache = melCache
        self.promptID = promptID
        self.features = features
        self.chunkFrames = chunkFrames
        self.cacheFrames = cacheFrames
        self.windowFrames = windowFrames
        self.loadSeconds = loadSeconds
    }

    /// Load all CoreML modules + build the streaming pipeline. Heavy (~634 MB);
    /// call from a detached/background task. Throws `NemotronASRService.ServiceError`
    /// when the model bundle is missing so the caller can degrade gracefully.
    ///
    /// `modelsDirectoryOverride` / `signaturesOverride` mirror `NemotronASRService`:
    /// tests point them at the repo's checked-in model dir + signatures; in the app
    /// both are nil and models resolve from the bundle via `ModelLocator`.
    static func make(variant: String = "multilingual",
                     tier: StreamingTier = .recommended,
                     language: ASRLanguage,
                     modelsDirectoryOverride: URL? = nil,
                     signaturesOverride: ModelSignatures? = nil) throws -> StreamingTranscriber {
        let dir: URL
        if let override = modelsDirectoryOverride {
            dir = override
        } else {
            let locator = ModelLocator(variant: variant, tier: tier)
            guard case .available(let located) = locator.locate() else {
                if case .missing(let reason) = locator.locate() {
                    throw NemotronASRService.ServiceError.modelsMissing(reason)
                }
                throw NemotronASRService.ServiceError.modelsMissing("unknown")
            }
            dir = located
        }
        let sig: ModelSignatures
        if let override = signaturesOverride {
            sig = override
        } else {
            guard case .success(let loaded) = ModelSignatures.loadFromBundle() else {
                throw NemotronASRService.ServiceError.signaturesMissing
            }
            sig = loaded
        }

        let runner = CoreMLModelRunner(directory: dir, signatures: sig)
        let loadStart = CACurrentMediaTime()
        try runner.load(modules: ["preprocessor", "encoder", "decoder_joint"])
        let loadSeconds = CACurrentMediaTime() - loadStart
        Log.model.info("StreamingTranscriber loaded modules in \(String(format: "%.2f", loadSeconds))s")

        let encoder = try NemotronEncoderRunner(runner: runner)
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
        let melCache = MelFrameCache(features: features, cacheFrames: cacheFrames)

        return StreamingTranscriber(
            runner: runner, encoder: encoder, decoder: decoder, tokenizer: tokenizer,
            melCache: melCache, promptID: promptID, features: features,
            chunkFrames: chunkFrames, cacheFrames: cacheFrames,
            windowFrames: windowFrames, loadSeconds: loadSeconds)
    }

    /// Process one tier-sized chunk (e.g. 35,840 samples @ 2240 ms) and return the
    /// full partial transcript so far. The final stop() tail is processed as a
    /// normal full chunk (it is already zero-padded to tier size); a tiny trailing
    /// silence artifact is acceptable for the PoC.
    func ingest(_ samples: [Float]) throws -> String {
        // 1. Preprocess this chunk in isolation (shared with the offline path).
        let (chunkMelFull, melFrames) = try NemotronASRService.runPreprocessor(
            runner: runner, samples: samples, features: features)
        // 2. Keep the first `chunkFrames` mel columns; the preprocessor emits a
        //    trailing centering frame the offline path also drops.
        guard melFrames >= chunkFrames else {
            throw NemotronASRService.ServiceError.empty(
                "chunk mel \(melFrames) < expected \(chunkFrames)")
        }
        let chunkMel = sliceFirstColumns(chunkMelFull, srcFrames: melFrames, cols: chunkFrames)

        // 3. Prepend the 9-frame mel tail → [features, cacheFrames + chunkFrames].
        let window = melCache.prepend(chunkMel: chunkMel, chunkFrames: chunkFrames)

        // 4. Encode (threads encoder caches forward) → per-frame vectors.
        let frames = try encoder.encode(
            melTotalFrames: window, totalFrames: windowFrames,
            validFrames: cacheFrames + chunkFrames, promptID: promptID)

        // 5. Greedy RNN-T decode, continuing the persistent predictor state.
        let tokens = try decoder.decode(encoderFrames: frames)
        allTokens.append(contentsOf: tokens)
        return tokenizer.decode(allTokens)
    }

    /// Full decode of everything emitted this session (called on stop).
    func finalText() -> String {
        tokenizer.decode(allTokens)
    }

    /// Take the first `cols` columns of a row-major `[features, srcFrames]` mel
    /// buffer → row-major `[features, cols]`.
    private func sliceFirstColumns(_ mel: [Float], srcFrames: Int, cols: Int) -> [Float] {
        var out = [Float](repeating: 0, count: features * cols)
        for f in 0..<features {
            let src = f * srcFrames
            let dst = f * cols
            for t in 0..<cols { out[dst + t] = mel[src + t] }
        }
        return out
    }
}
