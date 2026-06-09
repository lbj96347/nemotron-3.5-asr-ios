import XCTest
import CoreML
@testable import NemotronASRPoC

/// Phase 5 verification: the CoreML modules load and produce the expected
/// shapes, validating the critical 224-vs-233 mel carry before the decode loop.
///
/// These tests load the real `.mlmodelc` bundles from the repo's `Models/` dir
/// (the simulator shares the host filesystem). If the models aren't downloaded,
/// the tests skip rather than fail.
final class CoreMLPipelineTests: XCTestCase {

    /// repo root, derived from this file's path.
    private static var repoRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // NemotronASRPoCTests/
            .deletingLastPathComponent()   // repo root
    }

    private static var modelsDir: URL {
        repoRoot.appendingPathComponent("Models/multilingual/2240ms")
    }

    private func loadSignatures() throws -> ModelSignatures {
        let url = Self.repoRoot.appendingPathComponent("NemotronASRPoC/ASR/ModelSignatures.json")
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(ModelSignatures.self, from: data)
    }

    private func requireModels() throws {
        let dj = Self.modelsDir.appendingPathComponent("decoder_joint.mlmodelc")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: dj.path),
                          "models not downloaded — run scripts/download_models.sh")
    }

    func testPreprocessorAndEncoderShapes() throws {
        try requireModels()
        let sig = try loadSignatures()
        let runner = CoreMLModelRunner(directory: Self.modelsDir, signatures: sig)
        try runner.load(modules: ["preprocessor", "encoder"])

        // 2240 ms of silence = 35,840 samples @ 16 kHz.
        let samplesPerChunk = StreamingTier.ms2240.samplesPerChunk
        XCTAssertEqual(samplesPerChunk, 35_840)
        let audio = try CoreMLModelRunner.floatArray(
            [Float](repeating: 0, count: samplesPerChunk), shape: [1, samplesPerChunk])
        let audioLen = try CoreMLModelRunner.int32Array([Int32(samplesPerChunk)], shape: [1])

        let pre = try runner.predict(module: "preprocessor", inputs: [
            "audio": MLFeatureValue(multiArray: audio),
            "audio_length": MLFeatureValue(multiArray: audioLen),
        ])
        let mel = try XCTUnwrap(pre.featureValue(for: "mel")?.multiArrayValue)
        let melShape = mel.shape.map { $0.intValue }
        XCTAssertEqual(melShape[0], 1)
        XCTAssertEqual(melShape[1], 128)
        // Centered STFT yields 1 + samples/hop = 225 frames; the chunk stride is
        // chunk_mel_frames (224). The extra frame is the centering artifact.
        let emittedFrames = melShape[2]
        XCTAssertGreaterThanOrEqual(emittedFrames, 224, "expected ≥224 mel frames, got \(emittedFrames)")

        // Take the first 224 chunk frames, prepend a 9-frame zero tail → [1,128,233].
        let features = 128, chunkFrames = 224, cacheFrames = 9
        let flat = CoreMLModelRunner.toFloats(mel)   // [features, emittedFrames] row-major
        var melWithTail = [Float](repeating: 0, count: features * (cacheFrames + chunkFrames))
        let total = cacheFrames + chunkFrames
        for f in 0..<features {
            for t in 0..<chunkFrames {
                melWithTail[f * total + cacheFrames + t] = flat[f * emittedFrames + t]
            }
        }
        XCTAssertEqual(melWithTail.count, 128 * 233)

        let melArray = try CoreMLModelRunner.floatArray(melWithTail, shape: [1, 128, 233])
        let melLen = try CoreMLModelRunner.int32Array([233], shape: [1])
        let cacheChannel = try CoreMLModelRunner.zeros(shape: [1, 24, 42, 1024])
        let cacheTime = try CoreMLModelRunner.zeros(shape: [1, 24, 1024, 8])
        let cacheLen = try CoreMLModelRunner.int32Array([0], shape: [1])
        let promptID = try CoreMLModelRunner.int32Array([0], shape: [1])  // en-US

        let enc = try runner.predict(module: "encoder", inputs: [
            "mel": MLFeatureValue(multiArray: melArray),
            "mel_length": MLFeatureValue(multiArray: melLen),
            "cache_channel": MLFeatureValue(multiArray: cacheChannel),
            "cache_time": MLFeatureValue(multiArray: cacheTime),
            "cache_len": MLFeatureValue(multiArray: cacheLen),
            "prompt_id": MLFeatureValue(multiArray: promptID),
        ])
        let encoded = try XCTUnwrap(enc.featureValue(for: "encoded")?.multiArrayValue)
        let encShape = encoded.shape.map { $0.intValue }
        XCTAssertEqual(encShape, [1, 1024, 28], "encoder should emit [1,1024,28] per 2.24s chunk")
        XCTAssertNotNil(enc.featureValue(for: "cache_channel_out")?.multiArrayValue)
        XCTAssertNotNil(enc.featureValue(for: "encoded_length")?.multiArrayValue)
    }
}
