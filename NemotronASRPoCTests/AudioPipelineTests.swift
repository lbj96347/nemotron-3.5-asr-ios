import XCTest
import AVFoundation
@testable import NemotronASRPoC

final class AudioPipelineTests: XCTestCase {

    // MARK: - AudioChunkBuffer

    func testChunkBufferEmitsTierAlignedChunks() {
        let buffer = AudioChunkBuffer(tier: .ms2240)   // 35,840 samples @ 16 kHz
        let target = StreamingTier.ms2240.samplesPerChunk
        XCTAssertEqual(target, 35_840)

        // Feed 2.5 chunks worth; expect exactly 2 complete chunks.
        let chunks = buffer.append(Array(repeating: 0.1, count: target * 2 + target / 2))
        XCTAssertEqual(chunks.count, 2)
        XCTAssertTrue(chunks.allSatisfy { $0.count == target })
        XCTAssertEqual(buffer.bufferedSampleCount, target / 2)
    }

    func testChunkBufferFlushPadsToFullChunk() {
        let buffer = AudioChunkBuffer(tier: .ms560)
        let target = StreamingTier.ms560.samplesPerChunk
        _ = buffer.append(Array(repeating: 0.2, count: target / 4))
        let tail = buffer.flushFinal()
        XCTAssertNotNil(tail)
        XCTAssertEqual(tail?.count, target)            // zero-padded up to a full chunk
        XCTAssertNil(buffer.flushFinal())              // nothing left
    }

    func testChunkBufferOverlapCarriesContext() {
        let overlap = 100
        let buffer = AudioChunkBuffer(tier: .ms560, overlapSamples: overlap)
        let target = StreamingTier.ms560.samplesPerChunk
        let first = buffer.append(Array(repeating: 1.0, count: target))
        XCTAssertEqual(first.first?.count, target)     // first chunk has no carry-over
        let second = buffer.append(Array(repeating: 1.0, count: target))
        XCTAssertEqual(second.first?.count, target + overlap)  // second prepends overlap
    }

    // MARK: - AudioResampler

    func testResamplerDownsamplesTo16k() throws {
        guard let inputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: 48_000, channels: 1, interleaved: false
        ) else { return XCTFail("input format") }
        let resampler = try XCTUnwrap(AudioResampler(inputFormat: inputFormat))
        XCTAssertEqual(resampler.outputFormat.sampleRate, 16_000)

        // 0.5 s of 48 kHz silence → ~0.5 s of 16 kHz (8,000 samples ± slack).
        let frames: AVAudioFrameCount = 24_000
        let buffer = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: inputFormat, frameCapacity: frames))
        buffer.frameLength = frames
        let out = resampler.resample(buffer)
        XCTAssertGreaterThan(out.count, 7_000)
        XCTAssertLessThan(out.count, 9_000)
    }

    // MARK: - LatencyTracker

    func testLatencyPercentilesAndRTF() {
        let tracker = LatencyTracker()
        for i in 1...10 {
            tracker.record(inferenceSeconds: Double(i) / 1000.0, audioSeconds: 1.0)
        }
        XCTAssertEqual(tracker.percentile(50), 5.5, accuracy: 0.5)   // median ~5.5 ms
        XCTAssertGreaterThan(tracker.percentile(90), tracker.percentile(50))
        // 55 ms total inference over 10 s audio → RTF 0.0055
        XCTAssertEqual(tracker.realTimeFactor, 0.0055, accuracy: 0.0001)
    }
}
