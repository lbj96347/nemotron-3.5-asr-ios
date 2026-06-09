import XCTest
@testable import NemotronASRPoC

/// Phase 7 parity: the live per-chunk `StreamingTranscriber` must match the proven
/// offline `NemotronASRService` path. For a single tier-sized chunk the two are
/// exactly equivalent (both preprocess the same one chunk), so we assert equality
/// there; for a multi-chunk clip per-chunk vs continuous preprocessing diverges at
/// STFT edges, so we only assert a non-empty, plausible transcript.
///
/// Model-gated: skips when `Models/multilingual/2240ms` is absent.
final class StreamingTranscriberTests: XCTestCase {

    private static var repoRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // NemotronASRPoCTests/
            .deletingLastPathComponent()   // repo root
    }

    override func setUp() {
        super.setUp()
        executionTimeAllowance = 60 * 60
    }

    // One chunk: streaming and offline consume the identical single chunk, so the
    // transcripts must be byte-for-byte equal — proves the streaming wiring.
    func testStreamingMatchesOfflineSingleChunk() async throws {
        let ctx = try Self.context()
        let oneChunkSeconds = StreamingTier.ms2240.seconds   // 2.24 s → exactly 1 chunk

        let offline = try NemotronASRService(variant: "multilingual", tier: .ms2240)
            .transcribeFile(
                url: ctx.url, language: ctx.language, maxSeconds: oneChunkSeconds,
                modelsDirectoryOverride: ctx.modelsDir, signaturesOverride: ctx.sig)

        let streamed = try await Self.streamTranscript(
            url: ctx.url, language: ctx.language, maxSeconds: oneChunkSeconds, ctx: ctx)

        XCTAssertEqual(streamed, offline.transcript,
                       "streaming single-chunk transcript must match the offline path")
    }

    // Several chunks fed sequentially: state threads correctly and yields a
    // plausible, non-empty transcript (exact match not expected — see file note).
    func testStreamingMultiChunkProducesTranscript() async throws {
        let ctx = try Self.context()
        let streamed = try await Self.streamTranscript(
            url: ctx.url, language: ctx.language, maxSeconds: 12, ctx: ctx)
        XCTAssertGreaterThan(
            streamed.trimmingCharacters(in: .whitespacesAndNewlines).count, 10,
            "multi-chunk streaming transcript is implausibly short")
    }

    // MARK: - Helpers

    private struct Context {
        let url: URL
        let language: ASRLanguage
        let modelsDir: URL
        let sig: ModelSignatures
    }

    /// Resolve models + signatures + a test clip, or skip the test.
    private static func context() throws -> Context {
        let root = repoRoot
        let modelsDir = root.appendingPathComponent("Models/multilingual/2240ms")
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: modelsDir.appendingPathComponent("decoder_joint.mlmodelc").path),
            "models not downloaded — run scripts/download_models.sh")
        let sig = try JSONDecoder().decode(ModelSignatures.self,
            from: Data(contentsOf: root.appendingPathComponent("NemotronASRPoC/ASR/ModelSignatures.json")))

        let testAudioDir = root.appendingPathComponent("TestAudio")
        let exts: Set<String> = ["m4a", "wav", "mp3", "aac", "flac", "caf", "aiff"]
        let entries = (try? FileManager.default.contentsOfDirectory(at: testAudioDir, includingPropertiesForKeys: nil)) ?? []
        let urls = entries
            .filter { exts.contains($0.pathExtension.lowercased()) }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        try XCTSkipUnless(!urls.isEmpty, "no audio files found in TestAudio/")

        // Prefer an English clip for a stable single-chunk comparison.
        let url = urls.first { $0.lastPathComponent.lowercased().contains("-en")
                               || $0.lastPathComponent.lowercased().contains("english") } ?? urls[0]
        let language: ASRLanguage = (url.lastPathComponent.lowercased().contains("-en")
                                     || url.lastPathComponent.lowercased().contains("english"))
            ? .englishUS : .cantoneseTraditional
        return Context(url: url, language: language, modelsDir: modelsDir, sig: sig)
    }

    /// Feed `url` through `StreamingTranscriber` in tier-sized chunks (final chunk
    /// zero-padded, exactly as `AudioChunkBuffer` emits on stop) and return the
    /// final transcript.
    private static func streamTranscript(
        url: URL, language: ASRLanguage, maxSeconds: Double?, ctx: Context
    ) async throws -> String {
        let samples = try AudioFileLoader.load16kMono(url: url, maxSeconds: maxSeconds)
        let transcriber = try StreamingTranscriber.make(
            tier: .ms2240, language: language,
            modelsDirectoryOverride: ctx.modelsDir, signaturesOverride: ctx.sig)

        let per = StreamingTier.ms2240.samplesPerChunk   // 35,840
        var i = 0
        while i < samples.count {
            let end = min(i + per, samples.count)
            var chunk = Array(samples[i..<end])
            if chunk.count < per {
                chunk.append(contentsOf: repeatElement(0, count: per - chunk.count))
            }
            _ = try await transcriber.ingest(chunk)
            i += per
        }
        return await transcriber.finalText()
    }
}
