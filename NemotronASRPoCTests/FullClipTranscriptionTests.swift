import XCTest
@testable import NemotronASRPoC

/// Comprehensive transcription: runs audio files in `TestAudio/` through the full
/// CoreML pipeline (whole file, segmented internally for arbitrary length) and
/// writes each transcript + metrics to `TranscriptOutput/<name>.<lang>.txt` for
/// offline review / cross-model comparison.
///
/// Each file is transcribed with the prompt that fits its language (see
/// `language(for:)`). Long-running (≈ RTF 0.14); no hard time limit. Skips if
/// models / audio are absent.
final class FullClipTranscriptionTests: XCTestCase {

    private static var repoRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // NemotronASRPoCTests/
            .deletingLastPathComponent()   // repo root
    }

    /// Pick the prompt language per file from a language hint in its name.
    /// Cantonese has no dedicated prompt and zh-TW (5) is degenerate on this
    /// model, so Cantonese uses `auto` (101); English uses en-US (0).
    /// Convention: name clips with a hint, e.g. `sample-en.m4a` / `sample-yue.m4a`.
    /// Default: Cantonese/auto.
    private func language(for url: URL) -> ASRLanguage {
        let name = url.lastPathComponent.lowercased()
        if name.contains("english") || name.contains("-en") { return .englishUS }
        if name.contains("cantonese") || name.contains("yue") || name.contains("zh") {
            return .cantoneseTraditional
        }
        return .cantoneseTraditional
    }

    override func setUp() {
        super.setUp()
        executionTimeAllowance = 60 * 60   // no timeout — long clips take minutes
    }

    // Transcribe every file in TestAudio/ with its per-file language.
    func testTranscribeAllTestAudio() throws {
        let urls = try audioURLs()
        for url in urls { try transcribeAndWrite(url: url, language: language(for: url)) }
    }

    // Transcribe just the English clip (name contains an "english"/"-en" hint).
    func testTranscribeEnglishClip() throws {
        let url = try audioURLs().first { language(for: $0) == .englishUS }
        try XCTSkipUnless(url != nil, "no English clip found in TestAudio/ (name it e.g. sample-en.m4a)")
        try transcribeAndWrite(url: url!, language: .englishUS)
    }

    // onProgress fires once per segment, ending at totalSegments; isCancelled aborts.
    func testImportProgressAndCancellation() throws {
        let root = Self.repoRoot
        let modelsDir = root.appendingPathComponent("Models/multilingual/2240ms")
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: modelsDir.appendingPathComponent("decoder_joint.mlmodelc").path),
            "models not downloaded — run scripts/download_models.sh")
        let sig = try JSONDecoder().decode(ModelSignatures.self,
            from: Data(contentsOf: root.appendingPathComponent("NemotronASRPoC/ASR/ModelSignatures.json")))
        let url = try audioURLs().first { language(for: $0) == .englishUS }
        try XCTSkipUnless(url != nil, "no English clip found in TestAudio/ (name it e.g. sample-en.m4a)")

        let service = NemotronASRService(variant: "multilingual", tier: .ms2240)

        // Progress: cap at 5 s → a single segment → onProgress called once with (1, 1).
        var calls: [(Int, Int)] = []
        let result = try service.transcribeFile(
            url: url!, language: .englishUS, maxSeconds: 5,
            modelsDirectoryOverride: modelsDir, signaturesOverride: sig,
            onProgress: { seg, total in calls.append((seg, total)) })
        XCTAssertFalse(calls.isEmpty, "onProgress never fired")
        XCTAssertEqual(calls.count, calls.last?.1, "onProgress call count should equal totalSegments")
        XCTAssertEqual(calls.last?.0, calls.last?.1, "final segment index should equal totalSegments")
        XCTAssertFalse(result.tokenIDs.isEmpty)

        // Cancellation: always-cancelled aborts before any segment completes.
        XCTAssertThrowsError(try service.transcribeFile(
            url: url!, language: .englishUS, maxSeconds: 5,
            modelsDirectoryOverride: modelsDir, signaturesOverride: sig,
            isCancelled: { true })) { error in
            XCTAssertTrue(error is CancellationError, "expected CancellationError, got \(error)")
        }
    }

    // MARK: - Helpers

    private func audioURLs() throws -> [URL] {
        let testAudioDir = Self.repoRoot.appendingPathComponent("TestAudio")
        let exts: Set<String> = ["m4a", "wav", "mp3", "aac", "flac", "caf", "aiff"]
        let entries = (try? FileManager.default.contentsOfDirectory(at: testAudioDir, includingPropertiesForKeys: nil)) ?? []
        let urls = entries
            .filter { exts.contains($0.pathExtension.lowercased()) }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        try XCTSkipUnless(!urls.isEmpty, "no audio files found in TestAudio/")
        return urls
    }

    private func transcribeAndWrite(url: URL, language: ASRLanguage) throws {
        let root = Self.repoRoot
        let modelsDir = root.appendingPathComponent("Models/multilingual/2240ms")
        let outputDir = root.appendingPathComponent("TranscriptOutput")
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: modelsDir.appendingPathComponent("decoder_joint.mlmodelc").path),
            "models not downloaded — run scripts/download_models.sh")

        let sig = try JSONDecoder().decode(ModelSignatures.self,
            from: Data(contentsOf: root.appendingPathComponent("NemotronASRPoC/ASR/ModelSignatures.json")))
        try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)

        let service = NemotronASRService(variant: "multilingual", tier: .ms2240)
        print("=== transcribing \(url.lastPathComponent) [\(language.rawValue)] — whole file ===")
        let started = Date()
        let result = try service.transcribeFile(
            url: url, language: language, maxSeconds: nil,
            modelsDirectoryOverride: modelsDir, signaturesOverride: sig)

        let stamp = ISO8601DateFormatter().string(from: started)
        let body = """
        # Transcript: \(url.lastPathComponent)
        # model:      \(sig.metadata?.model ?? "nemotron-3.5-asr-streaming-multilingual-0.6b")
        # language:   \(language.rawValue) (prompt_id=\(result.promptID))
        # tier:       2240 ms
        # generated:  \(stamp)
        # audio:      \(String(format: "%.1f", result.audioSeconds)) s
        # inference:  \(String(format: "%.1f", result.inferenceSeconds)) s  (RTF \(String(format: "%.3f", result.realTimeFactor)))
        # chunks:     \(result.chunkCount)
        # tokens:     \(result.tokenIDs.count)
        # ---------------------------------------------------------------

        \(result.transcript)

        """
        let base = url.deletingPathExtension().lastPathComponent.replacingOccurrences(of: "/", with: "_")
        let outURL = outputDir.appendingPathComponent("\(base).\(language.rawValue).txt")
        try body.write(to: outURL, atomically: true, encoding: .utf8)
        print("=== wrote \(outURL.path) — \(result.tokenIDs.count) tokens, RTF \(String(format: "%.3f", result.realTimeFactor)) ===")

        XCTAssertFalse(result.tokenIDs.isEmpty, "no tokens emitted for \(url.lastPathComponent)")
        XCTAssertGreaterThan(result.transcript.trimmingCharacters(in: .whitespacesAndNewlines).count, 20,
                             "transcript too short for \(url.lastPathComponent)")
        XCTAssertTrue(FileManager.default.fileExists(atPath: outURL.path), "output file not written")
    }
}
