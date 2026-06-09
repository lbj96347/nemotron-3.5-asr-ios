import XCTest
@testable import NemotronASRPoC

/// LOOP TARGET (Phases 5–6 done): transcribe the first 60 s of the meeting clip
/// through the full CoreML pipeline and assert a sane, non-empty transcript.
///
/// Loads the real models + audio from the repo (the simulator shares the host
/// filesystem); skips if either is absent rather than failing.
final class EndToEndTranscriptionTests: XCTestCase {

    private static var repoRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // NemotronASRPoCTests/
            .deletingLastPathComponent()   // repo root
    }

    func testTranscribeMeetingClip() throws {
        let root = Self.repoRoot
        let modelsDir = root.appendingPathComponent("Models/multilingual/2240ms")

        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: modelsDir.appendingPathComponent("decoder_joint.mlmodelc").path),
            "models not downloaded — run scripts/download_models.sh")

        // Pick the first English clip in TestAudio/ (name it with an "english"/"-en"
        // hint, e.g. sample-en.m4a). The folder is gitignored — provide your own.
        let exts: Set<String> = ["m4a", "wav", "mp3", "aac", "flac", "caf", "aiff"]
        let testAudioDir = root.appendingPathComponent("TestAudio")
        let entries = (try? FileManager.default.contentsOfDirectory(at: testAudioDir, includingPropertiesForKeys: nil)) ?? []
        let audioURL = entries
            .filter { exts.contains($0.pathExtension.lowercased()) }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
            .first { let n = $0.lastPathComponent.lowercased(); return n.contains("english") || n.contains("-en") }
        try XCTSkipUnless(
            audioURL != nil,
            "no English clip present in TestAudio/ (name it e.g. sample-en.m4a)")
        let clipURL = audioURL!

        let sigURL = root.appendingPathComponent("NemotronASRPoC/ASR/ModelSignatures.json")
        let sig = try JSONDecoder().decode(ModelSignatures.self, from: Data(contentsOf: sigURL))

        let service = NemotronASRService(variant: "multilingual", tier: .ms2240)
        let result = try service.transcribeFile(
            url: clipURL,
            language: .englishUS,
            maxSeconds: 60,
            modelsDirectoryOverride: modelsDir,
            signaturesOverride: sig)

        print("=== TRANSCRIPT (60s) ===")
        print(result.transcript)
        print("=== chunks=\(result.chunkCount) tokens=\(result.tokenIDs.count) audio=\(String(format: "%.1f", result.audioSeconds))s infer=\(String(format: "%.1f", result.inferenceSeconds))s RTF=\(String(format: "%.2f", result.realTimeFactor)) ===")

        // Sanity: non-empty, has real length, contains spaces (word boundaries),
        // not all-blank, and not collapsed to a single repeated token.
        let text = result.transcript.trimmingCharacters(in: .whitespacesAndNewlines)
        XCTAssertGreaterThanOrEqual(text.count, 20, "transcript too short: \(text.count) chars")
        XCTAssertTrue(text.contains(" "), "expected word boundaries / spaces in English transcript")
        XCTAssertFalse(result.tokenIDs.isEmpty, "no tokens emitted (all-blank?)")

        // Repetition guard: a degenerate decode collapses to one token.
        let uniqueTokens = Set(result.tokenIDs)
        XCTAssertGreaterThan(uniqueTokens.count, 5, "degenerate repetition: only \(uniqueTokens.count) unique tokens")
    }
}
