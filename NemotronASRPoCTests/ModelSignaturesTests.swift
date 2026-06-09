import XCTest
@testable import NemotronASRPoC

/// Validates the Phase 4 deliverable: ModelSignatures.json decodes and exposes
/// the real tensor names, shapes, vocab, and language→prompt_id map. Loads the
/// repo's checked-in JSON via #filePath (simulator tests share the host FS).
final class ModelSignaturesTests: XCTestCase {

    private func loadSignatures() throws -> ModelSignatures {
        let here = URL(fileURLWithPath: #filePath)
        let jsonURL = here
            .deletingLastPathComponent()      // NemotronASRPoCTests/
            .deletingLastPathComponent()      // repo root
            .appendingPathComponent("NemotronASRPoC/ASR/ModelSignatures.json")
        let data = try Data(contentsOf: jsonURL)
        return try JSONDecoder().decode(ModelSignatures.self, from: data)
    }

    func testCoreSignaturesPresent() throws {
        let sig = try loadSignatures()

        // Encoder is stateful via explicit cache tensors + prompt conditioning.
        let encoder = try XCTUnwrap(sig.modules["encoder"])
        XCTAssertNotNil(encoder.input("mel"))
        XCTAssertNotNil(encoder.input("prompt_id"))
        XCTAssertEqual(encoder.input("cache_channel")?.dimensions, [1, 24, 42, 1024])
        XCTAssertEqual(encoder.input("cache_time")?.dimensions, [1, 24, 1024, 8])
        XCTAssertNotNil(encoder.output("cache_channel_out"))
        XCTAssertNotNil(encoder.output("cache_len_out"))

        // Fused decoder⊕joint is the default RNN-T step.
        let dj = try XCTUnwrap(sig.modules["decoder_joint"])
        XCTAssertEqual(dj.input("h_in")?.dimensions, [2, 1, 640])   // 2-layer LSTM, hidden 640
        XCTAssertEqual(dj.input("encoder")?.dimensions, [1, 1024, 1])
        // logits over vocab (13087) + blank = 13088
        XCTAssertEqual(dj.output("logits")?.dimensions, [1, 1, 1, 13088])
    }

    func testVocabAndBlank() throws {
        let sig = try loadSignatures()
        XCTAssertEqual(sig.vocabSize, 13087)
        XCTAssertEqual(sig.blankTokenID, 13087)
        XCTAssertEqual(sig.metadata?.total_mel_frames, 233)
    }

    func testPromptIDResolution() throws {
        let sig = try loadSignatures()
        XCTAssertEqual(sig.promptID(forLanguageCode: "en-US"), 0)
        XCTAssertEqual(sig.promptID(forLanguageCode: "zh-CN"), 4)
        XCTAssertEqual(sig.promptID(forLanguageCode: "zh-TW"), 5)
        XCTAssertEqual(sig.promptID(forLanguageCode: "ja-JP"), 10)
        XCTAssertEqual(sig.promptID(forLanguageCode: "ko-KR"), 14)
        // Base-language fallback: an unlisted region resolves via "de".
        XCTAssertEqual(sig.promptID(forLanguageCode: "de-AT"), 9)
        // Cantonese has NO dedicated prompt → falls back to default (auto=101).
        XCTAssertEqual(sig.promptID(forLanguageCode: "yue-Hant-HK"), sig.metadata?.default_prompt_id)
        XCTAssertEqual(sig.metadata?.default_prompt_id, 101)
    }

    func testLanguagePromptMapCoversUILanguages() throws {
        let sig = try loadSignatures()
        let map = sig.languagePromptMap()
        XCTAssertFalse(map.isEmpty)
        XCTAssertEqual(map.promptID(for: .englishUS), 0)
        XCTAssertEqual(map.promptID(for: .japanese), 10)
        XCTAssertEqual(map.promptID(for: .korean), 14)
    }
}
