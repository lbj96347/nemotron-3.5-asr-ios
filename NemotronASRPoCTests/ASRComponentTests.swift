import XCTest
@testable import NemotronASRPoC

/// Pure-logic unit tests for the Phase 6 components that don't need the CoreML
/// weights: tokenizer detokenization, mel-frame carry-over, and the greedy
/// RNN-T control flow (stubbed joint).
final class ASRComponentTests: XCTestCase {

    // MARK: - Tokenizer

    private func makeTokenizer() -> Tokenizer {
        // ▁ = U+2581 word boundary. blank = 99 here for the test.
        let pieces: [Int: String] = [
            0: "\u{2581}the",
            1: "\u{2581}quick",
            2: "\u{2581}brown",
            3: "fox",
            4: "\u{2581}",
            5: "<unk>",
            6: "<en-US>",
            10: "你",
            11: "好",
        ]
        return Tokenizer(pieces: pieces, blankID: 99)
    }

    func testDetokenizeWordBoundaries() {
        let tok = makeTokenizer()
        // "the quick brown" + "fox" (no boundary) → "brownfox"
        XCTAssertEqual(tok.decode([0, 1, 2, 3]), "the quick brownfox")
    }

    func testDetokenizeDropsBlankAndSpecials() {
        let tok = makeTokenizer()
        // blank (99) and special tokens (<unk>, <en-US>) are stripped.
        XCTAssertEqual(tok.decode([99, 5, 0, 6, 1, 99]), "the quick")
    }

    func testDetokenizeCJKNoSpaces() {
        let tok = makeTokenizer()
        // CJK pieces have no ▁, so they concatenate without spaces.
        XCTAssertEqual(tok.decode([10, 11]), "你好")
    }

    func testCollapseWhitespaceAndTrim() {
        // A lone ▁ becomes a space; leading/trailing/double spaces collapse.
        let tok = makeTokenizer()
        XCTAssertEqual(tok.decode([4, 0, 4, 4, 1]), "the quick")
    }

    func testIsSpecial() {
        XCTAssertTrue(Tokenizer.isSpecial("<unk>"))
        XCTAssertTrue(Tokenizer.isSpecial("<en-US>"))
        XCTAssertFalse(Tokenizer.isSpecial("hello"))
        XCTAssertFalse(Tokenizer.isSpecial("<"))
    }

    // MARK: - MelFrameCache

    func testMelFrameCachePrependsZerosOnFirstChunk() {
        // 2 features, 3-frame cache, chunk of 4 frames.
        let cache = MelFrameCache(features: 2, cacheFrames: 3)
        // chunkMel [features=2, frames=4] row-major:
        // f0: 1 2 3 4 ; f1: 5 6 7 8
        let chunk: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
        let out = cache.prepend(chunkMel: chunk, chunkFrames: 4)
        // total frames = 3 + 4 = 7
        // f0: [0,0,0, 1,2,3,4] ; f1: [0,0,0, 5,6,7,8]
        XCTAssertEqual(out, [0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 5, 6, 7, 8])
    }

    func testMelFrameCacheCarriesTailToNextChunk() {
        let cache = MelFrameCache(features: 1, cacheFrames: 2)
        // First chunk f0: [1,2,3,4] → tail becomes last 2 frames = [3,4]
        _ = cache.prepend(chunkMel: [1, 2, 3, 4], chunkFrames: 4)
        // Second chunk f0: [5,6,7,8] → prepend [3,4]
        let out = cache.prepend(chunkMel: [5, 6, 7, 8], chunkFrames: 4)
        XCTAssertEqual(out, [3, 4, 5, 6, 7, 8])
    }

    func testMelFrameCacheReset() {
        let cache = MelFrameCache(features: 1, cacheFrames: 2)
        _ = cache.prepend(chunkMel: [1, 2, 3, 4], chunkFrames: 4)
        cache.reset()
        let out = cache.prepend(chunkMel: [5, 6, 7, 8], chunkFrames: 4)
        XCTAssertEqual(out, [0, 0, 5, 6, 7, 8])  // tail zeroed again
    }

    // MARK: - RNN-T greedy control flow (stub joint)

    func testGreedyDecodeBlankAdvancesFrames() throws {
        // Every frame returns blank → no tokens emitted.
        let blank = 13087
        let out = try RNNTDecoder.greedyDecode(
            frameCount: 5, blankID: blank, maxSymbolsPerFrame: 10,
            step: { _, _ in blank })
        XCTAssertEqual(out, [])
    }

    func testGreedyDecodeEmitsThenAdvances() throws {
        let blank = 13087
        // Frame 0: emit 7 then blank. Frame 1: emit 8 then blank.
        let out = try RNNTDecoder.greedyDecode(
            frameCount: 2, blankID: blank, maxSymbolsPerFrame: 10
        ) { frame, token in
            switch (frame, token) {
            case (0, blank): return 7      // emit 7
            case (0, 7): return blank      // then advance
            case (1, 7): return 8          // emit 8 (token carried from prev emit)
            case (1, 8): return blank      // then advance
            default: return blank
            }
        }
        XCTAssertEqual(out, [7, 8])
    }

    func testGreedyDecodeCapsSymbolsPerFrame() throws {
        let blank = 13087
        // Always emits a non-blank → must be capped at maxSymbolsPerFrame per frame.
        var emittedCount = 0
        let out = try RNNTDecoder.greedyDecode(
            frameCount: 3, blankID: blank, maxSymbolsPerFrame: 4
        ) { _, _ in
            emittedCount += 1
            return 42  // never blank
        }
        // 3 frames × 4 symbols cap = 12 emissions.
        XCTAssertEqual(out.count, 12)
        XCTAssertTrue(out.allSatisfy { $0 == 42 })
    }
}
