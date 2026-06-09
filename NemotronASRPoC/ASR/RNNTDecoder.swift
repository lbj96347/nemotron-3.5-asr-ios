import CoreML
import Foundation

/// Greedy RNN-T decoder driving the fused `decoder_joint` module.
///
/// Recipe (language-agnostic — `prompt_id` already conditioned the encoder):
/// - initial token = blank (13087); LSTM state h/c = zeros `[2, 1, 640]`.
/// - for each encoder frame: run `decoder_joint(encoder[1,1024,1], token[1,1],
///   token_length=[1], h_in, c_in)` → `logits[1,1,1,13088]`; argmax.
///   - blank → advance to the next encoder frame (do not update token/state).
///   - non-blank → emit the token, set it as the next input token, carry
///     `h_out`/`c_out`, and re-run on the SAME frame.
/// - cap the inner loop at `maxSymbolsPerFrame` (10) to bound runaway emission.
final class RNNTDecoder {
    private let runner: CoreMLModelRunner
    private let blankID: Int
    private let decoderHidden: Int     // 640
    private let decoderLayers: Int     // 2
    let maxSymbolsPerFrame: Int

    // Persistent streaming state — carried ACROSS encoder chunks for the whole
    // utterance (RNN-T predictor is recurrent). Reset only between files.
    private var token: Int
    private var h: MLMultiArray
    private var c: MLMultiArray

    init(runner: CoreMLModelRunner, maxSymbolsPerFrame: Int = 10) throws {
        self.runner = runner
        let blank = runner.signatures.blankTokenID ?? 13087
        self.blankID = blank
        self.decoderHidden = runner.signatures.metadata?.decoder_hidden ?? 640
        self.decoderLayers = runner.signatures.metadata?.decoder_layers ?? 2
        self.maxSymbolsPerFrame = maxSymbolsPerFrame
        self.token = blank
        self.h = try CoreMLModelRunner.zeros(shape: [decoderLayers, 1, decoderHidden])
        self.c = try CoreMLModelRunner.zeros(shape: [decoderLayers, 1, decoderHidden])
    }

    /// Reset predictor state to start a new utterance (token = blank, h/c zero).
    func reset() throws {
        token = blankID
        h = try zeroState()
        c = try zeroState()
    }

    /// Abstracts a single `decoder_joint` step so the loop can be unit-tested
    /// against a stub. Returns the argmax token id and the next LSTM state.
    struct Step {
        let token: Int
        let h: MLMultiArray
        let c: MLMultiArray
    }

    private var stateShape: [Int] { [decoderLayers, 1, decoderHidden] }

    private func zeroState() throws -> MLMultiArray {
        try CoreMLModelRunner.zeros(shape: stateShape)
    }

    /// Greedy-decode one chunk's encoder frames, CONTINUING from the persistent
    /// predictor state (token/h/c) and updating it in place. Call `reset()`
    /// before a new utterance. Returns the token ids emitted for this chunk.
    func decode(encoderFrames: [[Float]]) throws -> [Int] {
        // The pure control flow drives `token` (the current predictor input);
        // we keep LSTM h/c in instance state and commit the candidate next-state
        // only on a non-blank emission (blank steps don't advance the predictor).
        var pendingH = h
        var pendingC = c

        let emitted = try Self.greedyDecode(
            frameCount: encoderFrames.count,
            blankID: blankID,
            maxSymbolsPerFrame: maxSymbolsPerFrame,
            initialToken: token
        ) { frameIndex, currentToken in
            let step = try runStep(encoderFrame: encoderFrames[frameIndex], token: currentToken, h: h, c: c)
            pendingH = step.h
            pendingC = step.c
            return step.token
        } onEmit: { emittedToken in
            self.token = emittedToken
            self.h = pendingH
            self.c = pendingC
        }
        return emitted
    }

    /// Pure greedy RNN-T control flow, independent of CoreML / MLMultiArray so it
    /// can be unit-tested with a stub `step`. For each frame, repeatedly query
    /// `step(frameIndex, currentToken)`; on blank → advance to the next frame; on
    /// a real token → emit it, make it the current token, call `onEmit` (to let
    /// the caller commit LSTM state), and re-query the same frame, capped at
    /// `maxSymbolsPerFrame`.
    static func greedyDecode(
        frameCount: Int,
        blankID: Int,
        maxSymbolsPerFrame: Int,
        initialToken: Int? = nil,
        step: (_ frameIndex: Int, _ token: Int) throws -> Int,
        onEmit: (_ emittedToken: Int) -> Void = { _ in }
    ) rethrows -> [Int] {
        var emitted: [Int] = []
        var token = initialToken ?? blankID
        for frame in 0..<frameCount {
            var symbols = 0
            while symbols < maxSymbolsPerFrame {
                let next = try step(frame, token)
                if next == blankID { break }  // advance to next frame; keep token
                emitted.append(next)
                token = next
                onEmit(next)
                symbols += 1
            }
        }
        return emitted
    }

    private func runStep(encoderFrame: [Float], token: Int, h: MLMultiArray, c: MLMultiArray) throws -> Step {
        let encDim = runner.signatures.metadata?.encoder_dim ?? 1024
        let encoderArray = try CoreMLModelRunner.floatArray(encoderFrame, shape: [1, encDim, 1])
        let tokenArray = try CoreMLModelRunner.int32Array([Int32(token)], shape: [1, 1])
        let tokenLen = try CoreMLModelRunner.int32Array([1], shape: [1])

        let inputs: [String: MLFeatureValue] = [
            "encoder": MLFeatureValue(multiArray: encoderArray),
            "token": MLFeatureValue(multiArray: tokenArray),
            "token_length": MLFeatureValue(multiArray: tokenLen),
            "h_in": MLFeatureValue(multiArray: h),
            "c_in": MLFeatureValue(multiArray: c),
        ]
        let out = try runner.predict(module: "decoder_joint", inputs: inputs)

        guard let logits = out.featureValue(for: "logits")?.multiArrayValue else {
            throw CoreMLModelRunner.RunnerError.missingOutput(module: "decoder_joint", name: "logits")
        }
        let best = Self.argmax(logits)
        let hOut = out.featureValue(for: "h_out")?.multiArrayValue ?? h
        let cOut = out.featureValue(for: "c_out")?.multiArrayValue ?? c
        return Step(token: best, h: hOut, c: cOut)
    }

    /// Argmax over a flat `logits[1,1,1,V]` array.
    static func argmax(_ logits: MLMultiArray) -> Int {
        let count = logits.count
        guard count > 0 else { return 0 }
        let ptr = logits.dataPointer.bindMemory(to: Float.self, capacity: count)
        var bestIdx = 0
        var bestVal = ptr[0]
        for i in 1..<count {
            if ptr[i] > bestVal { bestVal = ptr[i]; bestIdx = i }
        }
        return bestIdx
    }
}
