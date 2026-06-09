import AVFoundation

/// Converts arbitrary hardware-format PCM buffers into 16 kHz mono Float32.
/// Nemotron expects 16 kHz mono; the mic hardware is typically 44.1/48 kHz.
final class AudioResampler {
    static let targetSampleRate: Double = 16_000

    private let converter: AVAudioConverter
    let outputFormat: AVAudioFormat

    /// Fails if the source format is incompatible or the target format can't be built.
    init?(inputFormat: AVAudioFormat) {
        guard let outFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: AudioResampler.targetSampleRate,
            channels: 1,
            interleaved: false
        ) else { return nil }

        guard let converter = AVAudioConverter(from: inputFormat, to: outFormat) else { return nil }
        self.converter = converter
        self.outputFormat = outFormat
    }

    /// Resample one input buffer to 16 kHz mono Float32 samples.
    /// Returns an empty array if the converter produced no frames this call.
    func resample(_ inputBuffer: AVAudioPCMBuffer) -> [Float] {
        // Estimate output capacity from the sample-rate ratio (+ slack).
        let ratio = outputFormat.sampleRate / inputBuffer.format.sampleRate
        let capacity = AVAudioFrameCount(Double(inputBuffer.frameLength) * ratio + 1024)
        guard capacity > 0,
              let outputBuffer = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: capacity) else {
            return []
        }

        var fed = false
        var error: NSError?
        let status = converter.convert(to: outputBuffer, error: &error) { _, inputStatus in
            if fed {
                inputStatus.pointee = .noDataNow
                return nil
            }
            fed = true
            inputStatus.pointee = .haveData
            return inputBuffer
        }

        guard status != .error, error == nil,
              let channel = outputBuffer.floatChannelData else {
            return []
        }

        let frames = Int(outputBuffer.frameLength)
        return Array(UnsafeBufferPointer(start: channel[0], count: frames))
    }
}
