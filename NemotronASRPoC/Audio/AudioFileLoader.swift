import AVFoundation
import Foundation

/// Loads an audio file (e.g. `.m4a`/AAC) and returns the first N seconds as
/// 16 kHz mono Float32 samples, reusing `AudioResampler`.
///
/// AVFoundation decodes AAC natively; if it ever fails to open a file an
/// `afconvert` fallback could be added, but the meeting clip is standard AAC.
enum AudioFileLoader {

    enum LoadError: Error, CustomStringConvertible {
        case cannotOpen(URL, Error)
        case unsupportedFormat
        case readFailed(Error)
        var description: String {
            switch self {
            case .cannotOpen(let u, let e): return "cannot open \(u.lastPathComponent): \(e)"
            case .unsupportedFormat: return "unsupported audio format"
            case .readFailed(let e): return "audio read failed: \(e)"
            }
        }
    }

    /// Read up to `maxSeconds` of audio and return 16 kHz mono samples.
    /// Pass `maxSeconds = nil` to read the whole file.
    static func load16kMono(url: URL, maxSeconds: Double?) throws -> [Float] {
        let file: AVAudioFile
        do {
            file = try AVAudioFile(forReading: url)
        } catch {
            throw LoadError.cannotOpen(url, error)
        }

        let format = file.processingFormat
        let sourceRate = format.sampleRate
        guard sourceRate > 0 else { throw LoadError.unsupportedFormat }

        guard let resampler = AudioResampler(inputFormat: format) else {
            throw LoadError.unsupportedFormat
        }

        // Total frames to read (clamped to maxSeconds in the source's rate).
        let totalFrames = file.length
        let limitFrames: AVAudioFramePosition
        if let maxSeconds {
            limitFrames = min(totalFrames, AVAudioFramePosition(maxSeconds * sourceRate))
        } else {
            limitFrames = totalFrames
        }

        // Read in blocks so we never hold the whole file in one PCM buffer.
        let blockFrames: AVAudioFrameCount = 1 << 15   // 32k frames per read
        var samples: [Float] = []
        samples.reserveCapacity(Int(Double(limitFrames) * AudioResampler.targetSampleRate / sourceRate) + 1024)

        var remaining = limitFrames
        while remaining > 0 {
            let toRead = AVAudioFrameCount(min(AVAudioFramePosition(blockFrames), remaining))
            guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: toRead) else {
                throw LoadError.unsupportedFormat
            }
            do {
                try file.read(into: buffer, frameCount: toRead)
            } catch {
                throw LoadError.readFailed(error)
            }
            if buffer.frameLength == 0 { break }  // EOF
            samples.append(contentsOf: resampler.resample(buffer))
            remaining -= AVAudioFramePosition(buffer.frameLength)
            if buffer.frameLength < toRead { break }  // short read = EOF
        }
        return samples
    }
}
