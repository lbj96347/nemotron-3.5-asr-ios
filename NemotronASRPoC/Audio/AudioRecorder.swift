import AVFoundation

/// Captures microphone audio with `AVAudioEngine`, resamples to 16 kHz mono
/// Float32, and streams chunks aligned to the selected `StreamingTier`.
///
/// Callbacks are invoked on an internal serial queue, NOT the main thread.
/// The owner is responsible for hopping to `@MainActor` for UI updates.
final class AudioRecorder {
    enum RecorderError: Error, CustomStringConvertible {
        case permissionDenied
        case sessionConfigFailed(Error)
        case resamplerUnavailable
        case engineStartFailed(Error)

        var description: String {
            switch self {
            case .permissionDenied:        return "Microphone permission denied"
            case .sessionConfigFailed(let e): return "Audio session config failed: \(e.localizedDescription)"
            case .resamplerUnavailable:    return "Could not build 16 kHz resampler"
            case .engineStartFailed(let e): return "Audio engine failed to start: \(e.localizedDescription)"
            }
        }
    }

    private let engine = AVAudioEngine()
    private let chunkBuffer: AudioChunkBuffer
    private var resampler: AudioResampler?
    private let processingQueue = DispatchQueue(label: "ai.tokkong.nemotron.audio")

    /// Emitted for every complete tier-aligned chunk of 16 kHz mono samples.
    var onChunk: (([Float]) -> Void)?
    /// Total seconds captured (16 kHz) since `start`.
    private(set) var capturedSeconds: Double = 0

    init(tier: StreamingTier = .recommended) {
        self.chunkBuffer = AudioChunkBuffer(tier: tier)
    }

    func setTier(_ tier: StreamingTier) {
        processingQueue.sync { chunkBuffer.setTier(tier) }
    }

    /// Request microphone permission (async wrapper over the AVAudioApplication API).
    static func requestPermission() async -> Bool {
        await withCheckedContinuation { continuation in
            AVAudioApplication.requestRecordPermission { granted in
                continuation.resume(returning: granted)
            }
        }
    }

    func start() throws {
        let session = AVAudioSession.sharedInstance()
        do {
            try session.setCategory(.record, mode: .measurement, options: [.duckOthers])
            try session.setActive(true, options: .notifyOthersOnDeactivation)
        } catch {
            throw RecorderError.sessionConfigFailed(error)
        }

        let input = engine.inputNode
        let inputFormat = input.outputFormat(forBus: 0)

        guard let resampler = AudioResampler(inputFormat: inputFormat) else {
            throw RecorderError.resamplerUnavailable
        }
        self.resampler = resampler

        capturedSeconds = 0
        processingQueue.sync { chunkBuffer.reset() }

        input.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { [weak self] buffer, _ in
            self?.processingQueue.async {
                self?.handle(buffer: buffer)
            }
        }

        engine.prepare()
        do {
            try engine.start()
        } catch {
            input.removeTap(onBus: 0)
            throw RecorderError.engineStartFailed(error)
        }
    }

    /// Stop capture. Returns the final zero-padded chunk if any audio remained.
    @discardableResult
    func stop() -> [Float]? {
        engine.inputNode.removeTap(onBus: 0)
        if engine.isRunning { engine.stop() }
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)

        var tail: [Float]?
        processingQueue.sync {
            tail = chunkBuffer.flushFinal()
        }
        return tail
    }

    private func handle(buffer: AVAudioPCMBuffer) {
        guard let resampler else { return }
        let samples = resampler.resample(buffer)
        guard !samples.isEmpty else { return }
        capturedSeconds += Double(samples.count) / AudioResampler.targetSampleRate

        for chunk in chunkBuffer.append(samples) {
            onChunk?(chunk)
        }
    }
}
