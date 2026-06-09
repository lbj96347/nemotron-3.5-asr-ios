import Foundation
import QuartzCore

/// Coordinates the audio pipeline + benchmark instrumentation and drives `ASRState`.
///
/// Phase 1–3 scope: this captures mic audio, resamples to 16 kHz, chunks it to the
/// selected tier, and reports live benchmark metrics. Actual Nemotron inference
/// (Phases 5–7) plugs into `process(chunk:)` later — for now that hook records a
/// per-chunk timing and shows a placeholder so the full pipeline is demonstrable
/// before the model is downloaded.
@MainActor
final class RecordingController {
    let state: ASRState
    private let benchmark = BenchmarkLogger()
    private var recorder: AudioRecorder?
    private var tier: StreamingTier = .recommended
    private var sampleTimer: Timer?
    private var chunkCount = 0
    private var transcribeTask: Task<Void, Never>?

    /// Bundled CoreML signatures (Phase 4 output) + resolved language→prompt_id map.
    /// Loaded independently of the weights so the prompt map is available even
    /// before the ~634 MB model download is bundled.
    private(set) var signatures: ModelSignatures?
    private(set) var promptMap = LanguagePromptMap()

    init(state: ASRState) {
        self.state = state
        loadSignatures()
    }

    private func loadSignatures() {
        switch ModelSignatures.loadFromBundle() {
        case .success(let sig):
            signatures = sig
            promptMap = sig.languagePromptMap()
            Log.model.info("Loaded ModelSignatures: vocab=\(sig.vocabSize ?? -1), blank=\(sig.blankTokenID ?? -1)")
        case .failure(let error):
            Log.model.warning("ModelSignatures unavailable: \(String(describing: error))")
        }
    }

    /// Resolved prompt_id for the current language (nil if no map loaded). Falls
    /// back to the model's default (`auto`) for languages without a dedicated
    /// prompt — e.g. Cantonese, which has no `yue` entry in the dictionary.
    var currentPromptID: Int? {
        promptMap.promptID(for: state.selectedLanguage)
    }

    /// Check model availability and reflect it in the UI. Called on appear.
    func refreshAvailability() {
        let locator = ModelLocator(variant: "multilingual", tier: tier)
        switch locator.locate() {
        case .available:
            // Models present — Phase 5 will load them here. For now mark ready.
            state.status = .ready
        case .missing(let reason):
            // Graceful degradation: the shell + audio + benchmark still work.
            state.status = .modelsMissing(reason: reason)
        }
    }

    func setTier(_ tier: StreamingTier) {
        self.tier = tier
        recorder?.setTier(tier)
    }

    func setLanguage(_ language: ASRLanguage) {
        state.selectedLanguage = language
    }

    func start() async {
        state.status = .requestingPermission
        let granted = await AudioRecorder.requestPermission()
        guard granted else {
            state.status = .permissionDenied
            return
        }

        benchmark.reset()
        chunkCount = 0
        let recorder = AudioRecorder(tier: tier)
        recorder.onChunk = { [weak self] chunk in
            // Hops to main actor; chunk arrives on the audio queue.
            Task { @MainActor [weak self] in self?.process(chunk: chunk) }
        }
        self.recorder = recorder

        do {
            try recorder.start()
            benchmark.markRecordingStart(at: CACurrentMediaTime())
            state.status = .recording
            startSampling()
        } catch {
            state.status = .error(String(describing: error))
        }
    }

    func stop() {
        state.status = .finishing
        let tail = recorder?.stop()
        if let tail { process(chunk: tail) }
        recorder = nil
        stopSampling()
        state.commitPartial()
        refreshAvailabilityAfterStop()
        updateSnapshot()
    }

    func clear() {
        state.clearTranscripts()
        benchmark.reset()
        chunkCount = 0
        updateSnapshot()
    }

    // MARK: - Imported file transcription

    /// Transcribe an audio file picked via the Files importer through the full
    /// offline Nemotron pipeline (`NemotronASRService`). Runs off the main actor
    /// and reports per-segment progress; cancellable via `cancelTranscription()`.
    func transcribeImportedFile(url: URL) {
        guard transcribeTask == nil else { return }
        state.clearTranscripts()
        benchmark.reset()
        chunkCount = 0
        state.status = .transcribingFile(progress: "")

        let language = state.selectedLanguage
        let tier = tier

        transcribeTask = Task { [weak self] in
            // Heavy CoreML work runs off the main actor on a background task.
            let outcome = await Task.detached(priority: .userInitiated) { () -> Result<NemotronASRService.Result, Error> in
                let scoped = url.startAccessingSecurityScopedResource()
                defer { if scoped { url.stopAccessingSecurityScopedResource() } }
                do {
                    let service = NemotronASRService(tier: tier)
                    let result = try service.transcribeFile(
                        url: url,
                        language: language,
                        maxSeconds: nil,
                        onProgress: { seg, total in
                            Task { @MainActor [weak self] in
                                self?.state.status = .transcribingFile(progress: "segment \(seg)/\(total)")
                            }
                        },
                        isCancelled: { Task.isCancelled })
                    return .success(result)
                } catch {
                    return .failure(error)
                }
            }.value

            await MainActor.run { [weak self] in
                self?.finishImport(outcome)
            }
        }
    }

    func cancelTranscription() {
        transcribeTask?.cancel()
    }

    private func finishImport(_ outcome: Result<NemotronASRService.Result, Error>) {
        transcribeTask = nil
        switch outcome {
        case .success(let result):
            state.finalTranscript = result.transcript
            benchmark.sampleMemory()
            var snapshot = benchmark.snapshot(now: CACurrentMediaTime())
            snapshot.recordingSeconds = result.audioSeconds
            snapshot.realTimeFactor = result.realTimeFactor
            snapshot.averageChunkMS = result.inferenceSeconds / Double(max(1, result.chunkCount)) * 1000
            state.benchmark = snapshot
            refreshAvailabilityAfterStop()
        case .failure(let error):
            if error is CancellationError {
                refreshAvailabilityAfterStop()
            } else {
                state.status = .error(String(describing: error))
            }
        }
    }

    // MARK: - Per-chunk processing

    private func process(chunk: [Float]) {
        let audioSeconds = Double(chunk.count) / AudioResampler.targetSampleRate
        let (_, seconds) = LatencyTracker.measure {
            // PLACEHOLDER until Phase 5–7: real inference goes here
            // (preprocessor → encoder(cache) → RNN-T decode → tokenizer).
            _ = chunk.count
        }
        benchmark.recordChunk(inferenceSeconds: seconds, audioSeconds: audioSeconds)
        chunkCount += 1
        benchmark.markFirstPartial(at: CACurrentMediaTime())

        // Visible placeholder so the streaming UI is demonstrable pre-model.
        let prompt = currentPromptID.map(String.init) ?? "—"
        state.partialTranscript =
            "[chunk \(chunkCount): \(String(format: "%.2f", audioSeconds))s @ 16kHz · "
            + "\(state.selectedLanguage.rawValue) prompt_id=\(prompt) — awaiting model inference]"
        updateSnapshot()
    }

    // MARK: - Sampling loop

    private func startSampling() {
        sampleTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.benchmark.sampleMemory()
                self?.updateSnapshot()
            }
        }
    }

    private func stopSampling() {
        sampleTimer?.invalidate()
        sampleTimer = nil
    }

    private func updateSnapshot() {
        benchmark.sampleMemory()
        state.benchmark = benchmark.snapshot(now: CACurrentMediaTime())
    }

    private func refreshAvailabilityAfterStop() {
        let locator = ModelLocator(variant: "multilingual", tier: tier)
        if case .missing(let reason) = locator.locate() {
            state.status = .modelsMissing(reason: reason)
        } else {
            state.status = .ready
        }
    }
}
