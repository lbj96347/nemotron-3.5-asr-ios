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

    /// Live streaming pipeline (Phase 7). `transcriber` owns the loaded CoreML
    /// modules; mic chunks are fed into `chunkContinuation` and drained, strictly
    /// in order, by the single `streamTask` consumer loop.
    private var transcriber: StreamingTranscriber?
    private var streamTask: Task<Void, Never>?
    private var chunkContinuation: AsyncStream<[Float]>.Continuation?

    /// Re-entrancy guard for `start()`. The Start button is disabled during a
    /// session, but the `await requestPermission()` gap leaves a brief window where
    /// a double tap can kick off a SECOND `StreamingTranscriber.make()` — loading a
    /// second ~634 MB model and leaking the first. Set synchronously on entry.
    private var isStarting = false

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
        guard !isStarting else {
            Log.stream.warning("start() re-entry ignored — a session is already starting")
            return
        }
        isStarting = true
        defer { isStarting = false }
        Log.stream.info("start(): language=\(self.state.selectedLanguage.rawValue, privacy: .public), tier=\(self.tier.displayName, privacy: .public)")

        state.status = .requestingPermission
        let granted = await AudioRecorder.requestPermission()
        guard granted else {
            state.status = .permissionDenied
            return
        }

        benchmark.reset()
        chunkCount = 0
        state.clearTranscripts()

        // Load the streaming pipeline off the main actor. If the model bundle is
        // absent we keep recording audio + benchmarks but emit no transcript
        // (graceful degradation) — exactly the pre-model demo behavior.
        let language = state.selectedLanguage
        let tier = tier
        if case .available = ModelLocator(variant: "multilingual", tier: tier).locate() {
            state.status = .loadingModels
            do {
                let loaded = try await Task.detached(priority: .userInitiated) {
                    try StreamingTranscriber.make(tier: tier, language: language)
                }.value
                transcriber = loaded
                benchmark.recordModelLoad(seconds: loaded.loadSeconds)
            } catch {
                state.status = .error(String(describing: error))
                return
            }
        } else {
            transcriber = nil
        }

        // FIFO chunk stream: the audio queue yields, a single consumer drains in
        // order so no two chunks overlap (queue-all, never drop).
        let (stream, continuation) = AsyncStream<[Float]>.makeStream()
        chunkContinuation = continuation

        let recorder = AudioRecorder(tier: tier)
        recorder.onChunk = { [weak self] chunk in
            // Cheap, thread-safe yield from the audio queue — no main hop, no work.
            self?.chunkContinuation?.yield(chunk)
        }
        self.recorder = recorder

        streamTask = Task { [weak self] in
            for await chunk in stream {
                await self?.consume(chunk)
            }
        }

        do {
            try recorder.start()
            benchmark.markRecordingStart(at: CACurrentMediaTime())
            state.status = .recording
            startSampling()
        } catch {
            chunkContinuation?.finish()
            streamTask?.cancel()
            state.status = .error(String(describing: error))
        }
    }

    func stop() async {
        state.status = .finishing
        let tail = recorder?.stop()
        recorder = nil
        stopSampling()

        // Stop accepting new chunks, then drain everything already queued so the
        // transcript is complete before we finalize (queue-all, never drop).
        chunkContinuation?.finish()
        chunkContinuation = nil
        await streamTask?.value
        streamTask = nil

        // Final zero-padded tail chunk, if any, goes through the same path.
        if let tail { await consume(tail) }

        if let transcriber {
            state.finalTranscript = await transcriber.finalText()
            state.partialTranscript = ""
        } else {
            state.commitPartial()
        }
        transcriber = nil

        refreshAvailabilityAfterStop()
        updateSnapshot()
    }

    func clear() {
        state.clearTranscripts()
        benchmark.reset()
        chunkCount = 0
        if state.status != .recording { transcriber = nil }
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

    /// Drive one mic chunk through the streaming pipeline. Runs on the main actor
    /// for state mutation + benchmarking, but the CoreML work inside
    /// `transcriber.ingest` is `await`ed on the transcriber actor (off-main).
    /// Called serially by the single FIFO consumer loop in `start()`.
    private func consume(_ chunk: [Float]) async {
        let audioSeconds = Double(chunk.count) / AudioResampler.targetSampleRate

        guard let transcriber else {
            // No model bundled: record timing only so the benchmark panel + audio
            // pipeline still demo without a transcript (graceful degradation).
            benchmark.recordChunk(inferenceSeconds: 0, audioSeconds: audioSeconds)
            chunkCount += 1
            updateSnapshot()
            return
        }

        let start = CACurrentMediaTime()
        do {
            let text = try await transcriber.ingest(chunk)
            let seconds = CACurrentMediaTime() - start
            benchmark.recordChunk(inferenceSeconds: seconds, audioSeconds: audioSeconds)
            chunkCount += 1
            benchmark.markFirstPartial(at: CACurrentMediaTime())
            state.partialTranscript = text
            updateSnapshot()
            let footprint = MemoryMonitor.physicalFootprintMB() ?? 0
            Log.stream.info("chunk #\(self.chunkCount) ingested in \(Int(seconds * 1000)) ms, samples=\(chunk.count), text.count=\(text.count), footprint=\(Int(footprint)) MB")
        } catch {
            Log.stream.error("chunk ingest FAILED: \(String(describing: error), privacy: .public)")
            state.status = .error(String(describing: error))
        }
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
