# Nemotron ASR iOS PoC

On-device, offline, streaming speech recognition PoC using NVIDIA
**Nemotron-3.5-ASR Streaming 0.6B** (multilingual) via CoreML on iPhone/iPad.

See [`IMPLEMENTATION-PLAN.md`](IMPLEMENTATION-PLAN.md) for the full engineering plan
and [`proposed-plan.md`](proposed-plan.md) for the original product intent.

## Requirements

- Xcode 16+ (built/tested with Xcode 26.3)
- iOS 17+
- iPhone 15 Pro or newer recommended (ANE); device required for real inference
- [XcodeGen](https://github.com/yonsm/XcodeGen) (`brew install xcodegen`)
- Microphone permission
- CoreML model files downloaded separately (see **Model Setup**)

## Build & Run

```bash
# 1. Generate the Xcode project from project.yml (run after any project.yml change)
xcodegen generate

# 2a. Build for the simulator (shell + audio + benchmark; no real inference)
xcodebuild -scheme NemotronASRPoC -destination 'generic/platform=iOS Simulator' build

# 2b. Or open in Xcode and run on a physical device
open NemotronASRPoC.xcodeproj

# 3. Run unit tests (resampler, chunk buffer, latency tracker)
xcodebuild test -scheme NemotronASRPoC -destination 'platform=iOS Simulator,name=iPhone 17 Pro'

# Run a single test
xcodebuild test -scheme NemotronASRPoC \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' \
  -only-testing:NemotronASRPoCTests/AudioPipelineTests/testResamplerDownsamplesTo16k
```

The app **runs without the model present** — it shows a "models not found" status
but still captures mic audio, resamples to 16 kHz, chunks to the streaming tier, and
reports live benchmark metrics. This lets you validate the shell before the download.

The file-transcription tests are gated on user-supplied clips in `TestAudio/` (the
folder is gitignored — provide your own). Name each clip with a **language hint** so the
suite picks the right prompt, e.g. `sample-en.m4a` (English) or `sample-yue.m4a`
(Cantonese → `auto`). Tests skip cleanly when no matching clip (or the model) is present.

## Model Setup

Default target: **`multilingual` @ 2240 ms** (~634 MB; covers EN / zh / ja / ko;
Cantonese has no dedicated prompt and falls back to `auto`).

```bash
# Download all five .mlmodelc bundles + tokenizer + metadata into Models/multilingual/2240ms/
./scripts/download_models.sh multilingual 2240

# Re-inspect to refresh signatures (already checked in, but re-run if you change tier)
python3 scripts/inspect_model.py Models/multilingual/2240ms --out NemotronASRPoC/ASR/ModelSignatures.json

# Re-bundle the Models/ folder into the app
xcodegen generate
```

Expected files in `Models/multilingual/2240ms/`:

```
preprocessor.mlmodelc   # audio[1,?] → mel[1,128,?], mel_length
encoder.mlmodelc        # mel + caches + prompt_id → encoded + updated caches (stateful)
decoder_joint.mlmodelc  # B1 fused decoder⊕joint — default RNN-T step
decoder.mlmodelc        # fallback (unfused)
joint.mlmodelc          # fallback (unfused)
tokenizer.json          # 13,087-token multilingual vocab
metadata.json           # prompt dictionary, cache shapes, vocab/blank
```

Source: <https://huggingface.co/FluidInference/Nemotron-3.5-ASR-Streaming-Multilingual-0.6b-CoreML>

### Verified model signatures (Phase 4)

`NemotronASRPoC/ASR/ModelSignatures.json` is generated from the real model and
consumed at runtime so no tensor names / prompt IDs are hardcoded. Highlights:

- **Audio:** 16 kHz mono; preprocessor takes `audio`[1,?] (1–1,280,000 samples).
- **Mel:** 128 features, `total_mel_frames` = 233 (224 chunk + 9 pre-encode cache).
- **Encoder state (explicit tensors, not `MLState`):** `cache_channel`[1,24,42,1024],
  `cache_time`[1,24,1024,8], `cache_len`[1] — passed in and returned as `*_out`.
- **Encoder output:** `encoded`[1,1024,28] (≈28 frames per 2.24 s chunk) + `prompt_id`.
- **RNN-T step (`decoder_joint`):** LSTM state `h_in`/`c_in`[2,1,640], `token`[1,1],
  `encoder`[1,1024,1] → `logits`[1,1,1,13088], `h_out`/`c_out`.
- **Vocab:** 13,087 tokens, **blank_idx = 13087**.
- **Prompt IDs:** en-US=0, zh-CN=4, zh-TW=5, ja-JP=10, ko-KR=14, default(auto)=101.

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | XcodeGen scaffold + SwiftUI shell (Start/Stop/Clear, language + tier pickers, transcript + benchmark panel) | ✅ Done |
| 2 | Mic capture → 16 kHz mono → tier-aligned chunk buffer | ✅ Done |
| 3 | Benchmark instrumentation (load time, latency p50/p90/p99, RTF, peak memory, thermal) | ✅ Done |
| 4 | Model download + CoreML signature inspection (`scripts/`, `ModelSignatures.json`) | ✅ Done |
| 5 | CoreML loading + adaptive runner (`CoreMLModelRunner`, compute-unit fallback) | ✅ Done |
| 6 | RNN-T greedy decoder + tokenizer + file transcription (`NemotronASRService`) | ✅ Done |
| 7 | Live mic streaming integration | ⬜ Next |
| 8 | Accuracy test set + Whisper baseline + benchmark results | ⬜ |

End-to-end file transcription works: `EndToEndTranscriptionTests.testTranscribeMeetingClip`
transcribes the first 60 s of the meeting clip on the iOS Simulator (CPU/GPU) at
**RTF ≈ 0.14**. Run it with:

```bash
xcodebuild test -scheme NemotronASRPoC \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' \
  -only-testing:NemotronASRPoCTests/EndToEndTranscriptionTests/testTranscribeMeetingClip
```

## Architecture (current)

```
AVAudioEngine ─▶ AudioResampler (16 kHz mono) ─▶ AudioChunkBuffer (tier-aligned)
        │                                                  │
        └──────────────▶ RecordingController ◀─────────────┘
                              │   (drives BenchmarkLogger + ASRState)
                              ▼
                   SwiftUI: ContentView / TranscriptView / BenchmarkPanelView
```

`RecordingController.process(chunk:)` is the seam where Phases 5–7 plug in real
inference (preprocessor → encoder(cache) → RNN-T decode → tokenizer). Today it
records per-chunk timing and shows a placeholder partial.

## Inference recipe (Phases 5–6, implemented)

`NemotronASRService.transcribeFile(url:language:maxSeconds:)` runs the full path:

1. Decode the file → 16 kHz mono (`AudioFileLoader`), trimmed to `maxSeconds`.
2. Run the **preprocessor once on the whole clip** → a continuous mel `[128, N]`.
   (The preprocessor accepts up to ~80 s; a centered STFT yields `1 + samples/hop`
   frames, e.g. 225 for a 2240 ms chunk — the chunk stride is `chunk_mel_frames`=224.)
3. Slice the mel into fixed **233-frame encoder windows** (`pre_encode_cache`=9 frames
   of real left context + 224 new), advancing by 224. Zeros precede the start;
   the final window is zero-padded and reports a smaller `mel_length`.
4. `NemotronEncoderRunner` threads the three encoder caches (`cache_*` → `cache_*_out`)
   across windows for the whole file (never reset within a file); `prompt_id`
   conditions the encoder only.
5. `RNNTDecoder` greedy-decodes each window's frames, **persisting predictor state
   (token + LSTM h/c) across windows** (reset only per utterance), capped at 10
   symbols/frame.
6. `Tokenizer` detokenizes (SentencePiece `▁`→space; CJK needs no special case).

## Known issues / open items

- `prompt_id` map + CoreML tensor names are **loaded at runtime** from
  `ModelSignatures.json` (generated by `inspect_model.py`), never hardcoded. ✅
- The CoreML log line `[espresso] … ios17.slice_by_index: zero shape error` during
  encoder load is **benign** flexible-shape type-inference noise — the module loads
  and produces correct `[1,1024,28]` output.
- Live mic path (`RecordingController.process(chunk:)`) still shows a placeholder —
  Phase 7 wires the same encoder/decoder/tokenizer onto the streaming chunks.
- Transcript **accuracy** isn't measured yet (no WER/CER scorer or Whisper baseline);
  that's Phase 8. The 60 s smoke test only asserts a sane, non-degenerate transcript.
- Cantonese (`yue-Hant-HK`) coverage depends on the multilingual vocab; keep Whisper
  as a fallback per the plan if CER is weak.
- ANE behavior and the memory ceiling can only be validated on device (the Simulator
  runs CoreML on CPU/GPU — correctness-faithful but not representative of ANE perf).

## License

Source code is released under the [MIT License](LICENSE). This covers the app code only —
the NVIDIA Nemotron-3.5-ASR model weights are **not** included and remain subject to their
own license on [Hugging Face](https://huggingface.co/FluidInference/Nemotron-3.5-ASR-Streaming-Multilingual-0.6b-CoreML).

## Built with

Development of this PoC was assisted by these tools:

- **[WhisKey](https://whiskey.asktobuild.app)** — quick on-device dictation of notes,
  commit messages, and issue descriptions.
- **[TokKong](https://tokkong.forthrighttech.com)** — offline transcription and
  translation of reference material (NVIDIA / CoreML docs, model cards) during development.
- **[Lounge](https://lounge.asktobuild.app)** — surfaced long-running build, test, and
  agent jobs on the desktop.
- **[NotifyMe](https://github.com/lbj96347/notifyme)** — pushed alerts to phone when
  long transcription test runs and model downloads finished.
