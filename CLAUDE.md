# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

On-device, offline, streaming ASR PoC for iPhone/iPad using NVIDIA **Nemotron-3.5-ASR
Streaming 0.6B** (multilingual) via CoreML. The app is intentionally built to **run
without the model weights present** — it captures mic audio, resamples to 16 kHz,
chunks to the streaming tier, and reports benchmark metrics, showing a "models missing"
status until the ~634 MB CoreML bundle is downloaded. Real inference (Phases 5–7) is
not yet wired in; see the phase table in `README.md` for current state.

## Project generation & build

The Xcode project is **generated from `project.yml` via XcodeGen** — `*.xcodeproj` is
gitignored. Always run `xcodegen generate` after editing `project.yml`, after adding/
removing source files, or after downloading models (so `Models/` is re-bundled).

```bash
xcodegen generate                                    # regenerate the project
xcodebuild -scheme NemotronASRPoC \
  -destination 'generic/platform=iOS Simulator' build
xcodebuild test -scheme NemotronASRPoC \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro'
# single test:
xcodebuild test -scheme NemotronASRPoC \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' \
  -only-testing:NemotronASRPoCTests/AudioPipelineTests/testResamplerDownsamplesTo16k
```

The simulator builds/runs the full shell (audio + benchmark) but **cannot run real
inference** — a physical device (iPhone 15 Pro+ for ANE) is required for that.

## Model setup (separate from build)

```bash
./scripts/download_models.sh multilingual 2240       # → Models/multilingual/2240ms/
python3 scripts/inspect_model.py Models/multilingual/2240ms \
  --out NemotronASRPoC/ASR/ModelSignatures.json      # regenerate signatures if tier changes
xcodegen generate                                    # re-bundle Models/ into the app
```

`Models/**` is gitignored (license-bound weights). `Models/` is referenced in
`project.yml` as an **optional folder** (`type: folder`), so `.mlmodelc` bundles are
copied as resources when present and the app still builds when absent.

## Architecture

Audio pipeline (Phases 1–3, working today):

```
AVAudioEngine → AudioResampler (16 kHz mono) → AudioChunkBuffer (tier-aligned chunks)
                                                      │
                          RecordingController ←───────┘  (@MainActor)
                              drives BenchmarkLogger + ASRState (@Observable)
                              ↓
                   ContentView / TranscriptView / BenchmarkPanelView
```

- **`App/RecordingController.swift`** — the orchestrator. `process(chunk:)` is the
  single seam where Phases 5–7 plug in real inference (preprocessor → encoder(cache) →
  RNN-T decode → tokenizer). Today it only records per-chunk timing + a placeholder.
- **`ASR/ASRState.swift`** — `@Observable` `@MainActor` state bridging pipeline and UI
  (status enum, selected language, partial/final transcript, benchmark snapshot).
- **`Audio/`** — `AudioRecorder` (AVAudioEngine), `AudioResampler` (→16 kHz mono),
  `AudioChunkBuffer` (emits fixed-size chunks per `StreamingTier`: 560/1120/**2240**/4480 ms).
- **`Benchmark/`** — `LatencyTracker` (p50/p90/p99, RTF), `MemoryMonitor`, `BenchmarkLogger`.
- **`Support/ModelLocator.swift`** — locates/validates the bundle; reports a clear
  missing-reason for graceful degradation. Required modules: `preprocessor`, `encoder`,
  `decoder_joint` (fused default); `decoder`/`joint` are optional fallbacks.

### Signature-driven, never hardcoded

`scripts/inspect_model.py` reads the real CoreML model and emits
`NemotronASRPoC/ASR/ModelSignatures.json`. At runtime `ModelSignatures.swift` loads it
so **tensor names, shapes, the vocab/blank index, and the language→`prompt_id` map are
all data, never hardcoded** in Swift. When changing models/tiers, regenerate this JSON
rather than editing constants. Key facts it carries: 16 kHz mono, 128 mel features,
vocab 13,087 (blank_idx 13087), encoder caches are **explicit input/output tensors**
(`cache_channel`/`cache_time`/`cache_len`, not `MLState`) and must be threaded across
chunks. Languages without a dedicated prompt (e.g. Cantonese) fall back to `auto`.

## Reference docs

`IMPLEMENTATION-PLAN.md` (engineering plan + phase breakdown) and `proposed-plan.md`
(original product intent) are the source of truth for what each phase should deliver.
