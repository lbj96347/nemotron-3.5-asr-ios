# Implementation Plan: Nemotron-3.5-ASR Streaming 0.6B on iOS (PoC)

> Supersedes the high-level `proposed-plan.md` with a concrete, build-ready plan
> grounded in the **actual** model artifacts and the toolchain available on this
> machine. Read `proposed-plan.md` for the original product intent; this file is
> the engineering plan Claude Code will execute.

---

## 0. Findings that change the original plan

These were verified against the model card and the local environment, and they
correct several assumptions in `proposed-plan.md`:

1. **The model is 5 CoreML modules per tier, not 3.** Each tier folder contains:
   - `preprocessor.mlmodelc` — log-mel / feature extraction (the plan omitted this)
   - `encoder.mlmodelc` — Conformer, mixed INT8 / 6-bit palettized
   - `decoder.mlmodelc` — RNN-T prediction network
   - `joint.mlmodelc` — joint network
   - `decoder_joint.mlmodelc` — **B1 fused decoder⊕joint (this is the default path)**
   - `metadata.json`, `tokenizer.json`
   - A top-level `manifest.json` indexes `latin` + `multilingual` × tiers `{560, 1120, 2240, 4480}` ms.

2. **Chunk size is a tier, not 320 ms.** Streaming chunks are `{560, 1120, 2240, 4480}` ms.
   **2240 ms (2.24 s) is the recommended default.** The original "320 ms chunk" is wrong for this model.

3. **Two vocab variants.** `latin` = 2,828-token pruned vocab (en/es/fr/it/pt/de);
   `multilingual` = 13,087-token full vocab (everything else, incl. zh/ja/ko).
   **For TokKong's CJK + Cantonese targets we use `multilingual`.**

4. **Decoder is RNN-T / TDT, not Whisper-style.** Greedy RNN-T loop with a blank token
   and per-step joint evaluation; this is the core custom Swift work.

5. **FluidAudio SPM (0.12.4) does NOT expose Nemotron** — only Parakeet TDT/EOU.
   So we **cannot** just call a library; we implement the runner ourselves but borrow
   FluidAudio's CoreML/ANE patterns as reference. (Re-check newer FluidAudio tags during
   Phase 5 in case Nemotron lands in the public API — if so, we can swap our runner for it.)

6. **`prompt_id` language mapping is not enumerated on the card.** Must be read from
   `metadata.json` / `tokenizer.json` at runtime. **Do not hardcode prompt IDs** — load them.

7. **Toolchain is ready:** Xcode 26.3, Swift 6.2, `xcodegen`, `coremlcompiler`, Python 3.13.
   We use **XcodeGen** (`project.yml`) so the `.xcodeproj` is reproducible and reviewable.

---

## 1. Target architecture

```
AVAudioEngine (mic, hardware SR)
   │  install tap, Float32
   ▼
AudioResampler (AVAudioConverter → 16 kHz mono Float32)
   ▼
AudioChunkBuffer (accumulate to tier window, e.g. 2240 ms, with overlap/lookahead)
   ▼
NemotronASRService  ──────────────────────────────────────────────┐
   │ 1. preprocessor.mlmodelc  → log-mel features                   │  stateful
   │ 2. encoder.mlmodelc(features, encoderCache) → enc, encoderCache│  caches kept
   │ 3. RNNTDecoder: loop decoder_joint.mlmodelc(enc_t, decState)   │  between chunks
   │      → token / blank, update decState                          │
   ▼                                                                │
Tokenizer (tokenizer.json) → text, CJK-aware joining ◄─────────────┘
   ▼
TranscriptState (@Observable) → SwiftUI partial + final
   ▼
BenchmarkLogger / MemoryMonitor / LatencyTracker → Benchmark panel
```

**Key principle (from the original plan, kept):** the CoreML input/output tensor
names, cache shapes, and `prompt_id` values are **discovered at runtime** from model
metadata, never hardcoded. `CoreMLModelRunner` introspects `MLModel.modelDescription`
and a generated `ModelSignatures.json` so the Swift decode loop adapts to the real model.

---

## 2. Repository layout (to be created)

```
nemotron-3.5-ios/
├── project.yml                      # XcodeGen spec (source of truth)
├── README.md
├── IMPLEMENTATION-PLAN.md           # this file
├── proposed-plan.md                 # original intent
├── .gitignore
├── scripts/
│   ├── download_models.sh           # pull multilingual/<tier> from HF
│   ├── inspect_model.py             # dump CoreML I/O names + shapes → ModelSignatures.json
│   └── compile_mlpackage.sh         # coremlcompiler .mlpackage → .mlmodelc (if needed)
├── Models/                          # .mlmodelc + tokenizer.json (gitignored, downloaded)
│   └── .gitkeep
├── Samples/                         # test wavs + reference transcripts (gitignored)
│   └── manifest.json
└── NemotronASRPoC/
    ├── App/
    │   ├── NemotronASRPoCApp.swift
    │   ├── ContentView.swift
    │   ├── TranscriptView.swift
    │   └── BenchmarkPanelView.swift
    ├── Audio/
    │   ├── AudioRecorder.swift
    │   ├── AudioResampler.swift
    │   └── AudioChunkBuffer.swift
    ├── ASR/
    │   ├── NemotronASRService.swift
    │   ├── CoreMLModelRunner.swift
    │   ├── ModelSignatures.swift     # decoded ModelSignatures.json
    │   ├── RNNTDecoder.swift
    │   ├── Tokenizer.swift
    │   ├── LanguagePrompt.swift
    │   └── ASRState.swift            # @Observable transcript + status
    ├── Benchmark/
    │   ├── BenchmarkLogger.swift
    │   ├── MemoryMonitor.swift
    │   └── LatencyTracker.swift
    ├── Support/
    │   ├── ModelLocator.swift        # find/validate Models/ at runtime
    │   └── Logging.swift
    └── Resources/
        ├── Info.plist                # NSMicrophoneUsageDescription
        └── Assets.xcassets
```

`Models/` and `Samples/` audio are **gitignored** (large, license-bound). The app must
**launch and degrade gracefully** when models are absent (clear "models not found" state),
so the shell is testable without the ~hundreds-of-MB download.

---

## 3. Build / run / test commands

```bash
# Generate the Xcode project from project.yml
xcodegen generate

# Download model (multilingual, 2240 ms default tier) into Models/
./scripts/download_models.sh multilingual 2240

# Inspect the real CoreML signatures → NemotronASRPoC/ASR/ModelSignatures.json
python3 scripts/inspect_model.py Models/multilingual/2240ms

# Build for simulator (smoke test the shell; ANE/real inference needs a device)
xcodebuild -scheme NemotronASRPoC -destination 'generic/platform=iOS Simulator' build

# Build + run on a connected device (real benchmark target)
xcodebuild -scheme NemotronASRPoC -destination 'platform=iOS,name=<device>' build

# Unit tests (resampler, chunk buffer, tokenizer, RNN-T decode on fixtures)
xcodebuild -scheme NemotronASRPoC -destination 'platform=iOS Simulator,name=iPhone 16 Pro' test

# Run a single test
xcodebuild test -scheme NemotronASRPoC \
  -destination 'platform=iOS Simulator,name=iPhone 16 Pro' \
  -only-testing:NemotronASRPoCTests/TokenizerTests/testCJKJoining
```

---

## 4. Implementation phases

Each phase ends in a **buildable, demonstrable** state. Implement in order.

### Phase 1 — Project scaffold + SwiftUI shell  *(no model needed)*
- `project.yml`: iOS 17 min, Swift 6, app target + test target, Info.plist with
  `NSMicrophoneUsageDescription`, `Models/` as a folder reference (not compiled).
- `.gitignore` (Models, Samples audio, `*.xcodeproj`, DerivedData, `.DS_Store`).
- SwiftUI: Start / Stop / Clear buttons, language `Picker`, partial + final transcript
  areas, benchmark panel. `ASRState` as `@Observable`.
- `git init` and an initial commit.
- **Done when:** `xcodegen generate` + simulator build succeeds; UI renders; buttons toggle state.

### Phase 2 — Audio capture + 16 kHz pipeline  *(no model needed)*
- `AudioRecorder`: `AVAudioSession` (`.record`), tap on input node, request mic permission.
- `AudioResampler`: `AVAudioConverter` from hardware SR → 16 kHz mono Float32.
- `AudioChunkBuffer`: accumulate to the selected tier window; emit chunks with the
  small lookahead/overlap the streaming encoder expects (configurable per tier).
- **Done when:** recording produces 16 kHz Float32 chunks of correct length; unit tests
  on resampler (e.g. 48k→16k length/energy) and chunk buffer pass.

### Phase 3 — Benchmark instrumentation  *(no model needed)*
- `MemoryMonitor` (`task_vm_info` / `phys_footprint`), `LatencyTracker` (p50/p90/p99, RTF),
  `BenchmarkLogger` (CSV export + on-screen panel: load time, first-partial latency,
  per-chunk latency, peak/avg memory, RTF, thermal state, duration).
- **Done when:** panel updates live with synthetic timings.

### Phase 4 — Model acquisition + signature inspection  *(needs model)*
- `scripts/download_models.sh`: fetch `multilingual/<tier>ms/*` + `tokenizer.json` +
  `metadata.json` from Hugging Face into `Models/`.
- `scripts/inspect_model.py`: load each `.mlmodelc`/`.mlpackage`, dump input/output
  names, dtypes, shapes, and any flexible/state inputs → `ModelSignatures.json`.
  Also extract `prompt_id` language map from `metadata.json`.
- `compile_mlpackage.sh`: only if HF ships `.mlpackage` (use `coremlcompiler compile`).
- **Done when:** `ModelSignatures.json` lists the real tensor names + the prompt-ID map.
  **This gates Phase 5 — no decode code is written against guessed names.**

### Phase 5 — CoreML loading + adaptive runner  *(needs model)*
- `ModelLocator`: validate `Models/`, surface "missing/invalid" to UI.
- `CoreMLModelRunner`: load preprocessor/encoder/decoder_joint with
  `MLModelConfiguration` (try `.all`, fall back `.cpuAndNeuralEngine` → `.cpuAndGPU` → `.cpuOnly`,
  selectable in UI). Use `MLState` if the models expose stateful caches; else thread
  cache tensors via `MLFeatureProvider` per `ModelSignatures`.
- Log per-model load time + memory delta.
- **Done when:** all modules load on device; a single preprocessor→encoder pass runs on a
  fixture wav without shape errors.

### Phase 6 — RNN-T / TDT greedy decoder + tokenizer
- `Tokenizer`: parse `tokenizer.json` (BPE/SentencePiece), id→piece, blank handling,
  CJK-aware detokenization (no spaces between CJK; normal spacing for Latin).
- `RNNTDecoder`: greedy loop over encoder frames — run `decoder_joint`, argmax, emit
  non-blank tokens, advance prediction-network state, cap inner loop (max symbols/step).
- `LanguagePrompt`: `ASRLanguage` enum → `prompt_id` **loaded from metadata** (not literals).
- `NemotronASRService`: orchestrate preprocessor→encoder(cache)→decoder loop per chunk,
  keep encoder + decoder state across chunks, push partial → `ASRState`, finalize on stop.
- **Done when:** a fixture wav produces a plausible transcript end-to-end on device.

### Phase 7 — Streaming integration + live demo
- Wire mic → chunk buffer → service on a background actor; partials on `@MainActor`.
- Language switching resets/sets `prompt_id` and decoder state appropriately.
- **Done when:** speaking into the mic shows partials within ~1 s of chunk completion and a
  final transcript on stop; 5-min continuous run without crash; peak memory < 1.5 GB.

### Phase 8 — Accuracy + Whisper baseline + docs
- `Samples/` test set + reference transcripts; compute WER (Latin) / CER (CJK) offline.
- Optional WhisperKit/whisper.cpp baseline target for the comparison table.
- `README.md` per §17 of the original plan; fill the benchmark + known-issues templates.
- **Done when:** README + benchmark template + known-issues list are complete.

---

## 5. Decisions needing confirmation (defaults chosen, change if needed)

- **Model variant/tier:** default **`multilingual` @ 2240 ms** (best for CJK + Cantonese,
  recommended tier). INT8/6-bit palettization is already baked into the published weights.
- **Min iOS:** **17.0** (matches original). Bump to 18 only if a CoreML state API forces it.
- **Project tool:** **XcodeGen** (`project.yml`), since it's installed and keeps the repo diffable.
- **Cantonese:** plan keeps Whisper as fallback for Cantonese per §15 if Nemotron CER is weak.

---

## 6. Top risks (and mitigations baked into the plan)

| Risk | Mitigation |
|---|---|
| CoreML I/O names / cache shapes unknown | `inspect_model.py` → `ModelSignatures.json`; runner adapts at runtime (Phase 4 gates Phase 5) |
| RNN-T needs custom Swift decode | Dedicated `RNNTDecoder`, tested on fixtures before live mic |
| Stateful cache handling | Prefer `MLState`; fall back to manual tensor threading per signatures |
| Memory > iOS limit / Jetsam | INT8/6-bit weights, `MemoryMonitor` with live cap warning, tier downshift option |
| CJK detokenization spacing | Tokenizer CJK-join unit tests in Phase 6 |
| prompt_id guesses wrong | Loaded from `metadata.json`, never hardcoded |
| Model absent during dev | App degrades gracefully; Phases 1–3 need no model |

---

## 7. Definition of done

App runs on a physical iPhone; all 5 modules load; mic → 16 kHz chunks; at least one full
Nemotron inference pass produces a partial/final transcript; benchmark metrics visible;
README documents setup, results template, and remaining issues. (Mirrors §19 of the original plan.)
