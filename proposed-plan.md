# PoC Plan: Run Nemotron-3.5-ASR Streaming 0.6B on iOS
## Goal
Build an iOS PoC demo to test whether NVIDIA Nemotron-3.5-ASR Streaming 0.6B can run locally on iPhone/iPad for offline, real-time speech recognition.
Research conclusion:
- llama.cpp is not suitable for this model.
- Best PoC route: CoreML first.
- Backup route: ONNX Runtime Mobile + CoreML Execution Provider.
- Compare against existing Whisper.cpp / WhisperKit baseline.
References:
- Nemotron is part of NVIDIA’s open model family.  [oai_citation:0‡NVIDIA Developer](https://developer.nvidia.com/topics/ai/nemotron?utm_source=chatgpt.com)
- Community CoreML ports already exist for Nemotron-3.5-ASR.  [oai_citation:1‡Hugging Face](https://huggingface.co/FluidInference/Nemotron-3.5-ASR-Streaming-Multilingual-0.6b-CoreML?utm_source=chatgpt.com)
- ONNX Runtime supports iOS and CoreML EP.  [oai_citation:2‡ONNX](https://onnx.ai/?utm_source=chatgpt.com)isper.cpp already supports iOS, Metal, and CoreML.  [oai_citation:3‡GitHub](https://github.com/ggml-org/whisper.cpp?utm_source=chatgpt.com)
---
# 1. PoC Scope
## In scope
- Native iOS app demo
- Offline speech recognition
- Microphone recording
- 16 kHz mono audio preprocessing
- Streaming chunk inference
- Display partial transcript
- Display final transcript
- Measure latency, RAM, CPU, battery, and accuracy
- Compare with Whisper.cpp baseline
## Out of scope
- Full TokKong integration
- App Store release
- Multi-speaker diarization
- Translation
- Cloud fallback
- Full production UI
---
# 2. Recommended Architecture
## Preferred Route: CoreML
```text
iOS Microphone
 ↓
AVAudioEngine
 ↓
16 kHz PCM Mono
 ↓
Audio Chunk Buffer
 ↓
Nemotron CoreML Encoder
 ↓
RNNT Decoder / Joint Model
 ↓
Streaming Text Output
 ↓
SwiftUI Demo UI

Use the community CoreML model first because it avoids building the full conversion pipeline.

Target model candidates:

FluidInference/Nemotron-3.5-ASR-Streaming-Multilingual-0.6b-CoreML
aufklarer/Nemotron-3.5-ASR-Streaming-0.6B-CoreML-INT8

Use INT8 first because memory is the biggest iOS risk.

⸻

3. Fallback Architecture

Backup Route: ONNX Runtime Mobile

iOS Microphone
 ↓
AVAudioEngine
 ↓
16 kHz PCM Mono
 ↓
ONNX Runtime Mobile
 ↓
CoreML Execution Provider
 ↓
Nemotron ONNX Model
 ↓
Streaming Text Output

Use this route only if CoreML integration fails.

ONNX Runtime iOS supports CoreML Execution Provider, but model operator compatibility must be tested.

⸻

4. Device Targets

Minimum test devices

iPhone 15 Pro / A17 Pro
iPhone 16 / A18
iPad Pro M-series

Optional lower-end stress test

iPhone 13
iPhone 14
iPad Air M1

Expected risk:

Nemotron INT8 memory usage may exceed 1GB during streaming.
Older iPhones may be killed by iOS Jetsam.

⸻

5. Success Criteria

The PoC is successful if:

1. App launches without crash
2. Model loads locally
3. Microphone audio is captured
4. Audio is converted to 16 kHz mono PCM
5. Streaming inference runs
6. Partial transcript appears within 1 second
7. Final transcript is generated
8. Peak memory stays under 1.5GB
9. App does not crash after 5 minutes continuous recording
10. Accuracy is comparable to or better than Whisper small/base for target languages

⸻

6. Test Languages

Priority languages for TokKong-style use case:

English
Cantonese / Traditional Chinese
Mandarin / Simplified Chinese
Japanese
Korean

If the CoreML model has separate Latin and multilingual versions, use:

multilingual/

because Chinese, Japanese, and Korean require full multilingual vocabulary.

⸻

7. Project Structure

NemotronASR-iOS-PoC/
├── NemotronASRPoC.xcodeproj
├── Models/
│   ├── NemotronEncoder.mlmodelc
│   ├── NemotronDecoder.mlmodelc
│   ├── NemotronJoint.mlmodelc
│   └── tokenizer.json / vocab.json
├── App/
│   ├── NemotronASRPoCApp.swift
│   ├── ContentView.swift
│   └── TranscriptView.swift
├── Audio/
│   ├── AudioRecorder.swift
│   ├── AudioResampler.swift
│   └── AudioChunkBuffer.swift
├── ASR/
│   ├── NemotronASRService.swift
│   ├── CoreMLModelRunner.swift
│   ├── RNNTDecoder.swift
│   ├── Tokenizer.swift
│   └── LanguagePrompt.swift
├── Benchmark/
│   ├── BenchmarkLogger.swift
│   ├── MemoryMonitor.swift
│   └── LatencyTracker.swift
└── README.md

⸻

8. Implementation Phases

Phase 1: Create iOS Demo Shell

Tasks:

1. Create a native SwiftUI iOS app.
2. Add microphone permission.
3. Build simple UI:
   - Start Recording
   - Stop Recording
   - Clear
   - Language selector
   - Partial transcript area
   - Final transcript area
   - Benchmark panel

UI fields:

Current language
Model load time
Current chunk latency
Average chunk latency
Peak memory
Total recording duration
Transcript text

⸻

Phase 2: Audio Capture

Use:

AVAudioEngine
AVAudioSession
AVAudioConverter

Requirements:

Input: device microphone
Output: 16 kHz mono Float32 PCM
Chunk size: 320 ms if supported by model
Sample rate: 16000 Hz
Channels: 1

Implementation notes:

1. Configure AVAudioSession with .playAndRecord or .record.
2. Install tap on input node.
3. Convert hardware sample rate to 16 kHz.
4. Normalize audio to Float32.
5. Push audio frames into chunk buffer.
6. Emit one chunk every 320 ms.

⸻

Phase 3: Load CoreML Models

Tasks:

1. Add .mlmodelc files to Xcode project.
2. Load model with MLModelConfiguration.
3. Test compute units:
   - .all
   - .cpuAndNeuralEngine
   - .cpuAndGPU
   - .cpuOnly
4. Log model load time and memory usage.

Preferred config:

let config = MLModelConfiguration()
config.computeUnits = .all

Fallback config:

config.computeUnits = .cpuAndNeuralEngine

Important:

Do not compile .mlpackage on device for PoC.
Use precompiled .mlmodelc when available.

⸻

Phase 4: Implement Streaming Inference

Expected streaming loop:

1. Receive 320 ms audio chunk.
2. Pass chunk + previous cache to encoder.
3. Update encoder cache.
4. Run RNNT decoder.
5. Run joint network.
6. Decode token IDs.
7. Update partial transcript.
8. Repeat until stop.

Pseudo-flow:

func processAudioChunk(_ chunk: [Float]) async {
    let encoderOutput = try await encoder.run(
        audioChunk: chunk,
        cache: currentCache,
        promptId: selectedLanguagePromptId
    )
    currentCache = encoderOutput.updatedCache
    let tokens = try await rnntDecoder.decode(
        encoderOutput: encoderOutput
    )
    let text = tokenizer.decode(tokens)
    await MainActor.run {
        transcriptState.updatePartial(text)
    }
}

⸻

9. Language Prompt Handling

Nemotron uses language-conditioning prompt IDs.

Create mapping:

enum ASRLanguage: String, CaseIterable {
    case englishUS = "en_us"
    case chineseMandarin = "zh_cn"
    case chineseTraditional = "zh_tw"
    case japanese = "ja_jp"
    case korean = "ko_kr"
}

Create prompt ID mapping based on the model card files.

struct LanguagePrompt {
    let language: ASRLanguage
    let promptId: Int
}

Important task:

Read the model repo config and confirm the exact prompt_id values.
Do not guess prompt IDs.

⸻

10. Tokenizer / Vocabulary

Tasks:

1. Locate vocab/tokenizer file from model repo.
2. Add tokenizer resource to app bundle.
3. Implement token ID to string conversion.
4. Handle blank token.
5. Handle punctuation tokens.
6. Handle CJK text without unwanted spaces.

Tokenizer logic:

English/French/German:
- Join tokens with normal spacing rules.
Chinese/Japanese/Korean:
- Prefer direct character-level joining.
- Avoid adding spaces between CJK characters.

⸻

11. Benchmarking

Collect these metrics:

Model load time
First partial transcript latency
Per-chunk latency p50
Per-chunk latency p90
Per-chunk latency p99
Real-time factor
Peak memory
Average memory
CPU usage
Thermal state
Battery drain over 5 minutes
Crash / Jetsam events

Real-time factor:

RTF = inference_time / audio_duration

Target:

RTF < 1.0 required
RTF < 0.3 preferred
RTF < 0.1 excellent

⸻

12. Accuracy Test Set

Create local test audio files:

Samples/
├── en_short.wav
├── en_long.wav
├── zh_short.wav
├── zh_long.wav
├── cantonese_short.wav
├── cantonese_long.wav
├── ja_short.wav
├── ko_short.wav
└── noisy_environment.wav

Each file should have:

- Reference transcript
- Language
- Duration
- Speaker notes
- Noise condition

Measure:

WER for English
CER for Chinese/Japanese/Korean
Subjective punctuation quality
Partial transcript stability
Final transcript quality

⸻

13. Compare Against Whisper.cpp

Add baseline comparison:

Whisper tiny
Whisper base
Whisper small
Nemotron INT8 CoreML

Comparison table:

Model
File size
Load time
RAM peak
RTF
First token latency
English WER
Chinese CER
Cantonese subjective score
Battery usage
Crash risk

⸻

14. Risk List

High Risk

1. CoreML model input/output names may be complex.
2. RNNT decoder may need custom Swift implementation.
3. Model may exceed iOS memory limits.
4. Chinese/Cantonese support may not be good enough.
5. Community CoreML model may work on macOS but not iOS.

Medium Risk

1. ANE may not support all operations.
2. CoreML may fall back to CPU/GPU.
3. Streaming cache shape may be hard to maintain.
4. Tokenizer behavior may differ from reference implementation.

Low Risk

1. Audio capture
2. 16 kHz resampling
3. SwiftUI demo
4. Benchmark logging

⸻

15. Fallback Plan

If CoreML fails

Try:

ONNX Runtime Mobile + CoreML EP

If ONNX CoreML EP fails

Try:

ONNX Runtime Mobile CPU only

If Nemotron is too heavy

Continue with:

Whisper.cpp
WhisperKit
Distil-Whisper
SenseVoice small

If Chinese/Cantonese quality is weak

Use Nemotron only for:

English
Japanese
Korean
Mandarin

Keep Whisper for Cantonese.

⸻

16. Deliverables

Claude Code should produce:

1. Working Xcode iOS project
2. SwiftUI demo interface
3. Microphone streaming input
4. CoreML model loading code
5. Streaming ASR service skeleton
6. Tokenizer implementation
7. Benchmark logger
8. README with setup instructions
9. Benchmark result template
10. Known issues list

⸻

17. README Requirements

README should include:

# Nemotron ASR iOS PoC
## Requirements
- Xcode 16+
- iOS 17+
- iPhone 15 Pro or newer recommended
- Microphone permission
- CoreML model files downloaded separately
## Model Setup
Place model files into:
Models/
Expected files:
- Encoder .mlmodelc
- Decoder .mlmodelc
- Joint .mlmodelc
- tokenizer/vocab file
- config file
## Run
1. Open project in Xcode
2. Select physical iPhone
3. Build and run
4. Select language
5. Tap Start Recording
6. Speak
7. Review transcript and benchmark metrics
## Benchmark
Record:
- Load time
- First partial latency
- Average chunk latency
- Peak memory
- RTF
- Accuracy notes

⸻

18. Claude Code Implementation Instruction

Please implement this PoC in small steps.

Priority order:

1. Build SwiftUI shell
2. Implement microphone capture
3. Implement 16 kHz resampling
4. Implement chunk buffer
5. Add benchmark logger
6. Add CoreML model loader
7. Inspect CoreML model input/output names
8. Implement CoreML runner
9. Implement placeholder tokenizer
10. Implement real tokenizer
11. Implement streaming ASR service
12. Add benchmark comparison screen
13. Add README

Important:

Do not assume model input/output tensor names.
First inspect the CoreML model metadata.
Then adapt the Swift code to the actual model signatures.

⸻

19. Definition of Done

The PoC is done when:

- App runs on physical iPhone
- Model loads successfully
- Microphone recording works
- 16 kHz chunks are generated
- At least one Nemotron inference pass succeeds
- Partial or final transcript appears
- Benchmark metrics are visible
- README explains remaining issues
