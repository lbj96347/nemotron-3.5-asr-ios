import Foundation
import Observation

/// High-level lifecycle of the recording / inference session, surfaced to the UI.
enum ASRSessionStatus: Equatable {
    case idle
    case requestingPermission
    case permissionDenied
    case modelsMissing(reason: String)
    case loadingModels
    case ready
    case recording
    case finishing
    /// Offline transcription of an imported file is running; `progress` is a
    /// short human label such as "segment 2/14" (empty while audio loads).
    case transcribingFile(progress: String)
    case error(String)

    var isBusy: Bool {
        switch self {
        case .requestingPermission, .loadingModels, .finishing, .transcribingFile: return true
        default: return false
        }
    }

    var canStartRecording: Bool {
        switch self {
        case .ready: return true
        default: return false
        }
    }
}

/// Shared, observable state bridging the audio/ASR pipeline and SwiftUI.
/// All mutations happen on the main actor.
@MainActor
@Observable
final class ASRState {
    var status: ASRSessionStatus = .idle
    var selectedLanguage: ASRLanguage = .englishUS

    /// In-progress hypothesis for the current chunk window (may change).
    var partialTranscript: String = ""
    /// Stable, committed transcript text.
    var finalTranscript: String = ""

    /// Live benchmark snapshot for the on-screen panel.
    var benchmark: BenchmarkSnapshot = .empty

    var statusMessage: String {
        switch status {
        case .idle:                       return "Idle"
        case .requestingPermission:       return "Requesting microphone access…"
        case .permissionDenied:           return "Microphone access denied. Enable it in Settings."
        case .modelsMissing(let reason):  return "Models not found — \(reason)"
        case .loadingModels:              return "Loading model…"
        case .ready:                      return "Ready"
        case .recording:                  return "Recording…"
        case .finishing:                  return "Finalizing transcript…"
        case .transcribingFile(let p):    return p.isEmpty ? "Transcribing file…" : "Transcribing file… \(p)"
        case .error(let message):         return "Error: \(message)"
        }
    }

    func clearTranscripts() {
        partialTranscript = ""
        finalTranscript = ""
    }

    /// Commit the current partial into the final transcript (called when a chunk
    /// is finalized or recording stops).
    func commitPartial() {
        guard !partialTranscript.isEmpty else { return }
        if finalTranscript.isEmpty {
            finalTranscript = partialTranscript
        } else {
            let joiner = selectedLanguage.isCJK ? "" : " "
            finalTranscript += joiner + partialTranscript
        }
        partialTranscript = ""
    }
}
