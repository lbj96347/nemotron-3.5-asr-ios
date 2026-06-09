import Foundation

/// Languages the PoC exposes in the UI. The order here is the picker order.
///
/// IMPORTANT: the `prompt_id` integer that Nemotron's encoder expects for each
/// language is **not** hardcoded here — it is loaded at runtime from the model's
/// `metadata.json` (see `LanguagePromptMap`). This enum only carries the BCP-47
/// style code we match against that metadata.
enum ASRLanguage: String, CaseIterable, Identifiable, Hashable {
    case englishUS        = "en-US"
    case mandarinSimplified = "zh-CN"
    case cantoneseTraditional = "yue-Hant-HK"
    case chineseTraditional = "zh-TW"
    case japanese         = "ja-JP"
    case korean           = "ko-KR"

    var id: String { rawValue }

    /// Human-facing label for the picker.
    var displayName: String {
        switch self {
        case .englishUS:             return "English (US)"
        case .mandarinSimplified:    return "Mandarin (zh-CN)"
        case .cantoneseTraditional:  return "Cantonese (yue-HK)"
        case .chineseTraditional:    return "Chinese Traditional (zh-TW)"
        case .japanese:              return "Japanese"
        case .korean:                return "Korean"
        }
    }

    /// Whether transcripts in this language should be detokenized without spaces
    /// between adjacent CJK characters.
    var isCJK: Bool {
        switch self {
        case .englishUS: return false
        default:         return true
        }
    }
}

/// Resolves an `ASRLanguage` to the integer `prompt_id` the model was trained
/// with. Populated from `metadata.json` at model-load time. Until the model is
/// downloaded and inspected (Phase 4), this map is empty and callers should
/// treat a missing entry as "prompt conditioning unavailable yet".
struct LanguagePromptMap {
    private let promptIDByCode: [String: Int]

    init(promptIDByCode: [String: Int] = [:]) {
        self.promptIDByCode = promptIDByCode
    }

    func promptID(for language: ASRLanguage) -> Int? {
        promptIDByCode[language.rawValue]
    }

    var isEmpty: Bool { promptIDByCode.isEmpty }
}
