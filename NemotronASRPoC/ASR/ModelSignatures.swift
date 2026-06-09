import Foundation

/// Strongly-typed view of `ModelSignatures.json` (produced by
/// `scripts/inspect_model.py`). The CoreML runner (Phase 5) and RNN-T decoder
/// (Phase 6) read tensor names / shapes and the language→prompt_id map from here
/// instead of hardcoding them, so the Swift code adapts to whatever the shipped
/// model actually exposes.
struct ModelSignatures: Codable, Equatable {
    struct TensorSpec: Codable, Equatable {
        let name: String
        let dtype: String?
        /// Shape entries are strings because MIL uses "?" for flexible dims.
        let shape: [String]?

        /// Concrete dimensions where known; "?" / non-numeric become nil.
        var dimensions: [Int?] { (shape ?? []).map { Int($0) } }
    }

    struct ModuleSignature: Codable, Equatable {
        let inputs: [TensorSpec]
        let outputs: [TensorSpec]

        func input(_ name: String) -> TensorSpec? { inputs.first { $0.name == name } }
        func output(_ name: String) -> TensorSpec? { outputs.first { $0.name == name } }
    }

    struct Metadata: Codable, Equatable {
        let model: String?
        let sample_rate: Int?
        let mel_features: Int?
        let chunk_mel_frames: Int?
        let pre_encode_cache: Int?
        let total_mel_frames: Int?
        let vocab_size: Int?
        let blank_idx: Int?
        let decoder_hidden: Int?
        let decoder_layers: Int?
        let encoder_dim: Int?
        let cache_channel_shape: [Int]?
        let cache_time_shape: [Int]?
        let default_prompt_id: Int?
    }

    let source: String?
    let modules: [String: ModuleSignature]
    let metadata: Metadata?
    let prompt_dictionary: [String: Int]?

    // MARK: - Loading

    enum LoadError: Error, CustomStringConvertible {
        case notFound
        case decodeFailed(Error)
        var description: String {
            switch self {
            case .notFound: return "ModelSignatures.json not found in bundle"
            case .decodeFailed(let e): return "ModelSignatures.json decode failed: \(e)"
            }
        }
    }

    /// Load the bundled signatures generated for the shipped model.
    static func loadFromBundle(_ bundle: Bundle = .main) -> Result<ModelSignatures, LoadError> {
        guard let url = bundle.url(forResource: "ModelSignatures", withExtension: "json") else {
            return .failure(.notFound)
        }
        do {
            let data = try Data(contentsOf: url)
            return .success(try JSONDecoder().decode(ModelSignatures.self, from: data))
        } catch {
            return .failure(.decodeFailed(error))
        }
    }

    // MARK: - Convenience

    var blankTokenID: Int? { metadata?.blank_idx }
    var vocabSize: Int? { metadata?.vocab_size }

    /// Resolve a BCP-47 style code to the model's `prompt_id`, falling back to
    /// the language part (e.g. "zh-TW" → "zh") and finally `default_prompt_id`.
    func promptID(forLanguageCode code: String) -> Int? {
        guard let dict = prompt_dictionary else { return metadata?.default_prompt_id }
        if let exact = dict[code] { return exact }
        if let base = code.split(separator: "-").first.map(String.init), let v = dict[base] {
            return v
        }
        return metadata?.default_prompt_id
    }

    /// Build the app-facing language→prompt_id map for the languages we expose.
    func languagePromptMap() -> LanguagePromptMap {
        var resolved: [String: Int] = [:]
        for language in ASRLanguage.allCases {
            if let id = promptID(forLanguageCode: language.rawValue) {
                resolved[language.rawValue] = id
            }
        }
        return LanguagePromptMap(promptIDByCode: resolved)
    }
}
