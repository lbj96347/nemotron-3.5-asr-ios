import Foundation

/// Locates and validates the Nemotron CoreML model bundle inside the app, and
/// reports clearly when it is missing so the app can degrade gracefully (the
/// PoC shell must run without the ~hundreds-of-MB download present).
struct ModelLocator {
    /// Model modules we expect on disk for a tier. `decoder_joint` is the fused
    /// default path; `decoder`/`joint` are optional fallbacks.
    static let requiredModules = ["preprocessor", "encoder", "decoder_joint"]
    static let optionalModules = ["decoder", "joint"]
    static let requiredFiles = ["tokenizer.json", "metadata.json"]

    let variant: String   // "multilingual" | "latin"
    let tier: StreamingTier

    init(variant: String = "multilingual", tier: StreamingTier = .recommended) {
        self.variant = variant
        self.tier = tier
    }

    /// Directory the model should live in, bundled under Models/<variant>/<tier>ms/.
    var bundleDirectory: URL? {
        let subpath = "Models/\(variant)/\(tier.rawValue)ms"
        // Models/ is a folder reference, so it lands under the bundle root.
        if let url = Bundle.main.resourceURL?.appendingPathComponent(subpath),
           FileManager.default.fileExists(atPath: url.path) {
            return url
        }
        return nil
    }

    enum Availability: Equatable {
        case available(URL)
        case missing(reason: String)
    }

    func locate() -> Availability {
        guard let dir = bundleDirectory else {
            return .missing(reason: "no \(variant)/\(tier.rawValue)ms folder in app bundle. Run scripts/download_models.sh.")
        }

        let fm = FileManager.default
        for module in Self.requiredModules {
            let compiled = dir.appendingPathComponent("\(module).mlmodelc")
            if !fm.fileExists(atPath: compiled.path) {
                return .missing(reason: "\(module).mlmodelc not found in \(dir.lastPathComponent).")
            }
        }
        for file in Self.requiredFiles {
            if !fm.fileExists(atPath: dir.appendingPathComponent(file).path) {
                return .missing(reason: "\(file) not found in \(dir.lastPathComponent).")
            }
        }
        return .available(dir)
    }
}
