import CoreML
import Foundation
import QuartzCore

/// Loads the Nemotron CoreML modules and runs generic, signature-driven
/// predictions. Tensor names/shapes/dtypes are never hardcoded here — callers
/// pass `MLFeatureValue`s keyed by the names in `ModelSignatures.json`, and the
/// helpers build/read `MLMultiArray`s according to those signatures.
///
/// Phase 5.1–5.2. The runner is intentionally low-level: higher layers
/// (`NemotronEncoderRunner`, `RNNTDecoder`, `NemotronASRService`) own the
/// streaming/decode logic and call `predict(...)` here.
final class CoreMLModelRunner {

    enum RunnerError: Error, CustomStringConvertible {
        case moduleNotFound(String)
        case loadFailed(module: String, underlying: Error)
        case missingOutput(module: String, name: String)
        case shapeError(String)

        var description: String {
            switch self {
            case .moduleNotFound(let m): return "CoreML module '\(m)' not found on disk"
            case .loadFailed(let m, let e): return "CoreML module '\(m)' failed to load: \(e)"
            case .missingOutput(let m, let n): return "module '\(m)' produced no output named '\(n)'"
            case .shapeError(let s): return "shape error: \(s)"
            }
        }
    }

    /// Per-module load metric for the benchmark panel / logs.
    struct LoadMetric {
        let module: String
        let loadSeconds: Double
        let memoryDeltaMB: Double
        let computeUnits: MLComputeUnits
    }

    let directory: URL
    let signatures: ModelSignatures
    private(set) var models: [String: MLModel] = [:]
    private(set) var loadMetrics: [LoadMetric] = []
    private(set) var resolvedComputeUnits: MLComputeUnits = .all

    init(directory: URL, signatures: ModelSignatures) {
        self.directory = directory
        self.signatures = signatures
    }

    /// Compute-unit fallback order: prefer ANE on device, degrade gracefully so
    /// the simulator (CPU/GPU only) still loads every module.
    static let computeUnitFallback: [MLComputeUnits] = [
        .all, .cpuAndNeuralEngine, .cpuAndGPU, .cpuOnly
    ]

    /// Load the named modules (e.g. ["preprocessor", "encoder", "decoder_joint"]).
    /// Each module is loaded with the first compute-unit setting that succeeds.
    func load(modules: [String]) throws {
        for module in modules {
            let url = directory.appendingPathComponent("\(module).mlmodelc")
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw RunnerError.moduleNotFound(module)
            }
            let model = try loadWithFallback(module: module, url: url)
            models[module] = model
        }
    }

    private func loadWithFallback(module: String, url: URL) throws -> MLModel {
        var lastError: Error?
        for units in Self.computeUnitFallback {
            let config = MLModelConfiguration()
            config.computeUnits = units
            let memBefore = MemoryMonitor.physicalFootprintMB() ?? 0
            let start = CACurrentMediaTime()
            do {
                let model = try MLModel(contentsOf: url, configuration: config)
                let loadSeconds = CACurrentMediaTime() - start
                let memAfter = MemoryMonitor.physicalFootprintMB() ?? memBefore
                loadMetrics.append(LoadMetric(
                    module: module,
                    loadSeconds: loadSeconds,
                    memoryDeltaMB: memAfter - memBefore,
                    computeUnits: units
                ))
                resolvedComputeUnits = units
                Log.model.info("Loaded \(module).mlmodelc in \(String(format: "%.2f", loadSeconds))s [\(Self.describe(units))], +\(String(format: "%.0f", memAfter - memBefore))MB")
                return model
            } catch {
                lastError = error
                Log.model.warning("Load \(module) with \(Self.describe(units)) failed: \(String(describing: error)); trying next compute unit")
            }
        }
        throw RunnerError.loadFailed(module: module, underlying: lastError ?? RunnerError.moduleNotFound(module))
    }

    // MARK: - Prediction

    /// Run a module by name with inputs keyed by signature tensor names.
    func predict(module: String, inputs: [String: MLFeatureValue]) throws -> MLFeatureProvider {
        guard let model = models[module] else { throw RunnerError.moduleNotFound(module) }
        let provider = try MLDictionaryFeatureProvider(dictionary: inputs.mapValues { $0 as Any })
        return try model.prediction(from: provider)
    }

    // MARK: - MLMultiArray helpers

    /// Build a Float32 `MLMultiArray` of the given shape from a flat row-major buffer.
    static func floatArray(_ values: [Float], shape: [Int]) throws -> MLMultiArray {
        let count = shape.reduce(1, *)
        guard values.count == count else {
            throw RunnerError.shapeError("expected \(count) floats for shape \(shape), got \(values.count)")
        }
        let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: count)
        values.withUnsafeBufferPointer { src in
            ptr.update(from: src.baseAddress!, count: count)
        }
        return array
    }

    /// Build a zero-filled Float32 `MLMultiArray`.
    static func zeros(shape: [Int]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
        let count = shape.reduce(1, *)
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: count)
        ptr.update(repeating: 0, count: count)
        return array
    }

    /// Build an Int32 scalar/vector `MLMultiArray`.
    static func int32Array(_ values: [Int32], shape: [Int]) throws -> MLMultiArray {
        let count = shape.reduce(1, *)
        guard values.count == count else {
            throw RunnerError.shapeError("expected \(count) ints for shape \(shape), got \(values.count)")
        }
        let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .int32)
        let ptr = array.dataPointer.bindMemory(to: Int32.self, capacity: count)
        values.withUnsafeBufferPointer { src in
            ptr.update(from: src.baseAddress!, count: count)
        }
        return array
    }

    /// Read a Float32 `MLMultiArray` back into a flat `[Float]` (row-major).
    static func toFloats(_ array: MLMultiArray) -> [Float] {
        let count = array.count
        if array.dataType == .float32 {
            let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: count)
            return Array(UnsafeBufferPointer(start: ptr, count: count))
        }
        // Fallback for non-float arrays (rare): go through NSNumber.
        return (0..<count).map { array[$0].floatValue }
    }

    /// Read an Int32/scalar `MLMultiArray`'s first element as Int.
    static func firstInt(_ array: MLMultiArray) -> Int {
        guard array.count > 0 else { return 0 }
        return array[0].intValue
    }

    /// Extract a single encoder frame `encoded[:, :, t]` → flat `[1024]`.
    /// `encoded` is `[1, dim, frames]` row-major, so element (0, d, t) lives at
    /// index `d * frames + t`.
    static func encoderFrame(from encoded: MLMultiArray, frame t: Int) throws -> [Float] {
        let shape = encoded.shape.map { $0.intValue }
        guard shape.count == 3, shape[0] == 1 else {
            throw RunnerError.shapeError("encoded expected [1, dim, frames], got \(shape)")
        }
        let dim = shape[1], frames = shape[2]
        guard t >= 0, t < frames else {
            throw RunnerError.shapeError("frame \(t) out of range 0..<\(frames)")
        }
        let ptr = encoded.dataPointer.bindMemory(to: Float.self, capacity: dim * frames)
        var out = [Float](repeating: 0, count: dim)
        for d in 0..<dim {
            out[d] = ptr[d * frames + t]
        }
        return out
    }

    static func describe(_ units: MLComputeUnits) -> String {
        switch units {
        case .all: return "all"
        case .cpuAndGPU: return "cpuAndGPU"
        case .cpuOnly: return "cpuOnly"
        case .cpuAndNeuralEngine: return "cpuAndNeuralEngine"
        @unknown default: return "unknown"
        }
    }
}
