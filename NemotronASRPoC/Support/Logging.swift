import OSLog

/// Centralized loggers for the PoC. Use subsystem-scoped categories so signposts
/// and Console filtering stay clean.
enum Log {
    private static let subsystem = "ai.tokkong.nemotron.poc"
    static let audio = Logger(subsystem: subsystem, category: "audio")
    static let asr = Logger(subsystem: subsystem, category: "asr")
    static let model = Logger(subsystem: subsystem, category: "model")
    static let benchmark = Logger(subsystem: subsystem, category: "benchmark")
    /// Per-chunk live-streaming trace. Filter Console/Xcode by `category: stream`
    /// to follow the recording pipeline chunk-by-chunk (frames, tokens, memory).
    static let stream = Logger(subsystem: subsystem, category: "stream")
}
