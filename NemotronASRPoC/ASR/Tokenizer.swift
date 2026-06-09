import Foundation

/// SentencePiece-style detokenizer for the Nemotron multilingual vocab.
///
/// `tokenizer.json` is a flat `{ "<id>": "<piece>" }` map of 13,087 pieces
/// (ids 0…13086). Word boundaries are encoded with `▁` (U+2581, "lower one
/// eighth block"). The RNN-T blank id (13087) is NOT in the map — it is the
/// loop's advance signal and is filtered out before detokenization.
struct Tokenizer {
    /// id → piece
    private let pieces: [Int: String]
    let blankID: Int

    /// Word-boundary marker used by SentencePiece.
    static let wordBoundary: Character = "\u{2581}"  // ▁

    init(pieces: [Int: String], blankID: Int) {
        self.pieces = pieces
        self.blankID = blankID
    }

    enum LoadError: Error, CustomStringConvertible {
        case fileNotFound(URL)
        case decodeFailed(Error)
        var description: String {
            switch self {
            case .fileNotFound(let u): return "tokenizer.json not found at \(u.path)"
            case .decodeFailed(let e): return "tokenizer.json decode failed: \(e)"
            }
        }
    }

    /// Load from a `tokenizer.json` flat map. `blankID` comes from
    /// `ModelSignatures` metadata (13087).
    static func load(contentsOf url: URL, blankID: Int) throws -> Tokenizer {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw LoadError.fileNotFound(url)
        }
        do {
            let data = try Data(contentsOf: url)
            let raw = try JSONDecoder().decode([String: String].self, from: data)
            var pieces: [Int: String] = [:]
            pieces.reserveCapacity(raw.count)
            for (k, v) in raw {
                if let id = Int(k) { pieces[id] = v }
            }
            return Tokenizer(pieces: pieces, blankID: blankID)
        } catch {
            throw LoadError.decodeFailed(error)
        }
    }

    var vocabCount: Int { pieces.count }

    func piece(for id: Int) -> String? { pieces[id] }

    /// True for special/markup tokens like `<unk>` or language tags `<xx-XX>`,
    /// which should not appear in user-facing text.
    static func isSpecial(_ piece: String) -> Bool {
        piece.hasPrefix("<") && piece.hasSuffix(">") && piece.count > 2
    }

    /// Detokenize a sequence of token ids into text.
    ///
    /// Algorithm: drop blanks + special tokens, concatenate pieces, replace `▁`
    /// with a space, then collapse runs of whitespace and trim. CJK needs no
    /// special-casing — spacing is already encoded in the pieces (no `▁` between
    /// adjacent CJK characters).
    func decode(_ ids: [Int]) -> String {
        var joined = ""
        joined.reserveCapacity(ids.count * 2)
        for id in ids {
            if id == blankID { continue }
            guard let piece = pieces[id] else { continue }
            if Self.isSpecial(piece) { continue }
            joined.append(piece)
        }
        let spaced = joined.replacingOccurrences(of: String(Self.wordBoundary), with: " ")
        return Self.collapseWhitespace(spaced)
    }

    /// Collapse runs of whitespace into single spaces and trim the ends.
    static func collapseWhitespace(_ s: String) -> String {
        let parts = s.split(whereSeparator: { $0 == " " || $0 == "\t" || $0 == "\n" })
        return parts.joined(separator: " ")
    }
}
