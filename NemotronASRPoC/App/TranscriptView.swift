import SwiftUI

/// Shows the live partial hypothesis and the committed final transcript.
struct TranscriptView: View {
    let partial: String
    let final: String

    private var isEmpty: Bool { partial.isEmpty && final.isEmpty }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("TRANSCRIPT")
                .font(.caption2.weight(.semibold))
                .foregroundStyle(.tertiary)

            if isEmpty {
                Text("Transcript will appear here…")
                    .font(.body)
                    .foregroundStyle(.tertiary)
                    .frame(maxWidth: .infinity, minHeight: 120, alignment: .topLeading)
            } else {
                section(title: "Partial", text: partial.isEmpty ? "—" : partial,
                        color: .secondary, mono: true, maxHeight: 120)

                Divider()

                section(title: "Final", text: final.isEmpty ? "—" : final,
                        color: .primary, mono: false, maxHeight: 200)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 12))
    }

    @ViewBuilder
    private func section(title: String, text: String, color: Color, mono: Bool, maxHeight: CGFloat) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title.uppercased())
                .font(.caption2.weight(.semibold))
                .foregroundStyle(.tertiary)
            ScrollView {
                Text(text)
                    .font(mono ? .system(.body, design: .monospaced) : .body)
                    .foregroundStyle(color)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
            }
            .frame(maxHeight: maxHeight)
        }
    }
}

#Preview {
    TranscriptView(partial: "recognizing speech…",
                   final: "This is the committed transcript.")
        .padding()
}
