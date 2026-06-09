import SwiftUI

/// Shows the live partial hypothesis and the committed final transcript.
struct TranscriptView: View {
    let partial: String
    let final: String

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            section(title: "Partial", text: partial.isEmpty ? "—" : partial,
                    color: .secondary, mono: true)

            Divider()

            section(title: "Final", text: final.isEmpty ? "—" : final,
                    color: .primary, mono: false)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder
    private func section(title: String, text: String, color: Color, mono: Bool) -> some View {
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
            .frame(maxHeight: 140)
        }
    }
}

#Preview {
    TranscriptView(partial: "recognizing speech…",
                   final: "This is the committed transcript.")
        .padding()
}
