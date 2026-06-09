import SwiftUI

/// Compact grid of live benchmark metrics.
struct BenchmarkPanelView: View {
    let snapshot: BenchmarkSnapshot

    private let columns = [GridItem(.flexible()), GridItem(.flexible())]

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("BENCHMARK")
                .font(.caption2.weight(.semibold))
                .foregroundStyle(.tertiary)

            LazyVGrid(columns: columns, alignment: .leading, spacing: 8) {
                metric("Model load", ms(snapshot.modelLoadMS))
                metric("First partial", ms(snapshot.firstPartialLatencyMS))
                metric("Chunk p50", ms(snapshot.chunkP50MS))
                metric("Chunk p90", ms(snapshot.chunkP90MS))
                metric("Chunk p99", ms(snapshot.chunkP99MS))
                metric("Avg chunk", ms(snapshot.averageChunkMS))
                metric("RTF", String(format: "%.3f", snapshot.realTimeFactor),
                       warn: snapshot.realTimeFactor >= 1.0)
                metric("Duration", String(format: "%.1f s", snapshot.recordingSeconds))
                metric("Mem (now)", mb(snapshot.currentMemoryMB))
                metric("Mem (peak)", mb(snapshot.peakMemoryMB),
                       warn: snapshot.peakMemoryMB >= 1500)
                metric("Thermal", snapshot.thermalState,
                       warn: snapshot.thermalState == "Serious" || snapshot.thermalState == "Critical")
            }
        }
        .padding()
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 12))
    }

    private func metric(_ label: String, _ value: String, warn: Bool = false) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label).font(.caption2).foregroundStyle(.secondary)
            Text(value)
                .font(.system(.callout, design: .monospaced).weight(.medium))
                .foregroundStyle(warn ? .red : .primary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func ms(_ value: Double) -> String { value > 0 ? String(format: "%.0f ms", value) : "—" }
    private func mb(_ value: Double) -> String { value > 0 ? String(format: "%.0f MB", value) : "—" }
}

#Preview {
    BenchmarkPanelView(snapshot: .empty).padding()
}
