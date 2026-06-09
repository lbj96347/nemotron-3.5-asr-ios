import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @State private var state = ASRState()
    @State private var controller: RecordingController?
    @State private var tier: StreamingTier = .recommended
    @State private var isImportingFile = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    statusBanner
                    controls
                    BenchmarkPanelView(snapshot: state.benchmark)
                    TranscriptView(partial: state.partialTranscript, final: state.finalTranscript)
                }
                .padding()
            }
            .navigationTitle("Nemotron ASR")
            .navigationBarTitleDisplayMode(.inline)
        }
        .onAppear {
            if controller == nil {
                let c = RecordingController(state: state)
                controller = c
                c.refreshAvailability()
            }
        }
    }

    // MARK: - Sections

    private var statusBanner: some View {
        HStack(spacing: 8) {
            Circle().fill(statusColor).frame(width: 10, height: 10)
            Text(state.statusMessage).font(.subheadline)
            Spacer()
            if state.status.isBusy { ProgressView() }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 10))
    }

    private var controls: some View {
        VStack(spacing: 12) {
            HStack(spacing: 12) {
                Picker("Language", selection: Binding(
                    get: { state.selectedLanguage },
                    set: { controller?.setLanguage($0) }
                )) {
                    ForEach(ASRLanguage.allCases) { lang in
                        Text(lang.displayName).tag(lang)
                    }
                }
                .pickerStyle(.menu)
                .disabled(state.status == .recording || isTranscribingFile)

                Picker("Tier", selection: Binding(
                    get: { tier },
                    set: { tier = $0; controller?.setTier($0) }
                )) {
                    ForEach(StreamingTier.allCases) { t in
                        Text(t.displayName).tag(t)
                    }
                }
                .pickerStyle(.menu)
                .disabled(state.status == .recording || isTranscribingFile)
            }

            HStack(spacing: 12) {
                Button {
                    Task { await controller?.start() }
                } label: {
                    Label("Start", systemImage: "mic.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(!canStart)

                Button {
                    controller?.stop()
                } label: {
                    Label("Stop", systemImage: "stop.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .disabled(state.status != .recording)

                Button {
                    controller?.clear()
                } label: {
                    Label("Clear", systemImage: "trash")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .disabled(state.status == .recording || isTranscribingFile)
            }

            HStack(spacing: 12) {
                Button {
                    isImportingFile = true
                } label: {
                    Label("Import File", systemImage: "doc.badge.arrow.up")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .disabled(state.status == .recording || isTranscribingFile)

                if isTranscribingFile {
                    Button(role: .cancel) {
                        controller?.cancelTranscription()
                    } label: {
                        Label("Cancel", systemImage: "xmark.circle")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                }
            }
        }
        .fileImporter(
            isPresented: $isImportingFile,
            allowedContentTypes: [.audio],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                if let url = urls.first { controller?.transcribeImportedFile(url: url) }
            case .failure(let error):
                state.status = .error(error.localizedDescription)
            }
        }
    }

    private var isTranscribingFile: Bool {
        if case .transcribingFile = state.status { return true }
        return false
    }

    private var canStart: Bool {
        switch state.status {
        case .ready, .modelsMissing: return true   // audio + benchmark work without the model
        default: return false
        }
    }

    private var statusColor: Color {
        switch state.status {
        case .recording:        return .red
        case .ready:            return .green
        case .transcribingFile: return .blue
        case .modelsMissing:    return .orange
        case .permissionDenied, .error: return .red
        default:                return .gray
        }
    }
}

#Preview {
    ContentView()
}
