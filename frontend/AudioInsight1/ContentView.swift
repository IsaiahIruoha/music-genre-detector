import SwiftUI
import UniformTypeIdentifiers
import PythonKit

struct ContentView: View {
    @State private var isListening = false

    var body: some View {
        VStack {
            NavBar(isListening: $isListening)  // Pass isListening to NavBar

        }
    }
}

struct MainView: View {
    @Binding var isListening: Bool

    var body: some View {
        VStack {
            ZStack {
                BackgroundView(isListening: $isListening)
                VStack {
                    ProductTextView(productName: "AudioInsight")

                    Spacer()

                    MainStatusView(isListening: $isListening, imageName: isListening ? "ear.badge.waveform" : "ear.trianglebadge.exclamationmark")
                        .padding(.bottom, 40)

                    Spacer()
                }

                if isListening {
                    ForEach(0..<10) { index in
                        MusicNoteView(index: index)
                    }
                }
            }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}

struct DefaultView: View {
    var dayOfWeek: String
    var imageName: String

    var body: some View {
        VStack {
            Text(dayOfWeek)
                .font(.system(size: 16, weight: .medium, design: .default))
                .foregroundColor(.white)
            Image(systemName: imageName)
                .renderingMode(.original)
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(width: 40, height: 40)
        }
    }
}

struct BackgroundView: View {
    @Binding var isListening: Bool

    var body: some View {
        LinearGradient(gradient: Gradient(colors: [.black , .gray]),
                       startPoint: .topLeading,
                       endPoint: .bottomTrailing)
            .ignoresSafeArea()
    }
}

struct ProductTextView: View {
    var productName: String

    var body: some View {
        Text(productName)
            .font(.system(size: 32, weight: .heavy, design: .rounded))
            .foregroundColor(.white)
            .padding()
    }
}

struct MainStatusView: View {
    @Binding var isListening: Bool
    var imageName: String // Add imageName as a parameter

    @State private var showDocumentPicker = false

    var body: some View {
        VStack(spacing: 8) {
            if !isListening {

                }
            }

            Text(isListening ? "Listening..." : "Waiting...")
                .font(.system(size: 20, weight: .light))
                .foregroundColor(.white)

            Button(action: {
                withAnimation {
                    isListening.toggle()
                }
                if isListening {
                    //startRecording()
                } else {
                    //stopRecording()
                } 
            }) {
                Image(systemName: imageName) // Use imageName here
                    .renderingMode(.template)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(width: 180, height: 180)
                    .font(.system(size: 70, weight: .light))
                    .foregroundColor(.white)
                    .offset(x: imageName == "ear.trianglebadge.exclamationmark" ? 10 : 0)
                    .transition(.opacity)
            }
        }
    }

    func startRecording() {
        let sys = Python.import("sys")
        sys.path.append("/Users/simonrisk/Documents/AudioInsight1/AudioInsight1")
        let soundCapture = Python.import("soundCapture")

        DispatchQueue.global(qos: .background).async {
            soundCapture.start_recording()
        }
    }

    func stopRecording() {
        let sys = Python.import("sys")
        sys.path.append("/Users/simonrisk/Documents/AudioInsight1/AudioInsight1")
        let soundCapture = Python.import("soundCapture")

        DispatchQueue.global(qos: .background).async {
            soundCapture.stop_recording("/Users/simonrisk/Documents/AudioInsight1/AudioInsight1/files")
        }
    }



struct ListenButton: View {
    var title: String
    var textColor: Color
    var backgroundColor: Color

    var body: some View {
        Text(title)
            .font(.system(size: 20, weight: .bold))
            .foregroundColor(textColor)
            .padding()
            .background(backgroundColor)
            .cornerRadius(10)
    }
}

struct DocumentPickerView: UIViewControllerRepresentable {
    @Binding var isPresented: Bool

    class Coordinator: NSObject, UIDocumentPickerDelegate {
        var parent: DocumentPickerView

        init(parent: DocumentPickerView) {
            self.parent = parent
        }

        func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
            parent.isPresented = false
            guard let url = urls.first else { return }
            do {
                let data = try Data(contentsOf: url)
                // Handle the selected file data
                print("File selected: \(data)")
            } catch {
                print("Failed to read file data: \(error)")
            }
        }

        func documentPickerWasCancelled(_ controller: UIDocumentPickerViewController) {
            parent.isPresented = false
        }
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(parent: self)
    }

    func makeUIViewController(context: Context) -> UIDocumentPickerViewController {
        let picker = UIDocumentPickerViewController(forOpeningContentTypes: [UTType.data])
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: UIDocumentPickerViewController, context: Context) {}
}

struct MusicNoteView: View {
    let index: Int
    @State private var animate = false

    var body: some View {
        Image(systemName: "music.note")
            .resizable()
            .aspectRatio(contentMode: .fit)
            .frame(width: 20, height: 20)
            .foregroundColor(.white) // Change this line to set the color to pink
            .offset(x: self.animate ? CGFloat.random(in: -200...200) : 0, y: self.animate ? CGFloat.random(in: -200...200) : 0)
            .opacity(self.animate ? 0 : 1)
            .onAppear {
                self.startAnimation(index: index)
            }
    }

    private func startAnimation(index: Int) {
        withAnimation(Animation.easeOut(duration: 1).delay(Double(index) * 0.1)) {
            self.animate = true
        }
        // Reset animation state to keep it continuous
        DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
            self.animate = false
            self.startAnimation(index: index)
        }
    }
}

struct Page3View: View {
    @State private var selectedCategory: Category = .genre
    @State private var genres = ["Rock", "Jazz", "Pop", "Metal", "HipHop", "Classical"]
    @State private var auras = ["Aura1", "Aura2", "Aura3", "Aura4", "Aura5", "Aura6"]
    @State private var selectedItem: CategoryItem? = nil
    @State private var showDescription = false

    var body: some View {
        VStack {
            HStack {
                Button(action: {
                    selectedCategory = .genre
                }) {
                    Text("Genre")
                        .padding()
                        .background(selectedCategory == .genre ? Color.pink : Color.gray)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                }

                Button(action: {
                    selectedCategory = .aura
                }) {
                    Text("Aura")
                        .padding()
                        .background(selectedCategory == .aura ? Color.pink : Color.gray)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                }
            }
            .padding()

            ScrollView {
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 100))]) {
                    if selectedCategory == .genre {
                        ForEach(genres, id: \.self) { genre in
                            CategoryItemView(name: genre, selectedItem: $selectedItem, showDescription: $showDescription)
                        }
                    } else {
                        ForEach(auras, id: \.self) { aura in
                            CategoryItemView(name: aura, selectedItem: $selectedItem, showDescription: $showDescription)
                        }
                    }
                }
            }
        }
        .background(LinearGradient(gradient: Gradient(colors: [.black, .gray]), startPoint: .topLeading, endPoint: .bottomTrailing).ignoresSafeArea())
        .sheet(item: $selectedItem) { item in
            CategoryDescriptionView(name: item.name)
        }
    }
}

struct CategoryItemView: View {
    var name: String
    @Binding var selectedItem: CategoryItem?
    @Binding var showDescription: Bool

    var body: some View {
        VStack {
            Rectangle()
                .fill(Color.blue)
                .frame(height: 100)
                .cornerRadius(10)
                .onTapGesture {
                    selectedItem = CategoryItem(id: name, name: name)
                    showDescription = true
                }

            Text(name)
                .foregroundColor(.white)
        }
        .padding()
    }
}

struct CategoryDescriptionView: View {
    var name: String

    var body: some View {
        VStack {
            Text(name)
                .font(.largeTitle)
                .padding()

            Text("This is a brief description of \(name).")

                .padding()

            Spacer()
        }
        .background(Color.white)
    }
}

struct CategoryItem: Identifiable {
    var id: String
    var name: String
}

enum Category {
    case genre, aura
}
