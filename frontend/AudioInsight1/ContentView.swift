import SwiftUI
import UniformTypeIdentifiers
import PythonKit
import Combine

func sendActionToBackend(isListening: Bool) {
    guard let url = URL(string: "https://music-genre-detector.onrender.com/api/action") else { return }
    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")

    let body: [String: Any] = ["isListening": isListening]
    request.httpBody = try? JSONSerialization.data(withJSONObject: body)

    URLSession.shared.dataTask(with: request) { data, response, error in
        if let error = error {
            print("Error: \(error)")
            return
        }
        if let response = response as? HTTPURLResponse {
            print("Status code: \(response.statusCode)")
        }
        if let data = data {
            if let jsonResponse = try? JSONSerialization.jsonObject(with: data, options: []) {
                print("Response JSON: \(jsonResponse)")
            }
        }
    }.resume()
}




/*
 assume the Flask server is running locally on http://localhost:5000 and has an endpoint /api/action to handle the button press.
 */



struct ContentView: View {
    @State private var isListening = false
    @State private var predictedGenre: String = "Unknown"

    var body: some View {
        ZStack {
            BackgroundView(isListening: $isListening)
            
            VStack {
                NavBar(isListening: $isListening)  // Pass isListening to NavBar
                

            }
            
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

struct BackgroundView: View {
    @Binding var isListening: Bool

    var body: some View {
        VStack {
            LinearGradient(gradient: Gradient(colors: [.black , .gray]),
                           startPoint: .topLeading,
                           endPoint: .bottomTrailing)
            .frame(height: UIScreen.main.bounds.height * 0.885)
            .ignoresSafeArea()
        }
        .background(Color.white)
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
    @State private var predictedGenre: String = "Unknown"
    var imageName: String // Add imageName as a parameter
    @State private var showDocumentPicker = false
    @State private var hasListened = 0
    var body: some View {
        VStack(spacing: 8) {
            Text(isListening ? "Listening..." : "")
                .font(.system(size: 20, weight: .light))
                .foregroundColor(.white)
                .onAppear {
                    checkForGenrePrediction()
                }
            Button(action: {
                withAnimation {
                    isListening.toggle()
                }
                if isListening {
                    hasListened += 1
                }
                sendActionToBackend(isListening: isListening)
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
            if hasListened < 1 {

            } else if hasListened > 0 && !isListening {
                    Text("Predicted Genre: \(predictedGenre)")
                        .font(.title)
                        .padding()
                        .onAppear {
                            checkForGenrePrediction()
                        }

            } else {
                
            }
        }
        
    }
    func checkForGenrePrediction() {
            Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { timer in
                guard !self.isListening else { return }
                
                // Check if the genre prediction file exists
                guard let url = URL(string: "https://music-genre-detector.onrender.com/api/genres") else { return }
                
                URLSession.shared.dataTask(with: url) { data, response, error in
                    if let error = error {
                        print("Error fetching predicted genre: \(error)")
                        return
                    }
                    guard let data = data else { return }
                    do {
                        if let json = try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any],
                           let genres = json["genres"] as? [String] {
                            DispatchQueue.main.async {
                                self.predictedGenre = genres.first ?? ""
                            }
                        } else {
                            print("Invalid JSON structure")
                        }
                    } catch {
                        print("Error parsing JSON: \(error)")
                    }
                }.resume()
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
    @State private var genres = ["Rock", "Jazz", "Pop", "Metal", "HipHop", "Classical", "Blues", "Country", "Reggae", "Disco"]
    @State private var auras = ["Aura1", "Aura2", "Aura3", "Aura4", "Aura5", "Aura6", "Aura7"]
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
                .fill(Color.red)
                .frame(width: 100, height: 150)
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
