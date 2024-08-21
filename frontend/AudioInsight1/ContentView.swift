import SwiftUI
import UniformTypeIdentifiers
import PythonKit
import Combine
import AVFoundation

class AudioRecorder: NSObject, ObservableObject {
    var audioRecorder: AVAudioRecorder?
    let fileName = "audioRecording.m4a"
    
    func startRecording() {
        let audioSession = AVAudioSession.sharedInstance()
        do {
            try audioSession.setCategory(.playAndRecord, mode: .default)
            try audioSession.setActive(true)
            
//            let settings = [
//                AVFormatIDKey: Int(kAudioFormatMPEG4AAC),
//                AVSampleRateKey: 12000,
//                AVNumberOfChannelsKey: 1,
//                AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue
//            ]
            
            let settings = [
                AVFormatIDKey: Int(kAudioFormatLinearPCM),
                AVSampleRateKey: 44100,  // Standard sample rate for high-quality audio
                AVNumberOfChannelsKey: 1, // Mono
                AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue
            ]
            
            let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let audioFileURL = documentsDirectory.appendingPathComponent(fileName)
            
            audioRecorder = try AVAudioRecorder(url: audioFileURL, settings: settings)
            audioRecorder?.record()
            print("Recording started")
        } catch {
            print("Failed to start recording: \(error)")
        }
    }

    func stopRecording() {
        audioRecorder?.stop()
        audioRecorder = nil
        print("Recording stopped")
    }

    
    func getAudioFileURL() -> URL? {
        let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return documentsDirectory.appendingPathComponent(fileName)
    }
}

func sendActionToBackend(isListening: Bool, audioFileURL: URL?) {
    guard let url = URL(string: "http://10.253.194.118:8000/api/upload") else { return }
    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    
    let boundary = UUID().uuidString
    request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
    
    var body = Data()
    
    body.append("--\(boundary)\r\n".data(using: .utf8)!)
    body.append("Content-Disposition: form-data; name=\"isListening\"\r\n\r\n".data(using: .utf8)!)
    body.append("\(isListening)\r\n".data(using: .utf8)!)
    
    if let audioFileURL = audioFileURL {
        // Check if the file exists at the given path
        if FileManager.default.fileExists(atPath: audioFileURL.path) {
            print("File exists at \(audioFileURL.path)")
        } else {
            print("File does not exist at \(audioFileURL.path)")
            return
        }
        
        do {
            let audioData = try Data(contentsOf: audioFileURL)
            body.append("--\(boundary)\r\n".data(using: .utf8)!)
            body.append("Content-Disposition: form-data; name=\"file\"; filename=\"\(audioFileURL.lastPathComponent)\"\r\n".data(using: .utf8)!)
            body.append("Content-Type: audio/m4a\r\n\r\n".data(using: .utf8)!)
            body.append(audioData)
            body.append("\r\n".data(using: .utf8)!)
        } catch {
            print("Error reading audio file data: \(error)")
            return
        }
    }
    
    body.append("--\(boundary)--\r\n".data(using: .utf8)!)
    request.httpBody = body
    
    URLSession.shared.dataTask(with: request) { data, response, error in
        if let error = error {
            print("Error: \(error)")
            return
        }
        if let response = response as? HTTPURLResponse {
            print("Status code: \(response.statusCode)")
        }
        if let data = data, let jsonResponse = try? JSONSerialization.jsonObject(with: data, options: []) {
            print("Response JSON: \(jsonResponse)")
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
    @StateObject private var audioRecorder = AudioRecorder()
    var imageName: String
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
                    audioRecorder.startRecording()
                    hasListened += 1
                } else {
                    audioRecorder.stopRecording()
                    sendActionToBackend(isListening: isListening, audioFileURL: audioRecorder.getAudioFileURL())

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
                guard let url = URL(string: "http://10.253.194.118:8000/api/genres") else { return }
                
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
