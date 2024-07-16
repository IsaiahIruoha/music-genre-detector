import SwiftUI
import UniformTypeIdentifiers

struct UploadFileView: View {
    @State private var showDocumentPicker = false
    @State private var youtubeLink = ""
    @State private var showingAlert = false

    var body: some View {
        VStack(spacing: 20) {
            Text("Upload Options")
                .font(.largeTitle)
                .foregroundColor(.white)
                .padding(.top, 40)

            Button(action: {
                showDocumentPicker = true
            }) {
                Text("Upload File")
                    .font(.system(size: 20, weight: .bold))
                    .padding()
                    .background(Color.white)
                    .foregroundColor(.black)
                    .cornerRadius(10)
                    .overlay(
                        RoundedRectangle(cornerRadius: 10)
                            .stroke(Color.white, lineWidth: 1)
                    )
            }
            .sheet(isPresented: $showDocumentPicker) {
                DocumentPickerView(isPresented: $showDocumentPicker)
            }

            Text("OR")
                .font(.title)
                .foregroundColor(.white)

            VStack(alignment: .leading) {
                Text("Enter YouTube Link")
                    .foregroundColor(.white)

                TextField("https://www.youtube.com/...", text: $youtubeLink)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
                    .padding()
                    .background(Color.white)
                    .cornerRadius(10)
            }
            .padding(.horizontal)

            Button(action: {
                if isValidYouTubeLink(youtubeLink) {
                    // Handle the YouTube link
                    print("YouTube link: \(youtubeLink)")
                    youtubeLink = ""
                } else {
                    showingAlert = true
                }
            }) {
                Text("Submit Link")
                    .font(.system(size: 20, weight: .bold))
                    .padding()
                    .background(Color.white)
                    .foregroundColor(.black)
                    .cornerRadius(10)
                    .overlay(
                        RoundedRectangle(cornerRadius: 10)
                            .stroke(Color.white, lineWidth: 1)
                    )
            }
            .alert(isPresented: $showingAlert) {
                Alert(title: Text("Invalid Link"), message: Text("Please enter a valid YouTube link."), dismissButton: .default(Text("OK")))
            }

            Spacer()
        }
        .padding()
        .background(LinearGradient(gradient: Gradient(colors: [.black, .gray]),
                                   startPoint: .topLeading,
                                   endPoint: .bottomTrailing)
                        .ignoresSafeArea())
    }

    private func isValidYouTubeLink(_ link: String) -> Bool {
        let pattern = #"^(https?\:\/\/)?(www\.youtube\.com|youtu\.?be)\/.+$"#
        let regex = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive)
        return regex?.firstMatch(in: link, options: [], range: NSRange(location: 0, length: link.utf16.count)) != nil
    }
}

struct UploadFileView_Previews: PreviewProvider {
    static var previews: some View {
        UploadFileView()
    }
}
