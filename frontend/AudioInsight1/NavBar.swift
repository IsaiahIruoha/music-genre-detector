import SwiftUI

struct NavBar: View {
    @Binding var isListening: Bool
    @State private var selectedTab: Int = 0

    var body: some View {
        TabView(selection: $selectedTab) {
            MainView(isListening: $isListening)
                .tabItem {
                    Image(systemName: "ear")
                    Text("Main")
                }
                .tag(0)

            UploadFileView()  // Use the new UploadFileView here
                .tabItem {
                    Image(systemName: "square.and.arrow.up")
                    Text("Page 2")
                }
                .tag(1)

            Page3View()
                .tabItem {
                    Image(systemName: "music.note")
                    Text("Page 3")
                }
                .tag(2)

            SettingsView()  // Add the new settings view here
                .tabItem {
                    Image(systemName: "person")
                    Text("Page 4")
                }
                .tag(3)
        }
    }
}

struct NavBar_Previews: PreviewProvider {
    static var previews: some View {
        NavBar(isListening: .constant(false))
            .previewLayout(.sizeThatFits)
    }
}
