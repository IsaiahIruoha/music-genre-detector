import SwiftUI

struct ContentView: View {
    @State private var isListening = false

    var body: some View {
        TabView {
            MainView(isListening: $isListening)
                .tabItem {
                    Image(systemName: "ear")
                    Text("Main")
                }

            Text("does this really need its own tab")
                .tabItem {
                    Image(systemName: "square.and.arrow.up")
                    Text("Page 2")
                }

            Page3View()
                .tabItem {
                    Image(systemName: "music.note")
                    Text("Page 3")
                }

            Text("Settings and stuff")
                .tabItem {
                    Image(systemName: "person")
                    Text("Page 4")
                }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
