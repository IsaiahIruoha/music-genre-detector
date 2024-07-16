import SwiftUI

struct SettingsView: View {
    @State private var notificationsEnabled = true
    @State private var soundEnabled = true
    @State private var appVersion = "1.0.0"

    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("General Settings")) {
                    Toggle(isOn: $notificationsEnabled) {
                        Text("Enable Notifications")
                    }
                    Toggle(isOn: $soundEnabled) {
                        Text("Enable Sound")
                    }
                }

                Section(header: Text("About")) {
                    HStack {
                        Text("App Version")
                        Spacer()
                        Text(appVersion)
                    }
                }
            }
            .navigationBarTitle("Settings", displayMode: .inline)
        }
    }
}

struct SettingsView_Previews: PreviewProvider {
    static var previews: some View {
        SettingsView()
    }
}
