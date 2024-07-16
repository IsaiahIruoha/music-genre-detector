//
//  AudioInsight1App.swift
//  AudioInsight1
//
//  Created by Simon Risk on 2024-05-28.
//

import SwiftUI
import SwiftData
import PythonKit
import UniformTypeIdentifiers

@main
struct AudioInsight1App: App {
    var sharedModelContainer: ModelContainer = {
        let schema = Schema([
            Item.self,
        ])
        let modelConfiguration = ModelConfiguration(schema: schema, isStoredInMemoryOnly: false)

        do {
            return try ModelContainer(for: schema, configurations: [modelConfiguration])
        } catch {
            fatalError("Could not create ModelContainer: \(error)")
        }
    }()

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .modelContainer(sharedModelContainer)
    }
}
