// swift-tools-version:5.9
import PackageDescription

//var exclude: [String] = []

#if os(Linux)
// Linux doesn't support CoreML, and will attempt to import the coreml source directory
exclude.append("coreml")
#endif

let package = Package(
    name: "SwiftWhisper",
    platforms: [
        .macOS(.v12),
        .iOS(.v15),
        .tvOS(.v13),
    ],
    products: [
//        .library(name: "SwiftWhisper", targets: ["SwiftWhisper"]),
        .library(name: "SwiftWhisperStream", targets: ["SwiftWhisperStream"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ctreffs/SwiftSDL2.git", from: "1.4.1"),
    ],
    targets: [
//        .target(name: "SwiftWhisper", dependencies: [.target(name: "whisper_cpp")]),
        .target(
            name: "SwiftWhisperStream",
            dependencies: [
                .target(name: "whisper_cpp"),
                .target(name: "LibWhisper"),
            ],
            swiftSettings: [.unsafeFlags([
                "-c release --arch arm64 --arch x86_64",
            ])]
        ),
        .target(name: "LibWhisper", dependencies: [
            .target(name: "whisper_cpp"),
        ]),
        .target(
            name: "whisper_cpp",
            dependencies: [
                .product(name: "SDL", package: "SwiftSDL2"),
            ], 
            exclude: [
                "ggml-metal.metal",
            ],
            resources: [
                .process("ggml-metal.metal"),
//                .copy("ggml-metal.metal"),
            ],
            publicHeadersPath: "include",
            cSettings: [
                .unsafeFlags(["-Wno-shorten-64-to-32"]),
                .define("GGML_USE_ACCELERATE", .when(platforms: [.macOS, .macCatalyst, .iOS])),
//                .define("WHISPER_USE_COREML", .when(platforms: [.macOS, .macCatalyst, .iOS])),
                .define("GGML_USE_METAL", .when(platforms: [.macOS, .macCatalyst, .iOS])),
//                .unsafeFlags(["-DGGML_USE_METAL"]),
//                .unsafeFlags(["-DGGML_USE_ACCELERATE"]),
//                    .define("GGML_USE_METAL", .when(platforms: [.macOS, .macCatalyst, .iOS])),
//                .define("WHISPER_COREML_ALLOW_FALLBACK", .when(platforms: [.macOS, .macCatalyst, .iOS])),
                .unsafeFlags(["-O3"]),
                .unsafeFlags(["-DNDEBUG"]),
                .unsafeFlags(["-pthread"]),
                .unsafeFlags(["-fno-objc-arc"]),
            ],
            linkerSettings: [
                .linkedFramework("Foundation"),
                .linkedFramework("Accelerate"),
                .linkedFramework("Metal"),
                .linkedFramework("MetalKit"),
                .linkedFramework("MetalPerformanceShaders"),
            ]),
//        .testTarget(name: "WhisperTests", dependencies: [.target(name: "SwiftWhisper")], resources: [.copy("TestResources/")])
    ],
    cxxLanguageStandard: CXXLanguageStandard.cxx20
)
