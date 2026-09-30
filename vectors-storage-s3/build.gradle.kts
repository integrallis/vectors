description = "Opt-in AWS SDK runtime for S3-compatible Vectors storage"

dependencies {
    api(platform("io.netty:netty-bom:4.1.138.Final"))
    api(project(":vectors-storage"))
    api("software.amazon.awssdk:s3:2.29.52")

    // Maven does not propagate this library's BOM to the SDK's transitive dependencies.
    // Declare the existing Netty runtime closure directly so Maven consumers get the same
    // patched versions as Gradle consumers. The BOM remains the single version source.
    listOf(
        "netty-buffer", "netty-codec", "netty-codec-http", "netty-codec-http2",
        "netty-common", "netty-handler", "netty-resolver", "netty-transport",
        "netty-transport-classes-epoll", "netty-transport-native-unix-common"
    ).forEach { runtimeOnly("io.netty:$it") }
}
