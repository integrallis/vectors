description = "Opt-in AWS SDK runtime for S3-compatible Vectors storage"

val governedDependencies = rootProject.extra["releaseDependencyModules"] as Map<*, *>

dependencies {
    api(platform("io.netty:netty-bom:4.1.139.Final"))
    api(project(":vectors-storage"))
    api("software.amazon.awssdk:s3:2.55.14")

    // Maven does not propagate this library's BOM to the SDK's transitive dependencies.
    // Declare the existing Netty runtime closure directly so Maven consumers get the same
    // patched versions as Gradle consumers. The BOM remains the single version source.
    listOf(
        "netty-buffer", "netty-codec", "netty-codec-http", "netty-codec-http2",
        "netty-common", "netty-handler", "netty-resolver", "netty-resolver-dns",
        "netty-codec-dns", "netty-transport",
        "netty-transport-classes-epoll", "netty-transport-native-unix-common"
    ).forEach { runtimeOnly("io.netty:$it") }
    // Keep the SDK's synchronous HTTP transport patched in Maven consumers too.
    listOf(
        "org.apache.httpcomponents.client5:httpclient5",
        "org.apache.httpcomponents.core5:httpcore5",
        "org.apache.httpcomponents.core5:httpcore5-h2"
    ).forEach { runtimeOnly("$it:${governedDependencies[it]}") }
}
