description = "Off-heap memory, mmap, arena-based storage, and on-disk formats"

// TestContainers + LocalStack version used for @Tag("integration") tests.
// Pin to a specific LocalStack image version to avoid Docker Hub auth requirements
// introduced in March 2026 for localstack/localstack:latest.
val testcontainersVersion = "1.21.4"
val awsSdkVersion = "2.29.52"

dependencies {
    api(project(":vectors-core"))

    // AWS SDK v2 S3: promoted to implementation so that S3StorageBackend is available
    // to runtime consumers (e.g. vectors-ivf integration tests) without them needing to
    // declare a separate S3 dependency. Public API of S3StorageBackend uses only java.net.URI
    // and java.lang.String so callers do not need the SDK on their compile classpath.
    implementation("software.amazon.awssdk:s3:$awsSdkVersion")

    // Integration tests: LocalStack container exercises the real S3 API surface.
    testImplementation("org.testcontainers:testcontainers:$testcontainersVersion")
    testImplementation("org.testcontainers:junit-jupiter:$testcontainersVersion")
    testImplementation("org.testcontainers:localstack:$testcontainersVersion")
}
