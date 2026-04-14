description = "Off-heap memory, mmap, arena-based storage, and on-disk formats"

// TestContainers + LocalStack version used for @Tag("integration") tests.
// Pin to a specific LocalStack image version to avoid Docker Hub auth requirements
// introduced in March 2026 for localstack/localstack:latest.
val testcontainersVersion = "1.21.4"
val awsSdkVersion = "2.29.52"

dependencies {
    api(project(":vectors-core"))

    // Integration tests: S3StorageBackendTest uses LocalStack to exercise the real S3 API surface.
    testImplementation("org.testcontainers:testcontainers:$testcontainersVersion")
    testImplementation("org.testcontainers:junit-jupiter:$testcontainersVersion")
    testImplementation("org.testcontainers:localstack:$testcontainersVersion")
    testImplementation("software.amazon.awssdk:s3:$awsSdkVersion")
}
