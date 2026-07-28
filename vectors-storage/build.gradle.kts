description = "Off-heap memory, mmap, arena-based storage, and on-disk formats"

// TestContainers + LocalStack version used for @Tag("integration") tests.
// Pin to a specific LocalStack image version to avoid Docker Hub auth requirements
// introduced in March 2026 for localstack/localstack:latest.
val testcontainersVersion = "1.21.4"
val awsSdkVersion = "2.29.52"

dependencies {
    api(project(":vectors-core"))

    // S3 is an opt-in capability. The implementation remains in this module so its established
    // package and API stay stable, while vectors-storage-s3 supplies the SDK at runtime.
    compileOnly("software.amazon.awssdk:s3:$awsSdkVersion")
    testImplementation("software.amazon.awssdk:s3:$awsSdkVersion")

    // Integration tests: LocalStack container exercises the real S3 API surface.
    testImplementation("org.testcontainers:testcontainers:$testcontainersVersion")
    testImplementation("org.testcontainers:junit-jupiter:$testcontainersVersion")
    testImplementation("org.testcontainers:localstack:$testcontainersVersion")

    // Integration tests against external S3-compatible endpoints (Cloudflare R2, AWS,
    // MinIO) load credentials from a gitignored .env file at the repo root. dotenv-java
    // is dependency-free and only used in test scope.
    testImplementation("io.github.cdimascio:dotenv-java:3.2.0")
}
