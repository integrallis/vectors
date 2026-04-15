description = "Inverted File (IVF) family indexes"

// Keep in sync with vectors-storage/build.gradle.kts
val testcontainersVersion = "1.21.4"
val awsSdkVersion = "2.29.52"

dependencies {
    api(project(":vectors-core"))
    api(project(":vectors-storage"))
    api(project(":vectors-quantization"))

    // Integration tests: DistributedVectorCollectionLocalStackIT exercises S3StorageBackend
    // against a real LocalStack S3 instance via TestContainers.
    // The SDK is also added here as testImplementation because the IT test constructs
    // S3Client directly to create the bucket in @BeforeAll setup.
    testImplementation("org.testcontainers:testcontainers:$testcontainersVersion")
    testImplementation("org.testcontainers:junit-jupiter:$testcontainersVersion")
    testImplementation("org.testcontainers:localstack:$testcontainersVersion")
    testImplementation("software.amazon.awssdk:s3:$awsSdkVersion")
}
