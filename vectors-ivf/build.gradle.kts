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

    // DistributedVectorCollectionR2IT: real end-user flow against Cloudflare R2 with
    // real text embeddings produced by an in-process ONNX sentence-transformer.
    // Credentials are loaded from a gitignored .env at the repo root (.env.example
    // documents the keys); the IT auto-skips when the variables are missing.
    val langchain4jVersion = "1.13.1"
    testImplementation("io.github.cdimascio:dotenv-java:3.2.0")
    testImplementation("dev.langchain4j:langchain4j:$langchain4jVersion")
    testImplementation("dev.langchain4j:langchain4j-embeddings-all-minilm-l6-v2:$langchain4jVersion-beta23")
}
