description = "Vectors Studio distributed backend — DistributedVectorCollection on R2 + sidecart text/blob"

val awsSdkVersion = "2.29.52"

dependencies {
    api(project(":vectors-studio-core"))
    api(project(":vectors-studio-sidecart"))
    api(project(":vectors-ivf"))
    api(project(":vectors-storage"))

    // S3StorageBackend lives in vectors-storage but the AWS SDK is brought in here so the
    // Cloudflare R2 path (and LocalStack for tests) can construct an S3Client.
    implementation("software.amazon.awssdk:s3:$awsSdkVersion")

    runtimeOnly("ch.qos.logback:logback-classic:1.5.15")

    testImplementation("org.assertj:assertj-core:3.27.2")
    testImplementation("org.junit.jupiter:junit-jupiter:5.11.4")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}
