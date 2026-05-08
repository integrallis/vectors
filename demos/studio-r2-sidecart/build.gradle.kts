description = "Demo: seed an R2 prefix + sidecart with the 24-doc demo corpus via BulkIngestor"

val langchain4jVersion = "1.13.1"
val awsSdkVersion = "2.29.52"

dependencies {
    implementation(project(":vectors-core"))
    implementation(project(":vectors-storage"))
    implementation(project(":vectors-ivf"))
    implementation(project(":vectors-ingest"))
    implementation(project(":vectors-studio-distributed"))
    implementation(project(":vectors-studio-sidecart"))

    // S3 client used by S3StorageBackend / Cloudflare R2.
    implementation("software.amazon.awssdk:s3:$awsSdkVersion")

    // PicoCLI for the seeder, dotenv-java for .env loading, langchain4j for embeddings.
    implementation("info.picocli:picocli:4.7.6")
    implementation("io.github.cdimascio:dotenv-java:3.2.0")
    implementation("dev.langchain4j:langchain4j:$langchain4jVersion")
    implementation("dev.langchain4j:langchain4j-embeddings-all-minilm-l6-v2:$langchain4jVersion-beta23")

    runtimeOnly("ch.qos.logback:logback-classic:1.5.15")

    testImplementation("org.assertj:assertj-core:3.27.2")
    testImplementation("org.junit.jupiter:junit-jupiter:5.11.4")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

// `runSeed` — embed the built-in 24-doc corpus, write vectors to R2, and write text rows
// to either an H2 file sidecart or a Cloudflare D1 sidecart. Reads R2 + D1 credentials
// from the demo's own .env first, falling back to the repo root for compatibility with
// other modules' integration tests.
tasks.register<JavaExec>("runSeed") {
    group = "application"
    description = "Embed corpus → R2 (vectors) + sidecart (text). See R2CorpusSeederDemo --help."
    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("com.integrallis.demos.studio.R2CorpusSeederDemo")
    jvmArgs("--add-modules", "jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED")
    systemProperty("dotenv.directory", project.projectDir.absolutePath)
}
