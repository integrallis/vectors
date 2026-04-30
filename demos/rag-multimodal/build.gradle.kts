plugins {
    id("org.openjfx.javafxplugin") version "0.1.0"
}

description = "Multimodal RAG demo: PDF ingestion, hybrid search, semantic caching"

val langchain4jVersion = "1.13.1"

javafx {
    version = "25"
    modules = listOf("javafx.controls", "javafx.fxml", "javafx.web", "javafx.swing")
}

dependencies {
    // java-vectors server (embedded) + client + supporting modules
    implementation(project(":vectors-server"))
    implementation(project(":vectors-server-client"))
    implementation(project(":vectors-core"))
    implementation(project(":vectors-db"))
    implementation(project(":vectors-hybrid"))
    implementation(project(":vectors-router"))
    implementation(project(":vectors-cache"))

    // LangChain4j
    implementation("dev.langchain4j:langchain4j:$langchain4jVersion")
    implementation("dev.langchain4j:langchain4j-embeddings-all-minilm-l6-v2:$langchain4jVersion-beta23")
    implementation("dev.langchain4j:langchain4j-open-ai:$langchain4jVersion")

    // PDF processing
    implementation("org.apache.pdfbox:pdfbox:3.0.3")

    // ONNX Runtime for local ML inference (document layout detection)
    implementation("com.microsoft.onnxruntime:onnxruntime:1.21.0")

    // .env file support
    implementation("io.github.cdimascio:dotenv-java:3.0.0")

    // Token counting
    implementation("com.knuddels:jtokkit:1.1.0")
    implementation("com.fasterxml.jackson.core:jackson-databind:2.18.2")
    implementation("com.fasterxml.jackson.dataformat:jackson-dataformat-yaml:2.18.2")

    // MaterialFX - Modern Material Design components
    implementation("io.github.palexdev:materialfx:11.17.0")

    // Test
    testImplementation("org.junit.jupiter:junit-jupiter:5.11.4")
    testImplementation("org.assertj:assertj-core:3.27.2")
    testImplementation("org.mockito:mockito-core:5.15.2")
    testImplementation("org.mockito:mockito-junit-jupiter:5.15.2")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

application {
    mainClass.set("com.integrallis.vectors.demo.rag.MultimodalRAGApp")
    applicationDefaultJvmArgs = listOf(
        "--add-modules", "jdk.incubator.vector",
        "--add-exports", "javafx.graphics/com.sun.javafx.util=ALL-UNNAMED"
    )
}

// Pass the project directory so dotenv-java can find .env regardless of CWD
// Set macOS Dock icon to our custom icon
tasks.named<JavaExec>("run") {
    systemProperty("dotenv.directory", project.projectDir.absolutePath)
    systemProperty("apple.awt.application.name", "java-vectors RAG")
    val dockIcon = rootProject.file("media/icons/icon.icns").absolutePath
    jvmArgs("-Xdock:name=java-vectors RAG", "-Xdock:icon=$dockIcon")
}

// Task to run the standalone CLI demo (no JavaFX)
tasks.register<JavaExec>("runStandalone") {
    group = "application"
    description = "Run standalone multimodal RAG demonstration (CLI, no JavaFX)"

    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("com.integrallis.vectors.demo.rag.MultimodalRAGStandalone")
    jvmArgs("--add-modules", "jdk.incubator.vector")
}
