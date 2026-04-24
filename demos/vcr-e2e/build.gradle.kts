description = "End-to-end tour of the VCR test harness module family"

dependencies {
    implementation(project(":vectors-vcr-core"))
    implementation(project(":vectors-vcr-junit5"))
    implementation(project(":vectors-vcr-langchain4j"))
    implementation(project(":vectors-vcr-serde-jackson"))
    implementation("dev.langchain4j:langchain4j-core:1.0.0-beta1")

    testImplementation("org.junit.jupiter:junit-jupiter:5.11.4")
    testImplementation("org.assertj:assertj-core:3.27.2")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
    testRuntimeOnly("ch.qos.logback:logback-classic:1.5.15")
}

// vcr-e2e is test-driven — we don't expose a main class
tasks.named<JavaExec>("run") { enabled = false }
