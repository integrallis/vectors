description = "Spring AI VectorStore adapter for java-vectors"

// Overridable so CI can compile/test the adapter against every supported Spring AI line
// (e.g. -PspringAiVersion=1.0.0 / 1.1.8 / 2.0.0). The adapter is compiled against the baseline
// so it stays forward-compatible across the range; the matrix proves it.
val springAiVersion = providers.gradleProperty("springAiVersion").getOrElse("1.1.4")

// The published jar is compiled once, against the baseline, and then loaded by applications on
// every supported line. -PspringAiCompileVersion pins what main is COMPILED against while
// -PspringAiVersion chooses what the tests RUN against, so a matrix cell can reproduce exactly
// that: baseline-compiled classes on a newer runtime. Unset, both are the same version.
val springAiCompileVersion =
    providers.gradleProperty("springAiCompileVersion").getOrElse(springAiVersion)

// Tests for API that only exists from Spring AI 2.1 (EmbeddedDocument, upsert, query-vector
// search) live in their own directory and join the test source set only on 2.1 and later; they
// cannot compile on the earlier lines.
val springAiMajorMinor = springAiVersion.split(".").take(2).map { it.toInt() }
val springAi21OrLater =
    springAiMajorMinor[0] > 2 || (springAiMajorMinor[0] == 2 && springAiMajorMinor[1] >= 1)
if (springAi21OrLater) {
    sourceSets["test"].java.srcDir("src/springAi21Test/java")
}
val micrometerObservationVersion =
    providers.gradleProperty("micrometerObservationVersion").getOrElse("1.14.4")

dependencies {
    api(project(":vectors-db"))
    implementation(project(":vectors-hybrid"))
    compileOnly("org.springframework.ai:spring-ai-vector-store:$springAiCompileVersion")
    compileOnly("org.springframework.ai:spring-ai-model:$springAiCompileVersion")
    compileOnly("io.micrometer:micrometer-observation:$micrometerObservationVersion")
    testImplementation("org.springframework.ai:spring-ai-vector-store:$springAiVersion")
    testImplementation("org.springframework.ai:spring-ai-model:$springAiVersion")
    testImplementation("io.micrometer:micrometer-observation:$micrometerObservationVersion")
}
