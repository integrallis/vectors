description = "Spring AI VectorStore adapter for java-vectors"

// Overridable so CI can compile/test the adapter against every supported Spring AI line
// (e.g. -PspringAiVersion=1.0.0 / 1.1.8 / 2.0.0). The adapter is compiled against the baseline
// so it stays forward-compatible across the range; the matrix proves it.
val springAiVersion = providers.gradleProperty("springAiVersion").getOrElse("1.1.4")
val micrometerObservationVersion =
    providers.gradleProperty("micrometerObservationVersion").getOrElse("1.14.4")

dependencies {
    api(project(":vectors-db"))
    implementation(project(":vectors-hybrid"))
    compileOnly("org.springframework.ai:spring-ai-vector-store:$springAiVersion")
    compileOnly("org.springframework.ai:spring-ai-model:$springAiVersion")
    compileOnly("io.micrometer:micrometer-observation:$micrometerObservationVersion")
    testImplementation("org.springframework.ai:spring-ai-vector-store:$springAiVersion")
    testImplementation("org.springframework.ai:spring-ai-model:$springAiVersion")
    testImplementation("io.micrometer:micrometer-observation:$micrometerObservationVersion")
}
