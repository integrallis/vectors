description = "Spring AI VCR model wrappers (ChatModel, EmbeddingModel)"

dependencies {
    api(project(":vectors-vcr-core"))
    compileOnly("org.springframework.ai:spring-ai-model:1.0.0")
    compileOnly("io.micrometer:micrometer-observation:1.14.4")
    testImplementation("org.springframework.ai:spring-ai-model:1.0.0")
    testImplementation("io.micrometer:micrometer-observation:1.14.4")
    testImplementation(project(":vectors-vcr-serde-avaje"))
}
