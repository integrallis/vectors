description = "Spring AI VectorStore adapter for java-vectors"

dependencies {
    api(project(":vectors-db"))
    compileOnly("org.springframework.ai:spring-ai-vector-store:1.1.4")
    compileOnly("org.springframework.ai:spring-ai-model:1.1.4")
    compileOnly("io.micrometer:micrometer-observation:1.14.4")
    testImplementation("org.springframework.ai:spring-ai-vector-store:1.1.4")
    testImplementation("org.springframework.ai:spring-ai-model:1.1.4")
    testImplementation("io.micrometer:micrometer-observation:1.14.4")
}
