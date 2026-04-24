description = "Spring AI decorator that wraps an EmbeddingModel/ChatClient with vectors-cache"

dependencies {
    api(project(":vectors-cache"))
    compileOnly("org.springframework.ai:spring-ai-model:1.0.0")

    testImplementation("org.springframework.ai:spring-ai-model:1.0.0")

    testImplementation("org.junit.jupiter:junit-jupiter:5.11.4")
    testImplementation("org.assertj:assertj-core:3.27.2")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}
