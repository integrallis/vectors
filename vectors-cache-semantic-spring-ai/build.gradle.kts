description = "Spring AI decorator that answers from a semantically similar earlier prompt"

dependencies {
    api(project(":vectors-cache"))
    // Only for the test that proves an exact-key cache would have missed the paraphrase.
    testImplementation(project(":vectors-cache-spring-ai"))
    compileOnly("org.springframework.ai:spring-ai-model:1.1.4")

    testImplementation("org.springframework.ai:spring-ai-model:1.1.4")
    testImplementation("org.junit.jupiter:junit-jupiter:5.11.4")
    testImplementation("org.assertj:assertj-core:3.27.2")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}
