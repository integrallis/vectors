description = "LangChain4j decorator that wraps an EmbeddingModel with vectors-cache"

dependencies {
    api(project(":vectors-cache"))
    compileOnly("dev.langchain4j:langchain4j-core:1.13.1")

    testImplementation("dev.langchain4j:langchain4j-core:1.13.1")

    testImplementation("org.junit.jupiter:junit-jupiter:5.11.4")
    testImplementation("org.assertj:assertj-core:3.27.2")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}
