description = "LangChain4J EmbeddingStore adapter for java-vectors"

dependencies {
    api(project(":vectors-db"))
    compileOnly("dev.langchain4j:langchain4j-core:1.0.0-beta1")
    testImplementation("dev.langchain4j:langchain4j-core:1.0.0-beta1")
}
