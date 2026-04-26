description = "LangChain4J EmbeddingStore adapter for java-vectors"

dependencies {
    api(project(":vectors-db"))
    compileOnly("dev.langchain4j:langchain4j-core:1.13.1")
    testImplementation("dev.langchain4j:langchain4j-core:1.13.1")
}
