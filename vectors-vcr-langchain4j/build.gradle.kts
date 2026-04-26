description = "LangChain4j VCR model wrappers (ChatModel, EmbeddingModel, EmbeddingInterceptor)"

dependencies {
    api(project(":vectors-vcr-core"))
    compileOnly("dev.langchain4j:langchain4j-core:1.13.1")
    testImplementation("dev.langchain4j:langchain4j-core:1.13.1")
    testImplementation(project(":vectors-vcr-serde-avaje"))
}
