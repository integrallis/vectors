description = "LangChain4j RAG demo using JavaVectorsEmbeddingStore (drop-in for InMemoryEmbeddingStore)"

dependencies {
    implementation(project(":vectors-langchain4j"))
    implementation(project(":vectors-db"))
    implementation("dev.langchain4j:langchain4j-core:1.0.0-beta1")
}

application {
    mainClass.set("com.integrallis.vectors.demo.langchain4j.LangChain4jRagApp")
    applicationDefaultJvmArgs = listOf("--add-modules", "jdk.incubator.vector")
}
