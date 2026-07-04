description = "LangChain4J EmbeddingStore adapter for java-vectors"

// Overridable so CI can compile/test the adapter against every supported LangChain4j line
// (e.g. -Plangchain4jVersion=1.0.0 / 1.17.1). The adapter is compiled against the baseline so it
// stays forward-compatible across the range; the matrix proves it.
val langchain4jVersion = providers.gradleProperty("langchain4jVersion").getOrElse("1.13.1")

dependencies {
    api(project(":vectors-db"))
    compileOnly("dev.langchain4j:langchain4j-core:$langchain4jVersion")
    testImplementation("dev.langchain4j:langchain4j-core:$langchain4jVersion")
}
