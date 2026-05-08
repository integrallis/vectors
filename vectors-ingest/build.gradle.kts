description = "Streaming bulk ingestion: source → embedder → batched WAL/sidecart commit"

val langchain4jVersion = "1.13.1"

dependencies {
    api(project(":vectors-core"))
    api(project(":vectors-storage"))
    api(project(":vectors-ivf"))

    // langchain4j adapter is compileOnly so users bring their own version on the
    // runtime classpath; tests pull the full coordinate explicitly.
    compileOnly("dev.langchain4j:langchain4j-core:$langchain4jVersion")

    // JSONL parsing and cursor JSON serialisation.
    implementation("com.fasterxml.jackson.core:jackson-databind:2.18.2")

    testImplementation("dev.langchain4j:langchain4j-core:$langchain4jVersion")
    testImplementation("dev.langchain4j:langchain4j:$langchain4jVersion")
}
