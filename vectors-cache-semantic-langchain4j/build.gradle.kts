description = "LangChain4j decorator that answers from a semantically similar earlier request"

dependencies {
    api(project(":vectors-cache"))
    // Only for the test that proves an exact-key cache would have missed the paraphrase.
    testImplementation(project(":vectors-cache-langchain4j"))
    compileOnly("dev.langchain4j:langchain4j-core:1.13.1")

    testImplementation("dev.langchain4j:langchain4j-core:1.13.1")
    testImplementation("org.junit.jupiter:junit-jupiter:5.11.4")
    testImplementation("org.assertj:assertj-core:3.27.2")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}
