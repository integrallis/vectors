description = "Embedding cache demo using CachingEmbeddingModel (vectors-cache-langchain4j) + Caffeine"

dependencies {
    implementation(project(":vectors-cache-langchain4j"))
    implementation(project(":vectors-cache"))
    implementation("dev.langchain4j:langchain4j-core:1.13.1")

    // CI-gating assertions on the demo's golden path (audit T3.10).
    testImplementation("org.junit.jupiter:junit-jupiter:5.11.4")
    testImplementation("org.assertj:assertj-core:3.27.2")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

application {
    mainClass.set("com.integrallis.vectors.demo.cache.EmbeddingCacheApp")
    applicationDefaultJvmArgs = listOf("--add-modules", "jdk.incubator.vector")
}
