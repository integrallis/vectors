description = "Embedding cache demo using CachingEmbeddingModel (vectors-cache-langchain4j) + Caffeine"

dependencies {
    implementation(project(":vectors-cache-langchain4j"))
    implementation(project(":vectors-cache"))
    implementation("dev.langchain4j:langchain4j-core:1.13.1")
}

application {
    mainClass.set("com.integrallis.vectors.demo.cache.EmbeddingCacheApp")
    applicationDefaultJvmArgs = listOf("--add-modules", "jdk.incubator.vector")
}
