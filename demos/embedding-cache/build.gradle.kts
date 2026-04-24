description = "Embedding cache demo using mmap-persistent VectorCollection"

dependencies {
    implementation(project(":vectors-db"))
}

application {
    mainClass.set("com.integrallis.vectors.demo.cache.EmbeddingCacheApp")
    applicationDefaultJvmArgs = listOf("--add-modules", "jdk.incubator.vector")
}
