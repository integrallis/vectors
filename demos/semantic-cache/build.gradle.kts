description = "Semantic (similarity-threshold) cache demo using VectorDbSemanticCache"

dependencies {
    implementation(project(":vectors-cache-semantic-db"))
    implementation(project(":vectors-cache"))
    implementation(project(":vectors-db"))
    implementation(project(":vectors-core"))
}

application {
    mainClass.set("com.integrallis.vectors.demo.semanticcache.SemanticCacheApp")
    applicationDefaultJvmArgs = listOf("--add-modules", "jdk.incubator.vector")
}
