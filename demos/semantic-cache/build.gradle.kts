description = "Semantic (similarity-threshold) cache demo using VectorDbSemanticCache"

dependencies {
    implementation(project(":vectors-cache-semantic-db"))
    implementation(project(":vectors-cache"))
    implementation(project(":vectors-db"))
    implementation(project(":vectors-core"))

    // CI-gating assertions on the demo's golden path (audit T3.10).
    testImplementation("org.junit.jupiter:junit-jupiter:5.11.4")
    testImplementation("org.assertj:assertj-core:3.27.2")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

application {
    mainClass.set("com.integrallis.vectors.demo.semanticcache.SemanticCacheApp")
    applicationDefaultJvmArgs = listOf("--add-modules", "jdk.incubator.vector")
}
