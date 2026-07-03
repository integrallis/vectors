description = "SIMD rescore demo over a mock remote result set"

dependencies {
    implementation(project(":vectors-core"))

    // CI-gating assertions on the demo's golden path (audit T3.10).
    testImplementation("org.junit.jupiter:junit-jupiter:5.11.4")
    testImplementation("org.assertj:assertj-core:3.27.2")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

application {
    mainClass.set("com.integrallis.vectors.demo.rerank.RerankApp")
    applicationDefaultJvmArgs = listOf("--add-modules", "jdk.incubator.vector")
}
