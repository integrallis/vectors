description = "SIMD rescore demo over a mock remote result set"

dependencies {
    implementation(project(":vectors-core"))
}

application {
    mainClass.set("com.integrallis.vectors.demo.rerank.RerankApp")
    applicationDefaultJvmArgs = listOf("--add-modules", "jdk.incubator.vector")
}
