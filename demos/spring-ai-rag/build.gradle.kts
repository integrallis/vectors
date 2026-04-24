description = "Spring AI RAG demo using JavaVectorsVectorStore (drop-in for SimpleVectorStore)"

dependencies {
    implementation(project(":vectors-spring-ai"))
    implementation(project(":vectors-db"))
    implementation(platform("org.springframework.boot:spring-boot-dependencies:3.4.2"))
    implementation("org.springframework.boot:spring-boot-starter")
    implementation("org.springframework.ai:spring-ai-vector-store:1.0.0")
    implementation("org.springframework.ai:spring-ai-model:1.0.0")
    implementation("io.micrometer:micrometer-observation:1.14.4")
}

application {
    mainClass.set("com.integrallis.vectors.demo.springai.SpringAiRagApp")
    applicationDefaultJvmArgs = listOf("--add-modules", "jdk.incubator.vector")
}
