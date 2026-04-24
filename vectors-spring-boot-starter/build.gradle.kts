description = "Spring Boot auto-configuration starter for java-vectors"

dependencies {
    // Bring in the Spring AI adapter (which pulls in vectors-db transitively).
    // Marked as 'api' so consuming apps get VectorCollection + JavaVectorsVectorStore on their classpath.
    api(project(":vectors-spring-ai"))

    // Spring Boot autoconfigure — compile-only so users supply their own version via the Spring BOM.
    compileOnly("org.springframework.boot:spring-boot-autoconfigure:3.4.5")
    // Spring AI — compile-only; users provide via their Spring AI BOM.
    compileOnly("org.springframework.ai:spring-ai-vector-store:1.1.4")
    compileOnly("org.springframework.ai:spring-ai-model:1.1.4")
    // Micrometer — compile-only (transitive from Spring Boot Actuator in most apps).
    compileOnly("io.micrometer:micrometer-observation:1.14.4")

    // Test: we need actual Spring Boot test support to exercise the autoconfiguration.
    testImplementation("org.springframework.boot:spring-boot-test:3.4.5")
    testImplementation("org.springframework.boot:spring-boot-autoconfigure:3.4.5")
    testImplementation("org.springframework:spring-test:6.2.6")
    testImplementation("org.springframework.ai:spring-ai-vector-store:1.1.4")
    testImplementation("org.springframework.ai:spring-ai-model:1.1.4")
    testImplementation("io.micrometer:micrometer-observation:1.14.4")
}
