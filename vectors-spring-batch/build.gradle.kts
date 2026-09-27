description = "Spring Batch adapter for migrating java-vectors collections to the current on-disk format"

// Overridable so CI can compile the adapter against more than one Spring Batch line. The framework
// is compileOnly for the same reason as the Spring AI adapter: an application brings its own.
val springBatchVersion = providers.gradleProperty("springBatchVersion").getOrElse("5.2.2")
val springVersion = providers.gradleProperty("springFrameworkVersion").getOrElse("6.2.3")

dependencies {
    api(project(":vectors-db"))
    compileOnly("org.springframework.batch:spring-batch-core:$springBatchVersion")
    compileOnly("org.springframework.batch:spring-batch-infrastructure:$springBatchVersion")
    compileOnly("org.springframework:spring-core:$springVersion")
    testImplementation("org.springframework.batch:spring-batch-core:$springBatchVersion")
    testImplementation("org.springframework.batch:spring-batch-infrastructure:$springBatchVersion")
    testImplementation("org.springframework:spring-core:$springVersion")
}
