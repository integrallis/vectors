description = "Jakarta Batch (JSR-352) batchlet for migrating java-vectors collections to the current on-disk format"

// The portable batch spec, for runtimes that are not Spring: JBeret on WildFly or Quarkus, and
// Open Liberty's batch container. compileOnly for the same reason as every other adapter here —
// the container supplies the implementation.
val jakartaBatchVersion = providers.gradleProperty("jakartaBatchVersion").getOrElse("2.1.1")
val jakartaInjectVersion = providers.gradleProperty("jakartaInjectVersion").getOrElse("2.0.1")

dependencies {
    api(project(":vectors-db"))
    compileOnly("jakarta.batch:jakarta.batch-api:$jakartaBatchVersion")
    compileOnly("jakarta.inject:jakarta.inject-api:$jakartaInjectVersion")
    testImplementation("jakarta.batch:jakarta.batch-api:$jakartaBatchVersion")
    testImplementation("jakarta.inject:jakarta.inject-api:$jakartaInjectVersion")
}
