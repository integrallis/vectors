description = "Embedded vector collection with persistence, metadata filtering, and ANN indexes"

// Apache Arrow's unsafe memory allocator requires access to internal JDK APIs on JDK 17+.
tasks.withType<Test> {
    jvmArgs(
        "--add-opens", "java.base/java.nio=ALL-UNNAMED",
        "--add-opens", "java.base/sun.nio.ch=ALL-UNNAMED"
    )
}

dependencies {
    api(project(":vectors-core"))
    api(project(":vectors-storage"))
    api(project(":vectors-quantization"))
    implementation(project(":vectors-hnsw"))
    implementation(project(":vectors-vamana"))
    implementation(project(":vectors-ivf"))
    // Optional GPU backend. Keep it off the CPU database's published/runtime dependency graph:
    // consumers that select a CUVS_* index add vectors-gpu explicitly.
    compileOnly(project(":vectors-gpu"))
    testImplementation(project(":vectors-gpu"))

    // Arrow IPC is an opt-in capability. vectors-db-arrow supplies these at runtime without
    // imposing Arrow, Jackson, and FlatBuffers on every embedded database consumer.
    compileOnly("org.apache.arrow:arrow-vector:19.0.0")
    compileOnly("org.apache.arrow:arrow-memory-unsafe:19.0.0")
    testImplementation("org.apache.arrow:arrow-vector:19.0.0")
    testImplementation("org.apache.arrow:arrow-memory-unsafe:19.0.0")
}
