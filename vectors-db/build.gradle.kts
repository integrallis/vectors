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

    // Apache Arrow IPC — batch ingestion / export (G6)
    implementation("org.apache.arrow:arrow-vector:19.0.0")
    implementation("org.apache.arrow:arrow-memory-unsafe:19.0.0")
}
