description = "Facade: VectorCollection, Document, MetadataValue, FlatScanAdapter"

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
    // Optional GPU backend. api-scope so downstream modules can directly reference
    // com.nvidia.cuvs types exposed transitively by vectors-gpu.
    api(project(":vectors-gpu"))

    // Apache Arrow IPC — batch ingestion / export (G6)
    implementation("org.apache.arrow:arrow-vector:19.0.0")
    implementation("org.apache.arrow:arrow-memory-unsafe:19.0.0")
}
