description = "Facade: VectorCollection, Document, MetadataValue, FlatScanAdapter"

dependencies {
    api(project(":vectors-core"))
    api(project(":vectors-storage"))
    api(project(":vectors-quantization"))
    implementation(project(":vectors-hnsw"))
    implementation(project(":vectors-vamana"))
    implementation(project(":vectors-ivf"))
}
