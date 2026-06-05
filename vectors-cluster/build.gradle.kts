description = "Sharded write tier: consistent-hash document routing over N independent vectors-db shards (P3.2)"

dependencies {
    api(project(":vectors-core"))
    api(project(":vectors-db"))
    // Reuses the proven, fidelity-safe TopKMerger from the scatter-gather layer.
    implementation(project(":vectors-distributed"))
}
