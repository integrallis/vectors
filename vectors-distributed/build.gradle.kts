description = "Distributed vector search: scatter-gather, shard routing, cluster membership"

dependencies {
    api(project(":vectors-core"))
    api(project(":vectors-storage"))
    api(project(":vectors-db"))
    implementation(project(":vectors-ivf"))
}
