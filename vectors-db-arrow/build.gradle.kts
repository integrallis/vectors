description = "Opt-in Apache Arrow IPC runtime for Vectors import and export"

dependencies {
    api(project(":vectors-db"))
    api("org.apache.arrow:arrow-vector:19.0.0")
    api("org.apache.arrow:arrow-memory-unsafe:19.0.0")
}
