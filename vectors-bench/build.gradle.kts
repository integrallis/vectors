plugins {
    id("me.champeau.jmh") version "0.7.2"
}

description = "JMH benchmarks and ANN-Benchmarks harness"

dependencies {
    implementation(project(":vectors-core"))
    implementation(project(":vectors-storage"))
    implementation(project(":vectors-quantization"))
    implementation(project(":vectors-hnsw"))
    implementation(project(":vectors-vamana"))
    implementation(project(":vectors-ivf"))
    implementation(project(":vectors-db"))

    // HDF5 reader for ANN-Benchmarks datasets (MIT, pure Java, no native code)
    implementation("io.jhdf:jhdf:0.9.4")

    implementation("org.openjdk.jmh:jmh-core:1.37")
    annotationProcessor("org.openjdk.jmh:jmh-generator-annprocess:1.37")
}

jmh {
    // Vector API module required for SIMD kernels.
    // -Xmx/-Xms: large heap for 1M-vector datasets loaded into memory.
    // AlwaysPreTouch: eliminates OS page-fault noise during measurement.
    // G1GC with bounded pause target: keeps GC pauses out of latency tail.
    jvmArgs.addAll(listOf(
        "--add-modules", "jdk.incubator.vector",
        "-Xmx8g", "-Xms8g",
        "-XX:+AlwaysPreTouch",
        "-XX:+UseG1GC",
        "-XX:MaxGCPauseMillis=100"
    ))
}
