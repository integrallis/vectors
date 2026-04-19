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
    implementation(project(":vectors-distributed"))

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

    // Allow filtering via: ./gradlew :vectors-bench:jmh -Pjmh.includes="BenchmarkName"
    if (project.hasProperty("jmh.includes")) {
        includes.set(listOf(project.property("jmh.includes") as String))
    }

    // Persist results for audit trail — text format is human-readable; CSV available via
    // -Pjmh.rf=csv.  Output path is relative to the project directory.
    resultFormat.set("TEXT")
    resultsFile.set(project.file("build/results/jmh/results.txt"))
}

// ---------------------------------------------------------------------------
// recallQps — end-to-end recall-vs-QPS harness (not JMH; plain main-class)
//
// Usage:
//   ./gradlew :vectors-bench:recallQps                          # all datasets & algorithms
//   ./gradlew :vectors-bench:recallQps -Pbench.dataset=sift     # filter by dataset name
//   ./gradlew :vectors-bench:recallQps -Pbench.algo=hnsw        # filter by algorithm
//   ./gradlew :vectors-bench:recallQps -Pbench.dataset=sift -Pbench.algo=hnsw
//
// Results are written to build/results/recall-qps/ as CSV + JSON.
// Datasets are downloaded automatically on first run to ~/.cache/java-vectors/datasets/
// (or the path set by the JAVA_VECTORS_DATASET_DIR environment variable).
// ---------------------------------------------------------------------------
tasks.register<JavaExec>("recallQps") {
    group = "benchmark"
    description = "Run the recall-vs-QPS end-to-end harness (downloads datasets if needed)"

    classpath = sourceSets["main"].runtimeClasspath

    mainClass.set("com.integrallis.vectors.bench.RecallQpsBenchmark")

    jvmArgs(
        "--add-modules", "jdk.incubator.vector",
        "-Xmx12g", "-Xms4g",
        "-XX:+UseG1GC",
        "-XX:MaxGCPauseMillis=100"
    )

    // Optional dataset and algorithm filters forwarded as positional args.
    val datasetFilter = project.findProperty("bench.dataset") as String?
    val algoFilter    = project.findProperty("bench.algo")    as String?
    if (datasetFilter != null) args(datasetFilter)
    if (algoFilter    != null) args(algoFilter)

    // Stream harness stdout/stderr to the Gradle console in real time.
    standardOutput = System.out
    errorOutput    = System.err

    doFirst {
        val out = project.file("build/results/recall-qps")
        out.mkdirs()
        println("[recallQps] Results will be written to: ${out.absolutePath}")
    }
}
