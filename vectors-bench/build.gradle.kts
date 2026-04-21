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
    // Time-box overrides for short/dev runs:
    //   -Pjmh.fork=1 -Pjmh.warmup=1 -Pjmh.iterations=2 -Pjmh.timeOnIteration=1s
    (project.findProperty("jmh.fork") as String?)?.let { fork.set(it.toInt()) }
    (project.findProperty("jmh.warmup") as String?)?.let { warmupIterations.set(it.toInt()) }
    (project.findProperty("jmh.iterations") as String?)?.let { iterations.set(it.toInt()) }
    (project.findProperty("jmh.timeOnIteration") as String?)?.let { timeOnIteration.set(it) }
    (project.findProperty("jmh.resultFormat") as String?)?.let { resultFormat.set(it) }

    // Persist results for audit trail — text format is human-readable; CSV available via
    // -Pjmh.resultFormat=CSV.  Output path is relative to the project directory.
    resultFormat.set(project.findProperty("jmh.resultFormat") as String? ?: "TEXT")
    val resultExt = (resultFormat.get() as String).lowercase()
    resultsFile.set(project.file("build/results/jmh/results.$resultExt"))
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

    // Forward -Pbench.* properties as -Dbench.* system properties the harness reads for its
    // sweep profile, per-axis overrides, and HNSW build parallelism.
    listOf(
        "bench.profile",
        "bench.hnsw.m", "bench.hnsw.ef", "bench.hnsw.efSearch", "bench.hnsw.threads",
        "bench.vamana.r", "bench.vamana.l", "bench.vamana.alpha", "bench.vamana.lSearch",
        "bench.vamana.threads",
        "bench.ivf.nprobe",
        "bench.adc.pq", "bench.adc.clusters", "bench.adc.overQuery", "bench.adc.aniso"
    ).forEach { key ->
        (project.findProperty(key) as String?)?.let { systemProperty(key, it) }
    }

    // Stream harness stdout/stderr to the Gradle console in real time.
    standardOutput = System.out
    errorOutput    = System.err

    doFirst {
        val out = project.file("build/results/recall-qps")
        out.mkdirs()
        println("[recallQps] Results will be written to: ${out.absolutePath}")
    }
}

// ---------------------------------------------------------------------------
// buildScalability — HNSW/Vamana/FLAT build-time scaling sweep
//
// Usage:
//   ./gradlew :vectors-bench:buildScalability
//   ./gradlew :vectors-bench:buildScalability \
//       -Pbench.algo=hnsw -Pbench.sizes=10000,50000,100000 \
//       -Pbench.hnsw.m=16 -Pbench.hnsw.ef=100
//
// Captures per-run wall time, throughput, heap delta, and GC count/time.
// Results land in ${DatasetRegistry.dataDir()}/results/build-scalability.{csv,json}.
// ---------------------------------------------------------------------------
tasks.register<JavaExec>("buildScalability") {
    group = "benchmark"
    description = "Sweep HNSW/Vamana/FLAT build time vs corpus size (synthetic random)"

    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("com.integrallis.vectors.bench.BuildScalabilityBenchmark")

    jvmArgs(
        "--add-modules", "jdk.incubator.vector",
        "-Xmx12g", "-Xms4g",
        "-XX:+UseG1GC",
        "-XX:MaxGCPauseMillis=100",
        "-Xlog:gc*:file=${project.file("build/results/build-scalability-gc.log")}:uptime,level,tags:filecount=1"
    )

    // Forward -Pbench.* properties as -Dbench.* system properties the main class reads.
    listOf(
        "bench.dim", "bench.algo", "bench.sizes", "bench.dataset",
        "bench.hnsw.m", "bench.hnsw.ef",
        "bench.vamana.r", "bench.vamana.l"
    ).forEach { key ->
        (project.findProperty(key) as String?)?.let { systemProperty(key, it) }
    }

    standardOutput = System.out
    errorOutput    = System.err

    doFirst {
        val out = project.file("build/results")
        out.mkdirs()
        println("[buildScalability] GC log: ${project.file("build/results/build-scalability-gc.log").absolutePath}")
    }
}

// ---------------------------------------------------------------------------
// hnswScalingProbe — HNSW parallel-build thread scaling (Phase D premise check)
// ---------------------------------------------------------------------------
tasks.register<JavaExec>("hnswScalingProbe") {
    group = "benchmark"
    description = "Measure ConcurrentHnswGraphBuilder speedup across thread counts"

    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("com.integrallis.vectors.bench.HnswThreadScalingProbe")

    jvmArgs(
        "--add-modules", "jdk.incubator.vector",
        "-Xmx6g", "-Xms2g",
        "-XX:+UseG1GC"
    )

    listOf(
        "bench.n", "bench.dim", "bench.hnsw.m", "bench.hnsw.ef",
        "bench.threads", "bench.warmup", "bench.iters"
    ).forEach { key ->
        (project.findProperty(key) as String?)?.let { systemProperty(key, it) }
    }

    standardOutput = System.out
    errorOutput    = System.err
}

// ---------------------------------------------------------------------------
// pqTrainProbe — one-shot PQ training timing (Round-2.E validation)
// ---------------------------------------------------------------------------
tasks.register<JavaExec>("pqTrainProbe") {
    group = "benchmark"
    description = "Measure PQ training wall clock (sequential vs parallel)"

    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("com.integrallis.vectors.bench.PqTrainProbe")

    jvmArgs(
        "--add-modules", "jdk.incubator.vector",
        "-Xmx12g", "-Xms4g",
        "-XX:+UseG1GC"
    )

    listOf("probe.n", "probe.dim", "probe.m", "probe.ks", "probe.threads").forEach { key ->
        (project.findProperty(key) as String?)?.let { systemProperty(key, it) }
    }

    standardOutput = System.out
    errorOutput    = System.err
}

