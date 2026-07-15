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

    // Cache layer (P1.6 benchmarks) — declared on the jmh source set ONLY, not main. vectors-bench
    // is an implementation dependency of vectors-optimizer (and thus vectors-studio-web), so putting
    // these on `implementation` would leak Caffeine + the JSR-107 provider into those modules'
    // runtime classpaths. `jmhImplementation` keeps them on the benchmark classpath alone.
    // The JCache provider is only a testImplementation of vectors-cache-jcache, so it is not on the
    // bench classpath transitively and must be declared explicitly for the JMH fork.
    jmhImplementation(project(":vectors-cache"))
    jmhImplementation(project(":vectors-cache-jcache"))
    jmhImplementation(project(":vectors-cache-semantic-db"))
    jmhImplementation("com.github.ben-manes.caffeine:jcache:3.1.8")

    // HDF5 reader for ANN-Benchmarks datasets (MIT, pure Java, no native code)
    implementation("io.jhdf:jhdf:0.9.4")

    // AWS SDK v2 S3: required by WriteAheadLogS3Benchmark to construct the S3 client that
    // fronts S3StorageBackend. vectors-storage keeps it as `implementation` so it is not on
    // public consumers' compile classpath; benchmarks declare it directly.
    implementation("software.amazon.awssdk:s3:2.29.52")

    implementation("org.openjdk.jmh:jmh-core:1.37")
    annotationProcessor("org.openjdk.jmh:jmh-generator-annprocess:1.37")
}

jmh {
    // -Pivfpq.largeScale=true un-gates IvfPqLargeScaleBenchmark (n=1M trial setup) and
    // bumps the JVM heap to 16g to accommodate a 1M×768 float corpus (~3 GB live) plus
    // the IVF-PQ codebook and K-Means intermediate state.
    val largeScale = project.hasProperty("ivfpq.largeScale")
    val xmx = if (largeScale) "16g" else "8g"

    // Vector API module required for SIMD kernels.
    // -Xmx/-Xms: large heap for 1M-vector datasets loaded into memory.
    // AlwaysPreTouch: eliminates OS page-fault noise during measurement.
    // G1GC with bounded pause target: keeps GC pauses out of latency tail.
    jvmArgs.addAll(listOf(
        "--add-modules", "jdk.incubator.vector",
        "-Xmx$xmx", "-Xms$xmx",
        "-XX:+AlwaysPreTouch",
        "-XX:+UseG1GC",
        "-XX:MaxGCPauseMillis=100"
    ))

    // Allow filtering via: ./gradlew :vectors-bench:jmh -Pjmh.includes="BenchmarkName"
    if (project.hasProperty("jmh.includes")) {
        includes.set(listOf(project.property("jmh.includes") as String))
    }
    // Exclude multi-hour large-scale benchmarks from the default run. Users opt in by
    // setting -Pivfpq.largeScale=true (typically together with -Pjmh.includes=...).
    if (!largeScale) {
        excludes.addAll(listOf(".*LargeScale.*"))
    }
    // WriteAheadLogS3Benchmark requires an external S3 endpoint (LocalStack or real S3) and
    // is gated by -Pwal.s3.bucket=...; exclude it from the default run otherwise.
    if (!project.hasProperty("wal.s3.bucket")) {
        excludes.addAll(listOf(".*WriteAheadLogS3Benchmark.*"))
    }
    // Forward -Pwal.s3.* properties as -Dwal.s3.* JVM args read by WriteAheadLogS3Benchmark.
    listOf("wal.s3.bucket", "wal.s3.region", "wal.s3.endpoint").forEach { key ->
        (project.findProperty(key) as String?)?.let { jvmArgs.add("-D$key=$it") }
    }
    // Forward GGUF kernel controls so serial/parallel comparisons use the same JMH harness.
    listOf("vectors.gguf.parallel", "vectors.gguf.parallelThreshold").forEach { key ->
        (project.findProperty(key) as String?)?.let { jvmArgs.add("-D$key=$it") }
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
        "-Xmx12g", "-Xms12g",
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
        "bench.ivfpq.m", "bench.ivfpq.rescore",
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
// tieredIvfRecallQps — P1.3 recall-vs-QPS sweep for the tiered IVF substrate
// (BuoyIndex + TieredCluster + HyperDoor + DistributedVectorCollection).
//
// Self-contained: generates a mixture-of-Gaussians synthetic corpus, so it runs without an
// ANN-Benchmarks dataset download. Treat absolute recall/QPS as illustrative; the sweep is
// meaningful as a comparison across nprobe settings on the same substrate.
//
// Usage:
//   ./gradlew :vectors-bench:runTieredIvfRecallQps
//   ./gradlew :vectors-bench:runTieredIvfRecallQps \
//       -Pbench.tieredIvf.corpus=100000 -Pbench.tieredIvf.queries=500 \
//       -Pbench.tieredIvf.nprobe=1,4,8,16,32,64
// ---------------------------------------------------------------------------
tasks.register<JavaExec>("runTieredIvfRecallQps") {
    group = "benchmark"
    description = "Recall-vs-QPS sweep over the tiered IVF substrate (synthetic corpus)"

    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("com.integrallis.vectors.bench.TieredIvfRecallQpsBenchmark")

    jvmArgs(
        "--add-modules", "jdk.incubator.vector",
        "-Xmx4g", "-Xms2g"
    )

    listOf(
        "bench.tieredIvf.dim",
        "bench.tieredIvf.corpus",
        "bench.tieredIvf.queries",
        "bench.tieredIvf.nprobe"
    ).forEach { key ->
        (project.findProperty(key) as String?)?.let { systemProperty(key, it) }
    }

    standardOutput = System.out
    errorOutput    = System.err
}

// ---------------------------------------------------------------------------
// s3Bench — P1.8 turbopuffer object-storage benchmark (Cloudflare R2 / S3 / LocalStack)
//
// Reads R2 creds from the repo-root .env (VECTORS_R2_*); self-skips when absent unless
// -Pbench.s3.localstack=true (with a LocalStack container at -Pbench.s3.endpoint).
//
// Usage:
//   ./gradlew :vectors-bench:s3Bench
//   ./gradlew :vectors-bench:s3Bench -Pbench.s3.n=10000 -Pbench.s3.dim=128 -Pbench.s3.k=64 \
//       -Pbench.s3.q=1000 -Pbench.s3.nprobe=8
//   ./gradlew :vectors-bench:s3Bench -Pbench.s3.localstack=true -Pbench.s3.endpoint=http://localhost:4566
//
// Measures ingest throughput, cold-open latency, cold-vs-warm per-query latency + bytes-fetched
// (via the TieredCluster S3 read-through), tier promotion, and R2 cost. Cleans up its bucket prefix.
// ---------------------------------------------------------------------------
tasks.register<JavaExec>("s3Bench") {
    group = "benchmark"
    description = "Turbopuffer object-storage benchmark against Cloudflare R2 / S3 / LocalStack"

    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("com.integrallis.vectors.bench.TurbopufferS3Benchmark")

    jvmArgs(
        "--add-modules", "jdk.incubator.vector",
        "-Xmx4g", "-Xms4g",
        "-XX:+UseG1GC",
        "-XX:MaxGCPauseMillis=100"
    )

    // Make dotenv lookups deterministic regardless of Gradle's invocation directory.
    systemProperty("dotenv.directory", rootProject.projectDir.absolutePath)

    // Forward -Pbench.s3.* properties as -Dbench.s3.* system properties the harness reads.
    listOf(
        "bench.s3.n", "bench.s3.dim", "bench.s3.k", "bench.s3.q", "bench.s3.nprobe",
        "bench.s3.seed", "bench.s3.localstack", "bench.s3.endpoint", "bench.s3.bucket",
        "bench.s3.iAcceptCost"
    ).forEach { key ->
        (project.findProperty(key) as String?)?.let { systemProperty(key, it) }
    }

    standardOutput = System.out
    errorOutput = System.err

    doFirst {
        project.file("build/results/turbopuffer-s3").mkdirs()
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
        "-Xmx6g", "-Xms6g",
        "-XX:+AlwaysPreTouch",
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
