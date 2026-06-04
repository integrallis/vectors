/*
 * Copyright 2025-2026 Integrallis Software, LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.integrallis.vectors.bench;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.ivf.ClusterSplitter;
import com.integrallis.vectors.ivf.DistributedVectorCollection;
import com.integrallis.vectors.ivf.IvfBuildParams;
import com.integrallis.vectors.ivf.IvfHit;
import com.integrallis.vectors.ivf.TierPolicy;
import com.integrallis.vectors.storage.backend.S3StorageBackend;
import java.io.IOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.UUID;
import software.amazon.awssdk.auth.credentials.AwsBasicCredentials;
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.model.CreateBucketRequest;
import software.amazon.awssdk.services.s3.model.S3Exception;

/**
 * P1.8 — the "JVM turbopuffer" headline benchmark: object-storage-native vector serving on
 * Cloudflare R2 (or AWS S3 / LocalStack). Measures the real cost of serving vectors from object
 * storage: ingest throughput, cold-open latency, and — via the {@code TieredCluster} S3
 * read-through — true per-query cold-cache vs warm-cache latency and bytes-fetched-per-query, plus
 * a tier (T0→T3) promotion observation under a skewed workload.
 *
 * <p><b>Honesty:</b> the read-through path genuinely fetches each probed cluster from R2 on the
 * first probe (cold) and caches it in heap (warm). The current design still keeps the full vector
 * set heap-resident, so this measures the object-storage <em>read latency/bytes</em>, not the
 * memory savings of a fully paged index (future work). Reads R2 creds from the repo-root {@code
 * .env} via {@link R2Config}; self-skips when neither R2 nor {@code -Dbench.s3.localstack=true} is
 * configured.
 *
 * <pre>{@code
 * ./gradlew :vectors-bench:s3Bench
 * ./gradlew :vectors-bench:s3Bench -Pbench.s3.n=10000 -Pbench.s3.q=1000 -Pbench.s3.k=64
 * }</pre>
 */
public final class TurbopufferS3Benchmark {

  private static final SimilarityFunction METRIC = SimilarityFunction.COSINE;
  private static final int MAX_N = 200_000;
  private static final int MAX_Q = 20_000;

  private TurbopufferS3Benchmark() {}

  public static void main(String[] args) throws Exception {
    int n = intProp("bench.s3.n", 10_000);
    int dim = intProp("bench.s3.dim", 128);
    int k = intProp("bench.s3.k", 64);
    int q = intProp("bench.s3.q", 1_000);
    int nprobe = intProp("bench.s3.nprobe", 8);
    long seed = Long.getLong("bench.s3.seed", 42L);
    boolean iAcceptCost = Boolean.getBoolean("bench.s3.iAcceptCost");
    boolean useLocalstack = Boolean.getBoolean("bench.s3.localstack");

    if ((n > MAX_N || q > MAX_Q) && !iAcceptCost) {
      System.out.printf(
          "Refusing to run: n=%d (max %d) or q=%d (max %d) exceeds the cost-guard ceiling.%n"
              + "Re-run with -Dbench.s3.iAcceptCost=true if you really want this volume.%n",
          n, MAX_N, q, MAX_Q);
      return;
    }

    S3Client s3;
    String bucket;
    String where;
    R2Config cfg = R2Config.fromEnv();
    if (cfg != null) {
      s3 =
          S3Client.builder()
              .region(Region.of(cfg.region))
              .endpointOverride(URI.create(cfg.endpoint))
              .credentialsProvider(
                  StaticCredentialsProvider.create(
                      AwsBasicCredentials.create(cfg.accessKey, cfg.secretKey)))
              .forcePathStyle(true)
              .build();
      bucket = cfg.bucket;
      where = "Cloudflare R2 (" + cfg.endpoint + ", bucket=" + bucket + ")";
    } else if (useLocalstack) {
      String endpoint = System.getProperty("bench.s3.endpoint", "http://localhost:4566");
      s3 =
          S3Client.builder()
              .region(Region.US_EAST_1)
              .endpointOverride(URI.create(endpoint))
              .credentialsProvider(
                  StaticCredentialsProvider.create(AwsBasicCredentials.create("test", "test")))
              .forcePathStyle(true)
              .build();
      bucket = System.getProperty("bench.s3.bucket", "vectors-bench");
      createBucketIfMissing(s3, bucket);
      where = "LocalStack (" + endpoint + ", bucket=" + bucket + ")";
    } else {
      System.out.println(
          "P1.8 skipped: no R2 creds in .env (VECTORS_R2_*) and -Dbench.s3.localstack not set.");
      return;
    }

    String prefix = "p18-turbopuffer/" + UUID.randomUUID() + "/";
    CountingStorageBackend backend =
        new CountingStorageBackend(
            new PrefixedStorageBackend(new S3StorageBackend(s3, bucket), prefix));
    Path walDir = Files.createTempDirectory("p18-wal-");

    System.out.println("=== P1.8 turbopuffer S3/R2 benchmark ===");
    System.out.printf("backend=%s%n", where);
    System.out.printf(
        "n=%d dim=%d clusters=%d nprobe=%d queries=%d seed=%d%n", n, dim, k, nprobe, q, seed);
    System.out.printf("run prefix=%s  walDir=%s%n%n", prefix, walDir);

    // Cleanup runs exactly once — on normal exit via the finally, or on Ctrl-C via the shutdown
    // hook — so a killed run does not orphan objects (and cost) and a normal run does not
    // double-run
    // against an already-closed client.
    java.util.concurrent.atomic.AtomicBoolean cleaned =
        new java.util.concurrent.atomic.AtomicBoolean();
    Runtime.getRuntime()
        .addShutdownHook(
            new Thread(
                () -> {
                  if (cleaned.compareAndSet(false, true)) {
                    bestEffortCleanup(backend, prefix, walDir);
                  }
                }));

    try {
      runPhases(backend, walDir, n, dim, k, q, nprobe, seed);
    } finally {
      if (cleaned.compareAndSet(false, true)) {
        cleanup(backend, prefix, walDir);
      }
      s3.close();
    }
  }

  private static void runPhases(
      CountingStorageBackend backend,
      Path walDir,
      int n,
      int dim,
      int k,
      int q,
      int nprobe,
      long seed)
      throws IOException {
    Random rng = new Random(seed);
    float[][] vectors = new float[n][];
    String[] ids = new String[n];
    for (int i = 0; i < n; i++) {
      vectors[i] = unitVec(rng, dim);
      ids[i] = "doc-" + i;
    }
    float[][] queries = new float[q][];
    for (int i = 0; i < q; i++) {
      queries[i] = unitVec(rng, dim);
    }

    // ── INGEST ────────────────────────────────────────────────────────────────
    IvfBuildParams params = new IvfBuildParams(k, 30, 0f, false, seed, 0);
    ClusterSplitter splitter = new ClusterSplitter(10_000, 30, seed);
    TierPolicy policy = new TierPolicy(5, 2);
    backend.reset();
    long t0 = System.nanoTime();
    DistributedVectorCollection col =
        DistributedVectorCollection.build(
            vectors, ids, METRIC, params, splitter, policy, walDir, backend);
    double ingestSec = (System.nanoTime() - t0) / 1e9;
    long putCalls = backend.putCalls();
    long bytesPut = backend.bytesPut();
    System.out.printf(
        "INGEST: %d vectors in %.2fs = %.0f vec/s | PUTs=%d bytesPut=%.2f MB%n",
        n, ingestSec, n / ingestSec, putCalls, bytesPut / 1e6);
    col.close();

    // ── COLD OPEN (one ROUTING_KEY GET from R2 + local WAL replay) ─────────────
    backend.reset();
    t0 = System.nanoTime();
    DistributedVectorCollection reopened =
        DistributedVectorCollection.open(walDir, METRIC, policy, backend);
    double openMs = (System.nanoTime() - t0) / 1e6;
    System.out.printf(
        "COLD OPEN: %.2f ms | GETs=%d bytesFetched=%.1f KB (routing index + WAL replay)%n",
        openMs, backend.getCalls(), backend.bytesFetched() / 1e3);

    // ── COLD vs WARM QUERY (read-through fetches probed clusters from R2) ───────
    reopened.enableReadThrough();
    int sample = Math.min(q, 200);

    double[] coldMs = new double[sample];
    long coldBytes = 0;
    long coldGets = 0;
    for (int i = 0; i < sample; i++) {
      reopened.dropReadThroughCaches(); // force every probed cluster cold for this query
      backend.reset();
      t0 = System.nanoTime();
      List<IvfHit> hits = reopened.search(queries[i], 10, nprobe);
      coldMs[i] = (System.nanoTime() - t0) / 1e6;
      coldBytes += backend.bytesFetched();
      coldGets += backend.getCalls();
      if (hits.isEmpty() && n > 0) {
        throw new IllegalStateException("cold query returned no hits — read-through misconfigured");
      }
    }

    // Warm: caches now populated by the last query; re-run without dropping.
    double[] warmMs = new double[sample];
    long warmBytes = 0;
    // Prime all caches once so every probed cluster is warm.
    for (int i = 0; i < sample; i++) {
      reopened.search(queries[i], 10, nprobe);
    }
    for (int i = 0; i < sample; i++) {
      backend.reset();
      t0 = System.nanoTime();
      reopened.search(queries[i], 10, nprobe);
      warmMs[i] = (System.nanoTime() - t0) / 1e6;
      warmBytes += backend.bytesFetched();
    }

    System.out.printf(
        "COLD  query: p50=%.3f ms p99=%.3f ms | bytes/query=%.1f KB GETs/query=%.1f%n",
        pct(coldMs, 50),
        pct(coldMs, 99),
        (coldBytes / (double) sample) / 1e3,
        coldGets / (double) sample);
    System.out.printf(
        "WARM  query: p50=%.3f ms p99=%.3f ms | bytes/query=%.1f KB (served from heap cache)%n",
        pct(warmMs, 50), pct(warmMs, 99), (warmBytes / (double) sample) / 1e3);
    System.out.printf(
        "cold/warm speedup: %.1fx (p50)%n%n", pct(coldMs, 50) / Math.max(pct(warmMs, 50), 1e-9));

    reopened.close();

    // ── TIER PROMOTION (skewed/Zipf workload) ──────────────────────────────────
    // Reopen fresh so access counts reset to 0, then drive a concentrated Zipf workload over a
    // small query pool with small per-round batches: hot clusters cross the T1 threshold first and
    // T1 count climbs gradually with the skew (a broad workload would saturate all clusters at
    // once).
    DistributedVectorCollection promo =
        DistributedVectorCollection.open(walDir, METRIC, policy, backend);
    int pool = Math.min(q, 64);
    double[] zipf = zipfWeights(pool, 1.0);
    int rounds = 12;
    int batch = 8;
    System.out.println(
        "PROMOTION (fresh reopen, Zipf s=1.0 over " + pool + " queries, TierPolicy(5,2)):");
    for (int r = 1; r <= rounds; r++) {
      for (int b = 0; b < batch; b++) {
        promo.search(queries[sampleZipf(zipf, rng)], 10, nprobe);
      }
      try {
        promo.commit(); // applyTierPolicy promotes hot clusters to T1 (SQ8)
      } catch (IOException e) {
        System.out.println("  commit failed: " + e.getMessage());
      }
      System.out.printf("  round %2d: T1=%d / %d clusters%n", r, promo.t1ClusterCount(), k);
    }
    promo.close();
    System.out.println();

    // ── COST (R2 pricing: $0.015/GB-mo storage, $0 egress) ─────────────────────
    long storageBytes = totalStoredBytes(backend);
    double storageGb = storageBytes / 1e9;
    double monthlyStorage = storageGb * 0.015;
    System.out.printf(
        "COST: stored=%.2f MB across %d objects | storage=$%.6f/mo = $%.3e per vector/mo%n",
        storageBytes / 1e6, backend.list("").size(), monthlyStorage, monthlyStorage / n);
    System.out.printf(
        "      warm queries fetch 0 bytes -> $0 egress (R2 free egress is the headline vs S3)%n");
  }

  // ─── helpers ───────────────────────────────────────────────────────────────

  private static float[] unitVec(Random rng, int dim) {
    float[] v = new float[dim];
    double norm = 0;
    for (int i = 0; i < dim; i++) {
      v[i] = rng.nextFloat() * 2f - 1f;
      norm += (double) v[i] * v[i];
    }
    float inv = (float) (1.0 / Math.sqrt(norm));
    for (int i = 0; i < dim; i++) {
      v[i] *= inv;
    }
    return v;
  }

  private static double pct(double[] xs, int p) {
    double[] s = xs.clone();
    Arrays.sort(s);
    int idx = (int) Math.ceil(p / 100.0 * s.length) - 1;
    return s[Math.max(0, Math.min(s.length - 1, idx))];
  }

  /** Precomputes Zipf(s) cumulative weights over [0, q). */
  private static double[] zipfWeights(int q, double s) {
    double[] cum = new double[q];
    double sum = 0;
    for (int i = 0; i < q; i++) {
      sum += 1.0 / Math.pow(i + 1, s);
      cum[i] = sum;
    }
    for (int i = 0; i < q; i++) {
      cum[i] /= sum;
    }
    return cum;
  }

  private static int sampleZipf(double[] cum, Random rng) {
    double u = rng.nextDouble();
    int lo = 0;
    int hi = cum.length - 1;
    while (lo < hi) {
      int mid = (lo + hi) >>> 1;
      if (cum[mid] < u) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    return lo;
  }

  private static long totalStoredBytes(CountingStorageBackend backend) throws IOException {
    long total = 0;
    for (String key : backend.list("")) {
      byte[] v = backend.get(key);
      if (v != null) {
        total += v.length;
      }
    }
    return total;
  }

  private static void createBucketIfMissing(S3Client s3, String bucket) {
    try {
      s3.createBucket(CreateBucketRequest.builder().bucket(bucket).build());
    } catch (S3Exception e) {
      // BucketAlreadyOwnedByYou / BucketAlreadyExists — fine.
    }
  }

  private static void cleanup(CountingStorageBackend backend, String prefix, Path walDir)
      throws IOException {
    int deleted = 0;
    for (String key : backend.list("")) {
      backend.delete(key);
      deleted++;
    }
    List<String> remaining = backend.list("");
    if (!remaining.isEmpty()) {
      System.out.printf(
          "WARNING: %d objects remain under %s — delete manually.%n", remaining.size(), prefix);
    } else {
      System.out.printf("CLEANUP: deleted %d objects under %s; bucket clean.%n", deleted, prefix);
    }
    deleteRecursive(walDir);
  }

  private static void bestEffortCleanup(
      CountingStorageBackend backend, String prefix, Path walDir) {
    try {
      cleanup(backend, prefix, walDir);
    } catch (Exception e) {
      System.out.printf(
          "Shutdown cleanup failed for prefix %s: %s (delete manually).%n", prefix, e.getMessage());
    }
  }

  private static void deleteRecursive(Path dir) {
    try (var walk = Files.walk(dir)) {
      walk.sorted((a, b) -> b.getNameCount() - a.getNameCount())
          .forEach(
              p -> {
                try {
                  Files.deleteIfExists(p);
                } catch (IOException ignored) {
                  // best effort
                }
              });
    } catch (IOException ignored) {
      // best effort
    }
  }

  private static int intProp(String key, int def) {
    String v = System.getProperty(key);
    return v == null ? def : Integer.parseInt(v.trim());
  }
}
