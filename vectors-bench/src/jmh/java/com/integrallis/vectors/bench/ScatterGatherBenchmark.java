package com.integrallis.vectors.bench;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.Document;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.VectorCollection;
import com.integrallis.vectors.distributed.InProcessNodeDirectory;
import com.integrallis.vectors.distributed.LocalSearchRequest;
import com.integrallis.vectors.distributed.NodeId;
import com.integrallis.vectors.distributed.ScatterGatherExecutor;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

/**
 * JMH Tier-3: end-to-end scatter-gather latency across in-process simulated cluster nodes.
 *
 * <p>Three scenarios from the Phase 3 design gate:
 *
 * <ul>
 *   <li>{@link #scatterGather_3nodes_FLAT} — 3 FLAT nodes × 10K docs, D=128, k=10, target &lt; 5 ms
 *       p99
 *   <li>{@link #scatterGather_10nodes_FLAT} — 10 FLAT nodes × 10K docs, D=128, k=10, target &lt; 10
 *       ms p99
 *   <li>{@link #scatterGather_3nodes_HNSW} — 3 HNSW nodes × 100K docs, D=128, k=10, target &lt; 20
 *       ms p99
 * </ul>
 *
 * <p>Run:
 *
 * <pre>{@code
 * ./gradlew :vectors-bench:jmh -Pjmh.includes=ScatterGatherBenchmark
 * }</pre>
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread)
@Fork(
    value = 1,
    jvmArgsPrepend = {"--add-modules", "jdk.incubator.vector"})
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
public class ScatterGatherBenchmark {

  private static final int DIM = 128;
  private static final int K = 10;
  private static final Duration TIMEOUT = Duration.ofSeconds(5);

  // --- 3-node FLAT (10K docs/node) ---
  private ScatterGatherExecutor executor3Flat;
  private List<LocalSearchRequest> plan3Flat;

  // --- 10-node FLAT (10K docs/node) ---
  private ScatterGatherExecutor executor10Flat;
  private List<LocalSearchRequest> plan10Flat;

  // --- 3-node HNSW (100K docs/node) ---
  private ScatterGatherExecutor executor3Hnsw;
  private List<LocalSearchRequest> plan3Hnsw;

  private float[] query;

  @Setup(Level.Trial)
  public void setUp() {
    Random rng = new Random(42L);
    query = randomUnit(rng, DIM);

    executor3Flat = buildFlatCluster(3, 10_000, rng);
    plan3Flat = broadcastPlan(3, "node-", query);

    executor10Flat = buildFlatCluster(10, 10_000, rng);
    plan10Flat = broadcastPlan(10, "node-", query);

    executor3Hnsw = buildHnswCluster(3, 100_000, rng);
    plan3Hnsw = broadcastPlan(3, "hnode-", query);
  }

  @TearDown(Level.Trial)
  public void tearDown() {
    // Collections are in-memory; GC handles cleanup.
  }

  // ---- Benchmarks ----

  /** 3 FLAT nodes × 10K docs each — target &lt; 5 ms p99. */
  @Benchmark
  public void scatterGather_3nodes_FLAT(Blackhole bh) {
    bh.consume(executor3Flat.execute(plan3Flat, K));
  }

  /** 10 FLAT nodes × 10K docs each — target &lt; 10 ms p99. */
  @Benchmark
  public void scatterGather_10nodes_FLAT(Blackhole bh) {
    bh.consume(executor10Flat.execute(plan10Flat, K));
  }

  /** 3 HNSW nodes × 100K docs each — target &lt; 20 ms p99. */
  @Benchmark
  public void scatterGather_3nodes_HNSW(Blackhole bh) {
    bh.consume(executor3Hnsw.execute(plan3Hnsw, K));
  }

  // ---- Setup helpers ----

  private ScatterGatherExecutor buildFlatCluster(int numNodes, int docsPerNode, Random rng) {
    InProcessNodeDirectory.Builder dir = InProcessNodeDirectory.builder();
    for (int n = 0; n < numNodes; n++) {
      VectorCollection col =
          VectorCollection.builder()
              .dimension(DIM)
              .metric(SimilarityFunction.DOT_PRODUCT)
              .indexType(IndexType.FLAT)
              .build();
      loadDocs(col, docsPerNode, "n" + n + "-", rng);
      dir.register(new NodeId("node-" + n), col);
    }
    return new ScatterGatherExecutor(dir.build(), TIMEOUT);
  }

  private ScatterGatherExecutor buildHnswCluster(int numNodes, int docsPerNode, Random rng) {
    InProcessNodeDirectory.Builder dir = InProcessNodeDirectory.builder();
    for (int n = 0; n < numNodes; n++) {
      VectorCollection col =
          VectorCollection.builder()
              .dimension(DIM)
              .metric(SimilarityFunction.DOT_PRODUCT)
              .indexType(IndexType.HNSW)
              .build();
      loadDocs(col, docsPerNode, "h" + n + "-", rng);
      dir.register(new NodeId("hnode-" + n), col);
    }
    return new ScatterGatherExecutor(dir.build(), TIMEOUT);
  }

  private void loadDocs(VectorCollection col, int count, String prefix, Random rng) {
    for (int i = 0; i < count; i++) {
      col.add(Document.of(prefix + i, randomUnit(rng, DIM)));
    }
    col.commit();
  }

  private List<LocalSearchRequest> broadcastPlan(int numNodes, String nodePrefix, float[] q) {
    List<LocalSearchRequest> plan = new ArrayList<>(numNodes);
    for (int n = 0; n < numNodes; n++) {
      plan.add(
          new LocalSearchRequest(new NodeId(nodePrefix + n), q, new int[0], K, -Float.MAX_VALUE));
    }
    return plan;
  }

  private static float[] randomUnit(Random rng, int dim) {
    float[] v = new float[dim];
    float norm = 0f;
    for (int i = 0; i < dim; i++) {
      v[i] = rng.nextFloat() * 2 - 1;
      norm += v[i] * v[i];
    }
    norm = (float) Math.sqrt(norm);
    for (int i = 0; i < dim; i++) v[i] /= norm;
    return v;
  }
}
