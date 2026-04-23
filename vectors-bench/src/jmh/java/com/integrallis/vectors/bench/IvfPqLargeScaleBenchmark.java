package com.integrallis.vectors.bench;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.ivf.IvfBuildParams;
import com.integrallis.vectors.ivf.IvfIndex;
import com.integrallis.vectors.ivf.IvfSearchRequest;
import com.integrallis.vectors.ivf.IvfSearchResult;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

/**
 * Large-scale IVF-PQ sweep at {@code n = 1M} to verify sub-linear search scaling.
 *
 * <p>This benchmark is <strong>gated</strong>: the default {@code :vectors-bench:jmh} task excludes
 * classes matching {@code .*LargeScale.*}. Run it explicitly with:
 *
 * <pre>{@code
 * ./gradlew :vectors-bench:jmh \
 *     -Pjmh.includes=IvfPqLargeScaleBenchmark \
 *     -Pivfpq.largeScale=true
 * }</pre>
 *
 * <p>The {@code -Pivfpq.largeScale=true} flag both un-gates the benchmark and bumps the JMH JVM
 * heap to {@code -Xmx16g} (see {@code vectors-bench/build.gradle.kts}).
 *
 * <p><strong>Runtime budget.</strong> {@code @Setup(Trial)} builds the full IVF-PQ index from
 * scratch for every parameter combination; K-Means at {@code n = 1M} with {@code nlist = √n ≈ 1000}
 * dominates total wall clock. Rough extrapolation from the measured 100K numbers: ~5–10 min per
 * {@code dim=128} cell and ~45–90 min per {@code dim=768} cell. With four cells (2 dim × 2
 * rescoreFactor) plan for 2–6 h end-to-end. Search iterations themselves complete in &lt;100 ms.
 *
 * <p>{@code maxIter} is reduced to 10 (from the default 50) because the benchmark goal is to
 * measure search latency at scale, not k-means convergence quality.
 *
 * <p>Parameter axes mirror {@link IvfPqSearchBenchmark} so results can be cross-referenced against
 * the steady-state 10K/100K numbers: {@code pqSubspaces = 16}, {@code rescoreFactor in {1, 4}},
 * {@code dim in {128, 768}}. The sweep confirms that search scales with {@code √n} (via {@code
 * nprobe = √k = n^{1/4}}) rather than linearly.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread)
@Fork(
    value = 1,
    jvmArgsPrepend = {"--add-modules", "jdk.incubator.vector"})
@Warmup(iterations = 1, time = 2)
@Measurement(iterations = 3, time = 2)
public class IvfPqLargeScaleBenchmark {

  /** Corpus size. 1M is the practical single-machine ceiling under the default {@code -Xmx16g}. */
  @Param({"1000000"})
  int n;

  /** Embedding dimension. Both values divide {@code pqSubspaces = 16} cleanly. */
  @Param({"128", "768"})
  int dim;

  /** PQ subspaces (M). Fixed — axis here is {@code n}, not quantiser width. */
  @Param({"16"})
  int pqSubspaces;

  /** Rescore factor: {@code 1} = pure ADC; {@code 4} = wide-heap re-ranked on full precision. */
  @Param({"1", "4"})
  int rescoreFactor;

  private IvfIndex index;
  private float[] query;
  private int nprobe;

  @Setup(Level.Trial)
  public void setUp() {
    Random rng = new Random(77L);
    float[][] corpus = new float[n][dim];
    for (float[] row : corpus) {
      for (int d = 0; d < dim; d++) row[d] = rng.nextFloat() * 2f - 1f;
    }
    query = corpus[rng.nextInt(n)].clone();

    int k = (int) Math.max(4, Math.ceil(Math.sqrt(n)));
    // Probe √k clusters — the textbook IVF operating point at which recall plateaus.
    this.nprobe = (int) Math.max(1, Math.ceil(Math.sqrt(k)));
    // maxIter=10: benchmark measures search scaling, not k-means convergence quality.
    IvfBuildParams base = new IvfBuildParams(k, 10, 0f, false, 42L, 0);
    IvfBuildParams params = base.withPq(pqSubspaces);
    index = IvfIndex.build(corpus, null, SimilarityFunction.EUCLIDEAN, params);
  }

  /**
   * Searches the IVF-PQ index. Each invocation rebuilds the per-probe ADC tables, scores {@code
   * nprobe} partitions, and (when {@code rescoreFactor > 1}) re-ranks the wide candidate heap
   * against full-precision vectors.
   */
  @Benchmark
  public IvfSearchResult searchIvfPq(Blackhole bh) {
    IvfSearchRequest req = IvfSearchRequest.of(query, 10, nprobe, rescoreFactor);
    IvfSearchResult result = index.search(req);
    bh.consume(result);
    return result;
  }
}
