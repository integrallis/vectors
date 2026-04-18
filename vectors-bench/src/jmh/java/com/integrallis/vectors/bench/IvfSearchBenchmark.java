package com.integrallis.vectors.bench;

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
 * JMH benchmark for IVF nprobe sweep (G2 gap closure).
 *
 * <p>Measures the recall-QPS tradeoff of {@link IvfIndex#search} as {@code nprobe} increases.
 * Higher {@code nprobe} probes more clusters, improving recall at the cost of latency. The optimal
 * operating point is typically {@code nprobe} ≈ √k for good recall at sub-millisecond latency.
 *
 * <p>Run:
 *
 * <pre>{@code
 * ./gradlew :vectors-bench:jmh -Pjmh.includes=IvfSearchBenchmark
 * }</pre>
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Thread)
@Fork(
    value = 1,
    jvmArgsPrepend = {"--add-modules", "jdk.incubator.vector"})
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
public class IvfSearchBenchmark {

  /** Dataset size (corpus vectors). */
  @Param({"10000", "100000"})
  int n;

  /** Embedding dimension. */
  @Param({"128", "768"})
  int dim;

  /**
   * Number of clusters to probe per query. Lower = faster but lower recall; higher = slower but
   * closer to exhaustive search. The nprobe sweep is the primary IVF quality-throughput knob.
   */
  @Param({"1", "2", "4", "8", "16", "32"})
  int nprobe;

  private IvfIndex index;
  private float[] query;

  @Setup(Level.Trial)
  public void setUp() {
    Random rng = new Random(77L);
    float[][] corpus = new float[n][dim];
    for (float[] row : corpus) {
      for (int d = 0; d < dim; d++) row[d] = rng.nextFloat() * 2f - 1f;
    }
    query = corpus[rng.nextInt(n)].clone();

    int k = (int) Math.max(4, Math.ceil(Math.sqrt(n)));
    IvfBuildParams params = new IvfBuildParams(k, 50, 0.2f, true, 42L, 0);
    index =
        IvfIndex.build(
            corpus, null, com.integrallis.vectors.core.SimilarityFunction.EUCLIDEAN, params);
  }

  /**
   * Searches the IVF index with the current {@code nprobe} value. The JMH parametrization sweeps
   * nprobe from 1 to 32, producing the full recall-QPS tradeoff curve for the current dataset and
   * dimension combination.
   */
  @Benchmark
  public IvfSearchResult searchIvf(Blackhole bh) {
    IvfSearchRequest req = IvfSearchRequest.of(query, 10, nprobe);
    IvfSearchResult result = index.search(req);
    bh.consume(result);
    return result;
  }
}
