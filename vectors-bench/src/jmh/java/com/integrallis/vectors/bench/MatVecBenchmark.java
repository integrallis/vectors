package com.integrallis.vectors.bench;

import com.integrallis.vectors.core.VectorUtil;
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
 * JMH benchmark for the fused SIMD GEMV kernel ({@code matVecDot} / {@code matVecSquaredL2}).
 *
 * <p>Measures throughput of {@link VectorUtil#batchDotProduct} and {@link
 * VectorUtil#batchSquaredL2} (which now delegate to the 4-row-unrolled Panama GEMV) across the
 * centroid-count × dimension pairs typical in IVF workloads.
 *
 * <p>Run:
 *
 * <pre>{@code
 * ./gradlew :vectors-bench:jmh -Pjmh.includes=MatVecBenchmark
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
public class MatVecBenchmark {

  /** Number of centroid rows (IVF k). */
  @Param({"16", "64", "256", "1024"})
  int k;

  /** Vector dimension. */
  @Param({"128", "768", "1536"})
  int dim;

  private float[] query;
  private float[][] matrix;
  private float[] out;

  @Setup(Level.Trial)
  public void setUp() {
    Random rng = new Random(42L);
    query = new float[dim];
    for (int i = 0; i < dim; i++) query[i] = rng.nextFloat() * 2f - 1f;

    matrix = new float[k][dim];
    for (float[] row : matrix) for (int i = 0; i < dim; i++) row[i] = rng.nextFloat() * 2f - 1f;

    out = new float[k];
  }

  /**
   * Fused 4-row SIMD dot product (query loads once per SIMD chunk, reused across 4 centroid rows).
   */
  @Benchmark
  public void matVecDot_fused(Blackhole bh) {
    VectorUtil.batchDotProduct(query, matrix, out);
    bh.consume(out);
  }

  /** Fused 4-row SIMD squared L2 distance (same kernel pattern as {@code matVecDot_fused}). */
  @Benchmark
  public void matVecSquaredL2_fused(Blackhole bh) {
    VectorUtil.batchSquaredL2(query, matrix, out);
    bh.consume(out);
  }
}
