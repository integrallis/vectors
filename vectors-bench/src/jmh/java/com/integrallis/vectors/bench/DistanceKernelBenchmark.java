package com.integrallis.vectors.bench;

import com.integrallis.vectors.core.VectorUtilSupport;
import com.integrallis.vectors.core.VectorizationProvider;
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

/**
 * JMH Tier-1 microbenchmarks for SIMD vs. scalar distance kernels.
 *
 * <p>Measures raw throughput of each kernel at representative embedding dimensions. Run via:
 *
 * <pre>{@code
 * ./gradlew :vectors-bench:jmh -Pjmh.includes=DistanceKernelBenchmark
 * }</pre>
 *
 * <p>Each benchmark method is a single kernel call; the {@code @Param dim} axis sweeps 32 →
 * 1536-dimensional embeddings so both cache-resident and memory-bandwidth-limited regimes are
 * covered.
 */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Thread)
@Fork(
    value = 1,
    jvmArgsPrepend = {"--add-modules", "jdk.incubator.vector"})
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
public class DistanceKernelBenchmark {

  @Param({"32", "128", "256", "768", "1536"})
  public int dim;

  // float vectors
  private float[] a;
  private float[] b;

  // byte vectors
  private byte[] ba;
  private byte[] bb;

  // providers
  private VectorUtilSupport simd;
  private VectorUtilSupport scalar;

  @Setup(Level.Trial)
  public void setUp() {
    Random rng = new Random(42L);
    a = new float[dim];
    b = new float[dim];
    double normA = 0, normB = 0;
    for (int i = 0; i < dim; i++) {
      a[i] = rng.nextFloat() * 2f - 1f;
      b[i] = rng.nextFloat() * 2f - 1f;
      normA += a[i] * (double) a[i];
      normB += b[i] * (double) b[i];
    }
    // Normalize to unit sphere — produces valid cosine inputs
    float sqrtA = (float) Math.sqrt(normA);
    float sqrtB = (float) Math.sqrt(normB);
    for (int i = 0; i < dim; i++) {
      a[i] /= sqrtA;
      b[i] /= sqrtB;
    }

    ba = new byte[dim];
    bb = new byte[dim];
    rng.nextBytes(ba);
    rng.nextBytes(bb);

    simd = VectorizationProvider.getInstance();
    scalar = VectorizationProvider.newScalarProvider();
  }

  // --- SIMD float kernels ---

  @Benchmark
  public float simdDotProduct() {
    return simd.dotProduct(a, b);
  }

  @Benchmark
  public float simdL2Distance() {
    return simd.squareDistance(a, b);
  }

  @Benchmark
  public float simdCosine() {
    return simd.cosine(a, b);
  }

  // --- Scalar float kernels (baseline) ---

  @Benchmark
  public float scalarDotProduct() {
    return scalar.dotProduct(a, b);
  }

  @Benchmark
  public float scalarL2Distance() {
    return scalar.squareDistance(a, b);
  }

  @Benchmark
  public float scalarCosine() {
    return scalar.cosine(a, b);
  }

  // --- Byte kernels (used by SQ8 scoring) ---

  @Benchmark
  public int byteDotProduct() {
    return simd.dotProduct(ba, bb);
  }

  @Benchmark
  public int byteL2Distance() {
    return simd.squareDistance(ba, bb);
  }

  @Benchmark
  public float byteCosine() {
    return simd.cosine(ba, bb);
  }
}
