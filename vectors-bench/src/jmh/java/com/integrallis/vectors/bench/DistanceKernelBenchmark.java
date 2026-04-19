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
 * <p>Each benchmark method is a single kernel call; the {@code @Param dim} axis sweeps small to
 * large embeddings so both cache-resident and memory-bandwidth-limited regimes are covered.
 *
 * <p>Expanded from the original 5 dimensions to cover the full range: tiny (4-16), standard
 * (32-256), LLM embeddings (384-768), and large (1024-2048). Added l2normalize and hamming distance
 * kernels.
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

  @Param({"4", "8", "16", "32", "64", "128", "256", "384", "512", "768", "1024", "1536", "2048"})
  public int dim;

  // float vectors
  private float[] a;
  private float[] b;

  // unnormalized float vector for l2normalize benchmarks
  private float[] unnormalized;

  // byte vectors
  private byte[] ba;
  private byte[] bb;

  // binary vectors (long[]) for hamming distance
  private long[] binaryA;
  private long[] binaryB;

  // providers
  private VectorUtilSupport simd;
  private VectorUtilSupport scalar;

  @Setup(Level.Trial)
  public void setUp() {
    Random rng = new Random(42L);
    a = new float[dim];
    b = new float[dim];
    unnormalized = new float[dim];
    double normA = 0, normB = 0;
    for (int i = 0; i < dim; i++) {
      a[i] = rng.nextFloat() * 2f - 1f;
      b[i] = rng.nextFloat() * 2f - 1f;
      unnormalized[i] = rng.nextFloat() * 10f - 5f;
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

    // Binary vectors for hamming distance: ceil(dim / 64) longs
    int longs = (dim + 63) / 64;
    binaryA = new long[longs];
    binaryB = new long[longs];
    for (int i = 0; i < longs; i++) {
      binaryA[i] = rng.nextLong();
      binaryB[i] = rng.nextLong();
    }

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

  @Benchmark
  public float[] simdL2Normalize() {
    return simd.l2normalize(unnormalized, false);
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

  @Benchmark
  public float[] scalarL2Normalize() {
    return scalar.l2normalize(unnormalized, false);
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

  // --- Binary kernels (hamming distance for BQ) ---

  @Benchmark
  public int simdHammingDistance() {
    return simd.hammingDistance(binaryA, binaryB);
  }

  @Benchmark
  public int scalarHammingDistance() {
    return scalar.hammingDistance(binaryA, binaryB);
  }
}
