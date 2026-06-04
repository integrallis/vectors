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
 * JMH Tier-1 microbenchmarks for SIMD vs. scalar distance kernels — the P1.7 "scalar baseline"
 * suite.
 *
 * <p>Every kernel is measured on both the auto-selected SIMD provider ({@link
 * VectorizationProvider#getInstance()}) and the scalar provider ({@link
 * VectorizationProvider#newScalarProvider()}) within a single JVM, so the {@code simd*} / {@code
 * scalar*} pair is an apples-to-apples A/B (same JIT warmup, same heap). Dividing the SIMD score by
 * the scalar score yields the SIMD speed-up — the number that quantifies how much is lost if a JDK
 * upgrade silently disables Panama and the runtime falls back to {@code ScalarVectorUtilSupport}
 * (see {@link VectorizationProvider#getPanamaFailure()}). The curated ratios live in {@code
 * vectors-bench/jmh-results/scalar-baseline.txt}.
 *
 * <p>Run via:
 *
 * <pre>{@code
 * ./gradlew :vectors-bench:jmh -Pjmh.includes=DistanceKernelBenchmark
 * }</pre>
 *
 * <p>Each benchmark method is a single kernel call; the {@code @Param dim} axis sweeps small to
 * large embeddings so both cache-resident and memory-bandwidth-limited regimes are covered: tiny
 * (4-16), standard (32-256), LLM embeddings (384-768), and large (1024-2048).
 *
 * <p>Kernel × type coverage: dot / L2 / cosine / cosineNormalized for FLOAT32 and INT8, plus
 * hamming for BINARY. {@code cosineNormalized} is synthesized as a dot product over unit-normalized
 * inputs (there is no dedicated SPI method, and {@code a}/{@code b} are L2-normalized in {@link
 * #setUp()}). INT8 and BINARY have no {@code cosineNormalized} cell — INT8 has no normalize path
 * and BINARY's distance is hamming — so those are intentionally omitted rather than invented.
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

  // cosineNormalized has no dedicated SPI method. For unit-length vectors the cosine is exactly the
  // dot product, and setUp() L2-normalizes a/b, so the honest measurement is a dot product over
  // already-normalized inputs (the work a caller skips when it knows its vectors are unit-length).
  @Benchmark
  public float simdCosineNormalized() {
    return simd.dotProduct(a, b);
  }

  @Benchmark
  public float[] simdL2Normalize() {
    return simd.l2normalize(unnormalized, false);
  }

  // --- Scalar float kernels (P1.7 baseline) ---

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
  public float scalarCosineNormalized() {
    return scalar.dotProduct(a, b);
  }

  @Benchmark
  public float[] scalarL2Normalize() {
    return scalar.l2normalize(unnormalized, false);
  }

  // --- Byte (INT8) kernels, used by SQ8 scoring; SIMD vs. scalar baseline ---

  @Benchmark
  public int byteDotProduct() {
    return simd.dotProduct(ba, bb);
  }

  @Benchmark
  public int scalarByteDotProduct() {
    return scalar.dotProduct(ba, bb);
  }

  @Benchmark
  public int byteL2Distance() {
    return simd.squareDistance(ba, bb);
  }

  @Benchmark
  public int scalarByteL2Distance() {
    return scalar.squareDistance(ba, bb);
  }

  @Benchmark
  public float byteCosine() {
    return simd.cosine(ba, bb);
  }

  @Benchmark
  public float scalarByteCosine() {
    return scalar.cosine(ba, bb);
  }

  // --- Binary (BINARY) kernels: hamming distance for BQ; SIMD vs. scalar baseline ---
  // BINARY has no dot/L2/cosine/cosineNormalized kernel — hamming is the binary distance, so it is
  // the sole BINARY cell of the P1.7 matrix.

  @Benchmark
  public int simdHammingDistance() {
    return simd.hammingDistance(binaryA, binaryB);
  }

  @Benchmark
  public int scalarHammingDistance() {
    return scalar.hammingDistance(binaryA, binaryB);
  }
}
