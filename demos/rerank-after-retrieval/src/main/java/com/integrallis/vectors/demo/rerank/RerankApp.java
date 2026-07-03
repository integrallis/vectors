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
package com.integrallis.vectors.demo.rerank;

import com.integrallis.vectors.core.TopK;
import com.integrallis.vectors.core.VectorUtil;
import java.util.Arrays;
import java.util.Random;

/**
 * SIMD rescore demo.
 *
 * <p>Remote vector databases (Pinecone, Qdrant, managed Elasticsearch) return ~100–1000 candidates.
 * Reranking with a higher-fidelity metric is latency-critical and benefits from tight loops with
 * float32 precision.
 *
 * <p>This demo:
 *
 * <ol>
 *   <li>Fabricates 1000 "remote candidates" (id + vector) ranked by a noisy dot-product
 *   <li>Rescrores them with {@link VectorUtil#batchDotProduct} in a single SIMD sweep
 *   <li>Picks the top-{@value #TOP_K} with {@link TopK#select}
 *   <li>Reports a cold first-call latency and a steady-state (median-of-N) latency
 * </ol>
 *
 * <p><b>About the numbers.</b> The Panama Vector API kernel needs several thousand invocations
 * before the JIT promotes it to C2 steady state; the "cold" sample measured on the very first call
 * runs in the interpreter or at C1 and is typically 50–100× slower than the warm sample. This demo
 * reports both so that cost is visible. For rigorous, publication-grade numbers, run the JMH suite:
 *
 * <pre>
 *   ./gradlew :vectors-bench:jmh -Pjmh.includes=MatVecBenchmark
 * </pre>
 *
 * <p>Run:
 *
 * <pre>
 *   ./gradlew :demos:rerank-after-retrieval:run
 * </pre>
 */
public final class RerankApp {

  private static final int DIMENSION = 384;
  private static final int CANDIDATES = 1000;
  private static final int TOP_K = 10;

  /** Invocations executed before the timed window to reach JIT steady state. */
  private static final int WARMUP_INVOCATIONS = 2_000;

  /** Odd so the median is a single sample, not the average of two. */
  private static final int MEASUREMENT_SAMPLES = 31;

  private RerankApp() {}

  /**
   * Pure-function variant of the demo's golden path: runs the rescore over a synthetic candidate
   * set and returns the top-K id list plus the count of positions that differ from the noisy
   * pre-rescore ranking. Extracted so a unit test can gate the "rescore actually changes the
   * ranking" contract without parsing stdout (audit T3.10).
   */
  public static RescoreResult runRescore(long seed, int candidates, int topK) {
    Random rnd = new Random(seed);
    float[] query = randomUnit(DIMENSION, rnd);
    float[][] candidateVectors = new float[candidates][];
    String[] candidateIds = new String[candidates];
    float[] noisyRemoteScores = new float[candidates];
    for (int i = 0; i < candidates; i++) {
      candidateIds[i] = "doc-" + i;
      candidateVectors[i] = randomUnit(DIMENSION, rnd);
      noisyRemoteScores[i] =
          VectorUtil.dotProduct(query, candidateVectors[i]) + (rnd.nextFloat() - 0.5f) * 0.1f;
    }
    float[] exactScores = new float[candidates];
    VectorUtil.batchDotProduct(query, candidateVectors, exactScores);
    int[] topOrder = TopK.select(exactScores, topK);
    String[] topIds = new String[topK];
    float[] topScores = new float[topK];
    for (int r = 0; r < topK; r++) {
      topIds[r] = candidateIds[topOrder[r]];
      topScores[r] = exactScores[topOrder[r]];
    }
    long swaps = countRankingSwaps(noisyRemoteScores, exactScores, topK);
    return new RescoreResult(topIds, topScores, swaps);
  }

  /** Result of the demo's golden-path rescore. */
  public record RescoreResult(String[] topIds, float[] topScores, long swapsVsNoisy) {}

  public static void main(String[] args) {
    Random rnd = new Random(42L);

    float[] query = randomUnit(DIMENSION, rnd);
    float[][] candidateVectors = new float[CANDIDATES][];
    String[] candidateIds = new String[CANDIDATES];
    float[] noisyRemoteScores = new float[CANDIDATES];
    for (int i = 0; i < CANDIDATES; i++) {
      candidateIds[i] = "doc-" + i;
      candidateVectors[i] = randomUnit(DIMENSION, rnd);
      // The "remote" score is dot product plus noise — i.e. an approximate ranking we want to
      // sharpen with a precise in-VM rescore.
      noisyRemoteScores[i] =
          VectorUtil.dotProduct(query, candidateVectors[i]) + (rnd.nextFloat() - 0.5f) * 0.1f;
    }

    // Cold-path sample: very first call, before any warmup. Runs in the interpreter or at C1 and
    // captures the "first-invocation" latency users pay once per JVM.
    float[] coldScores = new float[CANDIDATES];
    long coldStart = System.nanoTime();
    VectorUtil.batchDotProduct(query, candidateVectors, coldScores);
    long coldNanos = System.nanoTime() - coldStart;

    // Warm up until the fused GEMV kernel reaches C2 steady state. Three calls are not enough — the
    // Panama Vector API inner loop contains a species-sized main body, a 4-row unrolled kernel, and
    // a scalar tail; each needs profiling data before it is compiled and inlined.
    float[] warm = new float[CANDIDATES];
    for (int i = 0; i < WARMUP_INVOCATIONS; i++) {
      VectorUtil.batchDotProduct(query, candidateVectors, warm);
    }

    // Steady-state sample: report the median of N runs. A single sample is sensitive to a GC
    // young-pause, a safepoint, or OS scheduling; the median is resistant to all three.
    float[] exactScores = new float[CANDIDATES];
    long[] runs = new long[MEASUREMENT_SAMPLES];
    for (int i = 0; i < MEASUREMENT_SAMPLES; i++) {
      long t0 = System.nanoTime();
      VectorUtil.batchDotProduct(query, candidateVectors, exactScores);
      runs[i] = System.nanoTime() - t0;
    }
    Arrays.sort(runs);
    long medianNanos = runs[MEASUREMENT_SAMPLES / 2];

    int[] topOrder = TopK.select(exactScores, TOP_K);

    System.out.printf(
        "rescored %d x %d-dim candidates:%n"
            + "  cold  (1st call)       = %8.3f ms (%7.1f ns/candidate)%n"
            + "  warm  (median of %2d)   = %8.3f ms (%7.1f ns/candidate)%n",
        CANDIDATES,
        DIMENSION,
        coldNanos / 1_000_000.0,
        (double) coldNanos / CANDIDATES,
        MEASUREMENT_SAMPLES,
        medianNanos / 1_000_000.0,
        (double) medianNanos / CANDIDATES);

    System.out.println("top-" + TOP_K + " by precise in-VM rescore:");
    for (int r = 0; r < TOP_K; r++) {
      int i = topOrder[r];
      System.out.printf(
          "  rank=%2d  id=%-10s  exact=%.4f  noisy=%.4f%n",
          r + 1, candidateIds[i], exactScores[i], noisyRemoteScores[i]);
    }

    long swaps = countRankingSwaps(noisyRemoteScores, exactScores, TOP_K);
    System.out.printf(
        "%d of top-%d positions differ from the noisy remote ranking — that is the value the"
            + " in-VM rescore adds.%n",
        swaps, TOP_K);
  }

  private static float[] randomUnit(int dim, Random rnd) {
    float[] v = new float[dim];
    float norm = 0f;
    for (int i = 0; i < dim; i++) {
      v[i] = (float) rnd.nextGaussian();
      norm += v[i] * v[i];
    }
    norm = (float) Math.sqrt(norm);
    if (norm > 0f) {
      for (int i = 0; i < dim; i++) {
        v[i] /= norm;
      }
    }
    return v;
  }

  private static long countRankingSwaps(float[] noisyScores, float[] exactScores, int k) {
    int[] noisyTop = TopK.select(noisyScores, k);
    int[] exactTop = TopK.select(exactScores, k);
    long differ = 0;
    for (int r = 0; r < k; r++) {
      if (noisyTop[r] != exactTop[r]) {
        differ++;
      }
    }
    return differ;
  }
}
