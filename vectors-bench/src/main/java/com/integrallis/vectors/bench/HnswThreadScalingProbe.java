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
import com.integrallis.vectors.hnsw.ConcurrentHnswGraphBuilder;
import com.integrallis.vectors.hnsw.HnswGraph;
import com.integrallis.vectors.hnsw.HnswGraphBuilder;
import com.integrallis.vectors.hnsw.InMemoryVectors;
import java.util.Arrays;
import java.util.Random;

/**
 * Probe: how well does the current ReentrantLock-based ConcurrentHnswGraphBuilder scale with thread
 * count? If speedup is near-linear to the core count, lock contention is not a real bottleneck and
 * a CAS-based rewrite would not pay off. If speedup flattens early, there is headroom.
 *
 * <p>Runs a synthetic 10k-vector / 128-dim build with M=16 / efC=100 at {1, 2, 4, 6, 8} threads
 * (and single-threaded for reference), and prints wall-time + speedup.
 */
public final class HnswThreadScalingProbe {

  private HnswThreadScalingProbe() {}

  public static void main(String[] args) {
    int n = Integer.getInteger("bench.n", 10_000);
    int dim = Integer.getInteger("bench.dim", 128);
    int m = Integer.getInteger("bench.hnsw.m", 16);
    int efC = Integer.getInteger("bench.hnsw.ef", 100);
    int[] threads = parseInts(System.getProperty("bench.threads"), new int[] {1, 2, 4, 6, 8});
    int warmup = Integer.getInteger("bench.warmup", 10);
    int iters = Integer.getInteger("bench.iters", 5);

    float[][] corpus = synthetic(n, dim, 42L);
    InMemoryVectors vec = new InMemoryVectors(corpus);

    System.out.printf("HNSW thread-scaling probe: n=%,d dim=%d M=%d efC=%d%n", n, dim, m, efC);

    // Reference: the single-threaded HnswGraphBuilder. This is the real baseline that any parallel
    // implementation must beat; ConcurrentHnswGraphBuilder.build(1) has extra per-node lock /
    // executor overhead even at parallelism=1.
    long seqNs =
        time(
            warmup,
            iters,
            () -> HnswGraphBuilder.create(m, efC, vec, SimilarityFunction.EUCLIDEAN, 42L).build());
    System.out.printf("  seq HnswGraphBuilder:           %,10.1f ms%n", seqNs / 1e6);

    long base = -1;
    for (int t : threads) {
      long ns =
          time(
              warmup,
              iters,
              () ->
                  ConcurrentHnswGraphBuilder.create(m, efC, vec, SimilarityFunction.EUCLIDEAN, 42L)
                      .build(t));
      if (base < 0) base = ns;
      double speedupVsSeq = seqNs / (double) ns;
      double speedupVsT1 = base / (double) ns;
      System.out.printf(
          "  concurrent(threads=%d):          %,10.1f ms  speedup vs seq=%.2fx  vs T1=%.2fx%n",
          t, ns / 1e6, speedupVsSeq, speedupVsT1);
    }
  }

  private static long time(int warmup, int iters, BuildOp op) {
    for (int i = 0; i < warmup; i++) {
      HnswGraph g = op.run();
      if (g == null) throw new IllegalStateException("null graph");
    }
    // Use the median across iterations. min() would mask lock-contention overhead by rewarding
    // the best-case run (no lock waits, warm caches) — not representative of production.
    long[] samples = new long[iters];
    for (int i = 0; i < iters; i++) {
      long t0 = System.nanoTime();
      HnswGraph g = op.run();
      long dt = System.nanoTime() - t0;
      if (g == null) throw new IllegalStateException("null graph");
      samples[i] = dt;
    }
    Arrays.sort(samples);
    return samples[samples.length / 2];
  }

  private static float[][] synthetic(int n, int dim, long seed) {
    Random r = new Random(seed);
    float[][] out = new float[n][dim];
    for (int i = 0; i < n; i++) {
      for (int d = 0; d < dim; d++) out[i][d] = r.nextFloat() * 2f - 1f;
    }
    return out;
  }

  private static int[] parseInts(String csv, int[] fallback) {
    if (csv == null || csv.isBlank()) return fallback;
    String[] parts = csv.split(",");
    int[] out = new int[parts.length];
    for (int i = 0; i < parts.length; i++) out[i] = Integer.parseInt(parts[i].trim());
    return out;
  }

  @FunctionalInterface
  private interface BuildOp {
    HnswGraph run();
  }
}
