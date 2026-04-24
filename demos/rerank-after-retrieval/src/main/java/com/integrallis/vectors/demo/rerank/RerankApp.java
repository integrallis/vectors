package com.integrallis.vectors.demo.rerank;

import com.integrallis.vectors.core.VectorUtil;
import java.util.Arrays;
import java.util.Comparator;
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
 *   <li>Times the operation and prints the top-10 rerank winners
 * </ol>
 *
 * <p>On a laptop class CPU, rescoring 1000 × 384-dim candidates typically completes in well under a
 * millisecond through the Panama Vector API.
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

  private RerankApp() {}

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

    // Warm up the SIMD kernel.
    float[] warm = new float[CANDIDATES];
    for (int i = 0; i < 3; i++) {
      VectorUtil.batchDotProduct(query, candidateVectors, warm);
    }

    float[] exactScores = new float[CANDIDATES];
    long start = System.nanoTime();
    VectorUtil.batchDotProduct(query, candidateVectors, exactScores);
    long elapsedNanos = System.nanoTime() - start;

    Integer[] order = new Integer[CANDIDATES];
    for (int i = 0; i < CANDIDATES; i++) {
      order[i] = i;
    }
    Arrays.sort(order, Comparator.comparingDouble((Integer i) -> exactScores[i]).reversed());

    System.out.printf(
        "rescored %d x %d-dim candidates in %.3f ms (%.1f ns/candidate)%n",
        CANDIDATES, DIMENSION, elapsedNanos / 1_000_000.0, (double) elapsedNanos / CANDIDATES);

    System.out.println("top-" + TOP_K + " by precise in-VM rescore:");
    for (int r = 0; r < TOP_K; r++) {
      int i = order[r];
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
    int n = noisyScores.length;
    Integer[] noisyOrder = new Integer[n];
    Integer[] exactOrder = new Integer[n];
    for (int i = 0; i < n; i++) {
      noisyOrder[i] = i;
      exactOrder[i] = i;
    }
    Arrays.sort(noisyOrder, Comparator.comparingDouble((Integer i) -> noisyScores[i]).reversed());
    Arrays.sort(exactOrder, Comparator.comparingDouble((Integer i) -> exactScores[i]).reversed());
    long differ = 0;
    for (int r = 0; r < k; r++) {
      if (!noisyOrder[r].equals(exactOrder[r])) {
        differ++;
      }
    }
    return differ;
  }
}
