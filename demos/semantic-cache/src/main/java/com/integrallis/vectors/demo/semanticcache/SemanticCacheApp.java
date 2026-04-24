package com.integrallis.vectors.demo.semanticcache;

import com.integrallis.vectors.cache.SemanticCache;
import com.integrallis.vectors.cache.semantic.PayloadCodec;
import com.integrallis.vectors.cache.semantic.VectorDbSemanticCache;
import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.VectorCollection;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Semantic-cache demo.
 *
 * <p>A semantic cache returns a cached answer when the <i>query</i> is sufficiently similar to a
 * previously-seen query — not just identical. This is the standard pattern for short-circuiting
 * expensive LLM calls on paraphrased questions ("what is HNSW?" vs "explain HNSW to me").
 *
 * <p>This demo:
 *
 * <ol>
 *   <li>Opens an in-memory FLAT/COSINE {@link VectorCollection}.
 *   <li>Wraps it in a {@link VectorDbSemanticCache} with a cosine threshold of 0.85.
 *   <li>Embeds a handful of canonical questions with a deterministic trigram hash and stores their
 *       canned answers.
 *   <li>Issues paraphrased questions, embeds them the same way, and reports hit/miss + score.
 * </ol>
 *
 * <p>Run:
 *
 * <pre>
 *   ./gradlew :demos:semantic-cache:run
 * </pre>
 */
public final class SemanticCacheApp {

  private static final int DIMENSION = 64;
  private static final double THRESHOLD = 0.85;

  private SemanticCacheApp() {}

  public static void main(String[] args) {
    AtomicInteger llmCalls = new AtomicInteger();

    try (VectorCollection backing =
            VectorCollection.builder()
                .dimension(DIMENSION)
                .metric(SimilarityFunction.COSINE)
                .indexType(IndexType.FLAT)
                .autoCommitThreshold(1)
                .build();
        SemanticCache<String> cache =
            VectorDbSemanticCache.builder(backing, PayloadCodec.identity())
                .threshold(THRESHOLD)
                .build()) {

      System.out.printf(
          "semantic cache: cosine threshold=%.2f, dimension=%d%n", THRESHOLD, DIMENSION);

      // ---- Seed canonical Q->A pairs.
      List<String[]> seed =
          List.of(
              new String[] {
                "What is HNSW?", "HNSW is a graph-based approximate nearest neighbor index."
              },
              new String[] {
                "Explain product quantization.",
                "PQ splits each vector into sub-vectors and quantizes each independently."
              },
              new String[] {
                "What is mmap persistence?",
                "mmap maps a file into the process's address space for zero-copy reads."
              });
      for (String[] qa : seed) {
        cache.put(qa[0], trigramVector(qa[0], DIMENSION), qa[1]);
      }
      System.out.printf("seeded %d canonical answers%n%n", seed.size());

      // ---- Issue paraphrases; hit iff top-1 score >= THRESHOLD.
      List<String> queries =
          List.of(
              "What is HNSW?", // exact match
              "Can you explain HNSW?", // paraphrase
              "Explain product quantization.", // exact match
              "Tell me about product quantization", // paraphrase
              "What is scalar quantization?", // novel — miss
              "How does mmap persistence work?"); // paraphrase

      int hits = 0;
      int misses = 0;
      for (String q : queries) {
        float[] qvec = trigramVector(q, DIMENSION);
        Optional<SemanticCache.Hit<String>> hit = cache.lookup(qvec);
        if (hit.isPresent()) {
          hits++;
          System.out.printf(
              "  HIT   (score=%.4f) %-40s -> %s%n", hit.get().score(), q, hit.get().value());
        } else {
          misses++;
          // Simulate the real LLM call on a miss, then backfill the cache.
          String answer = callRealLlm(q, llmCalls);
          cache.put(q, qvec, answer);
          System.out.printf("  MISS  %-50s -> %s%n", q, answer);
        }
      }

      System.out.println();
      System.out.printf(
          "hits: %d, misses: %d, real LLM calls saved: %d%n",
          hits, misses, hits); // every hit is one LLM call avoided
      System.out.printf("cache entries after run: %d%n", cache.stats().size());
    }
  }

  private static String callRealLlm(String question, AtomicInteger counter) {
    counter.incrementAndGet();
    return "[simulated LLM answer for: " + question + "]";
  }

  private static float[] trigramVector(String text, int dim) {
    float[] out = new float[dim];
    String normalized = text.toLowerCase();
    if (normalized.length() < 3) normalized = "  " + normalized + "  ";
    for (int i = 0; i <= normalized.length() - 3; i++) {
      int h =
          31 * (31 * normalized.charAt(i) + normalized.charAt(i + 1)) + normalized.charAt(i + 2);
      out[Math.floorMod(h, dim)] += 1.0f;
    }
    float norm = 0f;
    for (float v : out) norm += v * v;
    norm = (float) Math.sqrt(norm);
    if (norm > 0f) {
      for (int i = 0; i < dim; i++) out[i] /= norm;
    }
    return out;
  }
}
