package com.integrallis.vectors.demo.cache;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.Document;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.VectorCollection;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;

/**
 * Persistent embedding-cache demo.
 *
 * <p>Every production LLM pipeline recomputes embeddings. A mmap-persistent {@link
 * VectorCollection} beside the app is a zero-infra embedding cache: {@code get-or-embed} on miss,
 * sub-millisecond recall on hit, survives restarts, and scales to tens of millions of cached
 * embeddings on a commodity box.
 *
 * <p>Each text is keyed by a SHA-256 hash of its content. The cache is a {@code FLAT}-indexed
 * collection persisted at {@code ~/.java-vectors-demo/embedding-cache}. The first run populates the
 * cache (all misses). Subsequent runs hit the cache (zero calls into the real embedder).
 *
 * <p>Run twice to see the effect:
 *
 * <pre>
 *   ./gradlew :demos:embedding-cache:run     # first run: all misses
 *   ./gradlew :demos:embedding-cache:run     # second run: all hits
 * </pre>
 */
public final class EmbeddingCacheApp {

  private static final int DIMENSION = 128;

  private EmbeddingCacheApp() {}

  public static void main(String[] args) {
    Path storage =
        Path.of(System.getProperty("user.home"), ".java-vectors-demo", "embedding-cache");
    AtomicInteger realEmbedderCalls = new AtomicInteger();
    Function<String, float[]> realEmbedder =
        text -> {
          realEmbedderCalls.incrementAndGet();
          return trigramVector(text, DIMENSION);
        };

    List<String> texts =
        List.of(
            "What is HNSW?",
            "Explain product quantization.",
            "How does mmap persistence work?",
            "What is HNSW?", // duplicate — should hit the cache even on first run
            "Describe scalar quantization.",
            "Explain product quantization."); // duplicate

    try (VectorCollection cache =
        VectorCollection.builder()
            .dimension(DIMENSION)
            .metric(SimilarityFunction.COSINE)
            .indexType(IndexType.FLAT)
            .storagePath(storage)
            .autoCommitThreshold(1)
            .build()) {

      System.out.println("cache dir: " + storage);
      System.out.println("entries on open: " + cache.size());

      // Hydrate an in-memory id->vector lookup from the persisted collection. On a fresh run this
      // is empty; after the first run the collection restores its content from mmap storage.
      Map<String, float[]> inMemory = new HashMap<>(cache.size());
      for (Document d : cache.documents()) {
        inMemory.put(d.id(), d.vector());
      }

      int hits = 0;
      int misses = 0;
      for (String text : texts) {
        String key = cacheKey(text);
        float[] cached = inMemory.get(key);
        if (cached != null) {
          hits++;
          System.out.printf("  HIT   %s  (vec[0]=%.4f)%n", text, cached[0]);
        } else {
          misses++;
          float[] vec = realEmbedder.apply(text);
          cache.add(Document.of(key, vec, text));
          inMemory.put(key, vec);
          System.out.printf("  MISS  %s%n", text);
        }
      }
      cache.commit();

      System.out.println();
      System.out.printf(
          "hits: %d, misses: %d, real embedder calls: %d, cache entries after run: %d%n",
          hits, misses, realEmbedderCalls.get(), cache.size());
      System.out.println(
          "Re-run ./gradlew :demos:embedding-cache:run to see the cache replay on the next"
              + " process (all MISS become HIT, zero real embedder calls).");
    }
  }

  private static String cacheKey(String text) {
    try {
      MessageDigest sha = MessageDigest.getInstance("SHA-256");
      return HexFormat.of().formatHex(sha.digest(text.getBytes(StandardCharsets.UTF_8)));
    } catch (NoSuchAlgorithmException e) {
      throw new IllegalStateException("SHA-256 unavailable", e);
    }
  }

  private static float[] trigramVector(String text, int dimension) {
    float[] out = new float[dimension];
    String normalized = text.toLowerCase();
    if (normalized.length() < 3) {
      normalized = "  " + normalized + "  ";
    }
    for (int i = 0; i <= normalized.length() - 3; i++) {
      int h =
          31 * (31 * normalized.charAt(i) + normalized.charAt(i + 1)) + normalized.charAt(i + 2);
      out[Math.floorMod(h, dimension)] += 1.0f;
    }
    float norm = 0f;
    for (float v : out) {
      norm += v * v;
    }
    norm = (float) Math.sqrt(norm);
    if (norm > 0f) {
      for (int i = 0; i < dimension; i++) {
        out[i] /= norm;
      }
    }
    return out;
  }
}
