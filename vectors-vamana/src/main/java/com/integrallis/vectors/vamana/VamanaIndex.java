package com.integrallis.vectors.vamana;

import com.integrallis.vectors.core.SimilarityFunction;
import java.util.Objects;

/**
 * Thread-safe Vamana index with builder pattern API.
 *
 * <p>Uses {@link ThreadLocal} searchers for lock-free concurrent search. Construction is
 * single-threaded.
 *
 * <p>Usage:
 *
 * <pre>{@code
 * VamanaIndex index = VamanaIndex.builder(vectors, SimilarityFunction.EUCLIDEAN)
 *     .maxDegree(64)
 *     .searchListSize(128)
 *     .alpha(1.2f)
 *     .seed(42L)
 *     .build();
 *
 * SearchResult result = index.search(query, 10);
 * }</pre>
 */
public final class VamanaIndex {

  private final VamanaGraph graph;
  private final RandomAccessVectors vectors;
  private final SimilarityFunction sim;
  private final ThreadLocal<VamanaSearcher> threadLocalSearcher;

  private VamanaIndex(VamanaGraph graph, RandomAccessVectors vectors, SimilarityFunction sim) {
    this.graph = Objects.requireNonNull(graph, "graph must not be null");
    this.vectors = Objects.requireNonNull(vectors, "vectors must not be null");
    this.sim = Objects.requireNonNull(sim, "sim must not be null");
    this.threadLocalSearcher =
        ThreadLocal.withInitial(() -> new VamanaSearcher(graph, vectors, sim));
  }

  /**
   * Searches for the k nearest neighbors with default search list size.
   *
   * @param query the query vector
   * @param k number of results to return
   * @return search result with ranked node IDs and scores
   */
  public SearchResult search(float[] query, int k) {
    return threadLocalSearcher.get().search(query, k);
  }

  /**
   * Searches for the k nearest neighbors.
   *
   * @param query the query vector
   * @param k number of results to return
   * @param searchListSize beam width L
   * @return search result with ranked node IDs and scores
   */
  public SearchResult search(float[] query, int k, int searchListSize) {
    return threadLocalSearcher.get().search(query, k, searchListSize);
  }

  /**
   * Returns the underlying graph for testing and inspection.
   *
   * <p><b>Warning:</b> the returned {@link VamanaGraph} is the live graph backing this index.
   * Mutating it (e.g., calling {@link VamanaGraph#getNeighbors(int)} and clearing the returned
   * array) will permanently corrupt the index. Use only for read-only inspection.
   */
  public VamanaGraph graph() {
    return graph;
  }

  /** Returns the number of vectors in the index. */
  public int size() {
    return graph.size();
  }

  /**
   * Creates a builder for a float[][] dataset.
   *
   * @param vectors the dataset as a 2D array
   * @param sim similarity function
   * @return a new builder
   */
  public static Builder builder(float[][] vectors, SimilarityFunction sim) {
    return new Builder(new InMemoryVectors(vectors), sim);
  }

  /**
   * Creates a builder for a RandomAccessVectors dataset.
   *
   * @param vectors the dataset
   * @param sim similarity function
   * @return a new builder
   */
  public static Builder builder(RandomAccessVectors vectors, SimilarityFunction sim) {
    return new Builder(vectors, sim);
  }

  /** Builder for {@link VamanaIndex}. */
  public static final class Builder {

    private final RandomAccessVectors vectors;
    private final SimilarityFunction sim;
    private int maxDegree = 64;
    private int searchListSize = 128;
    private float alpha = 1.2f;
    private long seed = System.nanoTime();

    private Builder(RandomAccessVectors vectors, SimilarityFunction sim) {
      this.vectors = Objects.requireNonNull(vectors, "vectors must not be null");
      this.sim = Objects.requireNonNull(sim, "sim must not be null");
    }

    /** Sets the maximum degree R (default: 64). */
    public Builder maxDegree(int r) {
      if (r <= 0) {
        throw new IllegalArgumentException("maxDegree must be positive: " + r);
      }
      this.maxDegree = r;
      return this;
    }

    /** Sets the search list size L for construction (default: 128). */
    public Builder searchListSize(int l) {
      if (l <= 0) {
        throw new IllegalArgumentException("searchListSize must be positive: " + l);
      }
      this.searchListSize = l;
      return this;
    }

    /** Sets the diversity parameter alpha (default: 1.2). Must be >= 1.0. */
    public Builder alpha(float a) {
      if (a < 1.0f) {
        throw new IllegalArgumentException("alpha must be >= 1.0: " + a);
      }
      this.alpha = a;
      return this;
    }

    /** Sets the random seed for deterministic construction (default: System.nanoTime()). */
    public Builder seed(long seed) {
      this.seed = seed;
      return this;
    }

    /** Builds the index. */
    public VamanaIndex build() {
      VamanaGraphBuilder graphBuilder =
          VamanaGraphBuilder.create(maxDegree, searchListSize, alpha, vectors, sim, seed);
      VamanaGraph graph = graphBuilder.build();
      return new VamanaIndex(graph, vectors, sim);
    }
  }
}
