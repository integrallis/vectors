package com.integrallis.vectors.hnsw;

import com.integrallis.vectors.core.SimilarityFunction;

/**
 * Top-level HNSW index providing build and search functionality via a builder pattern.
 *
 * <p>Concurrent searches from multiple threads are safe: each thread uses its own {@link
 * HnswSearcher} instance (with its own scratch buffers) retrieved from a {@link ThreadLocal}.
 * Building the index is single-threaded.
 *
 * <p>Usage:
 *
 * <pre>{@code
 * HnswIndex index = HnswIndex.builder(vectors, SimilarityFunction.EUCLIDEAN)
 *     .maxConnections(16)
 *     .efConstruction(200)
 *     .build();
 *
 * // Thread-safe — each calling thread gets its own searcher automatically.
 * SearchResult result = index.search(queryVector, 10);
 * }</pre>
 */
public final class HnswIndex {

  private final HnswGraph graph;
  private final RandomAccessVectors vectors;
  private final SimilarityFunction similarityFunction;
  // One HnswSearcher per thread: each owns its own scratch BitSet and NodeQueues.
  private final ThreadLocal<HnswSearcher> threadLocalSearcher;

  private HnswIndex(
      HnswGraph graph, RandomAccessVectors vectors, SimilarityFunction similarityFunction) {
    this.graph = graph;
    this.vectors = vectors;
    this.similarityFunction = similarityFunction;
    this.threadLocalSearcher =
        ThreadLocal.withInitial(() -> new HnswSearcher(graph, vectors, similarityFunction));
  }

  /**
   * Searches for the k nearest neighbors to the query vector.
   *
   * <p>Thread-safe: each calling thread uses its own scratch buffers.
   *
   * @param query the query vector
   * @param k number of results to return
   * @param efSearch beam width (must be >= k)
   * @return search results sorted by score descending
   */
  public SearchResult search(float[] query, int k, int efSearch) {
    return threadLocalSearcher.get().search(query, k, efSearch);
  }

  /** Searches with default efSearch = max(k, 100). Thread-safe. */
  public SearchResult search(float[] query, int k) {
    return threadLocalSearcher.get().search(query, k);
  }

  /** Creates a new per-thread searcher instance with its own scratch buffers. */
  public HnswSearcher searcher() {
    return new HnswSearcher(graph, vectors, similarityFunction);
  }

  /** Returns the number of vectors in the index. */
  public int size() {
    return graph.size();
  }

  /** Returns the vector dimension. */
  public int dimension() {
    return vectors.dimension();
  }

  /** Returns the similarity function used by this index. */
  public SimilarityFunction similarityFunction() {
    return similarityFunction;
  }

  /** Creates a builder for an HNSW index from raw float vectors. */
  public static Builder builder(float[][] vectors, SimilarityFunction similarityFunction) {
    return new Builder(new InMemoryVectors(vectors), similarityFunction);
  }

  /** Creates a builder for an HNSW index from a RandomAccessVectors instance. */
  public static Builder builder(
      RandomAccessVectors vectors, SimilarityFunction similarityFunction) {
    return new Builder(vectors, similarityFunction);
  }

  /** Builder for configuring and constructing an HNSW index. */
  public static final class Builder {

    private final RandomAccessVectors vectors;
    private final SimilarityFunction similarityFunction;
    private int maxConnections = 16;
    private int efConstruction = 200;
    private long seed = System.nanoTime();

    private Builder(RandomAccessVectors vectors, SimilarityFunction similarityFunction) {
      this.vectors = vectors;
      this.similarityFunction = similarityFunction;
    }

    /** Sets M (max connections per upper layer). Default: 16. */
    public Builder maxConnections(int m) {
      this.maxConnections = m;
      return this;
    }

    /** Sets the ef parameter for construction-time search. Default: 200. */
    public Builder efConstruction(int ef) {
      this.efConstruction = ef;
      return this;
    }

    /** Sets the random seed for deterministic construction. */
    public Builder seed(long seed) {
      this.seed = seed;
      return this;
    }

    /** Builds the HNSW index by inserting all vectors. */
    public HnswIndex build() {
      var graphBuilder =
          HnswGraphBuilder.create(
              maxConnections, efConstruction, vectors, similarityFunction, seed);
      HnswGraph graph = graphBuilder.build();
      return new HnswIndex(graph, vectors, similarityFunction);
    }
  }
}
