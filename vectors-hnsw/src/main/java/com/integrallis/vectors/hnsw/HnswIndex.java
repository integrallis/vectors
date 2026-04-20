package com.integrallis.vectors.hnsw;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.quantization.CompressedVectors;
import com.integrallis.vectors.quantization.ScoreFunction;
import java.util.function.IntConsumer;
import java.util.function.IntPredicate;

/**
 * Top-level HNSW index providing build and search functionality via a builder pattern.
 *
 * <p>Concurrent searches from multiple threads are safe: each thread uses its own {@link
 * HnswSearcher} instance (with its own scratch buffers) retrieved from a {@link ThreadLocal}.
 * Building the index is single-threaded.
 *
 * <p>Supports optional quantization via {@link #enableQuantization(CompressedVectors)} for two-pass
 * search: a fast coarse pass using quantized scoring followed by full-precision rescoring.
 *
 * <p>Usage:
 *
 * <pre>{@code
 * HnswIndex index = HnswIndex.builder(vectors, SimilarityFunction.EUCLIDEAN)
 *     .maxConnections(16)
 *     .efConstruction(200)
 *     .build();
 *
 * // Full-precision search (thread-safe)
 * SearchResult result = index.search(queryVector, 10);
 *
 * // Two-pass quantized search (thread-safe)
 * index.enableQuantization(compressedVectors);
 * SearchResult twoPass = index.searchTwoPass(queryVector, 10);
 * }</pre>
 */
public final class HnswIndex {

  private final HnswGraph graph;
  private final RandomAccessVectors vectors;
  private final SimilarityFunction similarityFunction;
  // One HnswSearcher per thread: each owns its own scratch BitSet and NodeQueues.
  private final ThreadLocal<HnswSearcher> threadLocalSearcher;

  /**
   * Holds both fields of quantization state in a single record so they can be published atomically
   * via one {@code volatile} write to {@link #quantizationState}. Two separate {@code volatile}
   * fields would allow a concurrent reader to observe a torn state (e.g., {@code compressedVectors}
   * from one call and the old {@code threadLocalSearcher} from before).
   */
  private record QuantizationState(
      CompressedVectors compressed, ThreadLocal<HnswSearcher> searchers) {}

  // Single volatile reference — readers always see a consistent (compressed, searchers) pair.
  private volatile QuantizationState quantizationState;

  private HnswIndex(
      HnswGraph graph, RandomAccessVectors vectors, SimilarityFunction similarityFunction) {
    this.graph = graph;
    this.vectors = vectors;
    this.similarityFunction = similarityFunction;
    this.threadLocalSearcher =
        ThreadLocal.withInitial(() -> new HnswSearcher(graph, vectors, similarityFunction));
  }

  /**
   * Returns the raw vector source backing this index.
   *
   * <p>Useful for constructing an {@link AsyncVectorPrefetcher} against the same source without
   * copying vectors, or for direct vector access in custom scoring pipelines.
   */
  public RandomAccessVectors vectorSource() {
    return vectors;
  }

  /**
   * Attaches compressed vectors for two-pass search. Creates a {@link ThreadLocal} pool of
   * quantized searchers and publishes them atomically with the compressed-vector reference.
   *
   * @param compressed the compressed vectors (must match this index's size and dimension)
   * @throws IllegalArgumentException if size or dimension does not match
   */
  public void enableQuantization(CompressedVectors compressed) {
    if (compressed.size() != vectors.size()) {
      throw new IllegalArgumentException(
          "CompressedVectors size ("
              + compressed.size()
              + ") must match index size ("
              + vectors.size()
              + ")");
    }
    if (compressed.dimension() != vectors.dimension()) {
      throw new IllegalArgumentException(
          "CompressedVectors dimension ("
              + compressed.dimension()
              + ") must match index dimension ("
              + vectors.dimension()
              + ")");
    }
    SimilarityFunction sim = this.similarityFunction;
    ThreadLocal<HnswSearcher> tl =
        ThreadLocal.withInitial(
            () -> {
              NodeScorerFactory quantizedFactory =
                  query -> {
                    ScoreFunction sf = compressed.scoreFunctionFor(query, sim);
                    return sf::score;
                  };
              return new HnswSearcher(graph, vectors, sim, quantizedFactory);
            });
    // Single volatile write — atomically publishes both the compressed vectors and the
    // thread-local searcher pool.
    this.quantizationState = new QuantizationState(compressed, tl);
  }

  /** Removes quantization attachment. Subsequent {@link #searchTwoPass} falls back to exact. */
  public void disableQuantization() {
    this.quantizationState = null;
  }

  /** Returns true if quantized vectors are attached. */
  public boolean isQuantizationEnabled() {
    return quantizationState != null;
  }

  /**
   * Searches for the k nearest neighbors to the query vector using full-precision scoring.
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

  /**
   * Two-pass search: fast quantized coarse pass followed by full-precision rescore.
   *
   * <p>If no quantization is enabled, falls back to the full-precision searcher running the same
   * two-pass pipeline (over-fetch {@code overQueryFactor × k} candidates with exact scoring, then
   * rescore to the final top-k). {@code overQueryFactor} is always honoured regardless of whether
   * quantization is enabled — the no-quantization path is not a short-circuit that discards it.
   *
   * @param query the query vector
   * @param k number of final results
   * @param efSearch beam width for the coarse pass
   * @param overQueryFactor multiplier for coarse-pass k (e.g., 2.0 retrieves 2*k candidates)
   * @return the top-k results after rescoring, sorted by score descending
   */
  public SearchResult searchTwoPass(float[] query, int k, int efSearch, float overQueryFactor) {
    // Single volatile read — guaranteed to see a consistent (compressed, searchers) pair.
    QuantizationState qs = this.quantizationState;
    HnswSearcher searcher = qs != null ? qs.searchers().get() : threadLocalSearcher.get();
    return searcher.searchTwoPass(query, k, efSearch, overQueryFactor);
  }

  /** Two-pass search with default efSearch=max(k,100) and overQueryFactor=2.0. Thread-safe. */
  public SearchResult searchTwoPass(float[] query, int k) {
    return searchTwoPass(query, k, Math.max(k, 100), 2.0f);
  }

  /**
   * ACORN-style pre-filtered search: traverses the full graph for navigation but accumulates only
   * nodes that pass {@code predicate} into the result set. Thread-safe: each calling thread uses
   * its own scratch buffers.
   *
   * <p>Non-matching nodes are still explored as routing hops — they can bridge graph regions that
   * would otherwise be unreachable from matching nodes alone. This avoids the recall collapse that
   * naive post-filtering produces on highly selective filters.
   *
   * @param query the query vector
   * @param k number of matching results to return
   * @param efSearch beam width for matching results (must be ≥ k)
   * @param predicate ordinal-level filter — {@code true} means the node is eligible for results
   * @return the top-k matching results sorted by score descending
   */
  public SearchResult searchFiltered(float[] query, int k, int efSearch, IntPredicate predicate) {
    return threadLocalSearcher.get().searchFiltered(query, k, efSearch, predicate);
  }

  /** Pre-filtered search with default efSearch = max(k, 100). Thread-safe. */
  public SearchResult searchFiltered(float[] query, int k, IntPredicate predicate) {
    return threadLocalSearcher.get().searchFiltered(query, k, predicate);
  }

  /**
   * SSD-aware search with asynchronous prefetching.
   *
   * <p>Before scoring each candidate's neighbor array, the {@code prefetcher} is asked to issue
   * async touch-reads for all neighbor ordinals. For mmap-backed vector stores this causes the OS
   * to load the relevant pages into the page cache concurrently with scoring the <em>current</em>
   * candidate, hiding I/O latency.
   *
   * <p>Thread safety: The prefetch hook is set and cleared within this call on the calling thread's
   * own {@link HnswSearcher} instance (retrieved from the {@link ThreadLocal}), so no cross-thread
   * sharing occurs.
   *
   * @param query the query vector
   * @param k number of results to return
   * @param efSearch beam width (must be >= k)
   * @param prefetcher the async prefetcher to use; must be open and not {@code null}
   * @return search results sorted by score descending
   */
  public SearchResult searchWithPrefetch(
      float[] query, int k, int efSearch, AsyncVectorPrefetcher prefetcher) {
    HnswSearcher searcher = threadLocalSearcher.get();
    IntConsumer hook = prefetcher::prefetch;
    searcher.setPrefetchHook(hook);
    try {
      return searcher.search(query, k, efSearch);
    } finally {
      searcher.setPrefetchHook(null); // always reset — even on exception
    }
  }

  /**
   * Creates a new full-precision searcher instance with its own scratch buffers.
   *
   * <p><b>Always returns a full-precision searcher</b>, regardless of whether quantization is
   * enabled. For quantized two-pass search, call {@link #searchTwoPass} directly instead.
   */
  public HnswSearcher searcher() {
    return new HnswSearcher(graph, vectors, similarityFunction);
  }

  /**
   * Returns the underlying {@link HnswGraph}. Exposed for persistence paths that need to serialize
   * the graph topology to disk (e.g. {@code com.integrallis.vectors.db.storage.HnswGraphCodec} in
   * Step 4b). The returned graph aliases the index's internal reference; callers must not mutate it
   * through {@link HnswGraph#initNode} or {@link HnswGraph#setEntryNode}, which would corrupt
   * in-flight searches.
   */
  public HnswGraph graph() {
    return graph;
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

  /**
   * Wraps a pre-built {@link HnswGraph} together with its backing {@link RandomAccessVectors} and
   * similarity function into a searchable {@link HnswIndex} WITHOUT running any graph construction.
   * The caller is responsible for ensuring that {@code graph}, {@code vectors}, and {@code
   * similarityFunction} are mutually consistent (same vector count, same metric used when the graph
   * was originally built, etc.).
   *
   * <p>This factory exists exclusively for the persistence path in {@code vectors-db} Step 4b: a
   * decoded graph from {@code HnswGraphCodec.decode(byte[])} is paired with a {@code
   * MemorySegmentRandomAccessVectors} view of the mmap'd {@code vectors.bin} file, and the
   * resulting index is wrapped in {@code MappedHnswIndexAdapter} for read-only search. There is no
   * "rebuild-on-commit" concern here — the graph is loaded from disk and the index is never mutated
   * after construction.
   *
   * <p>The returned index shares its internal state with the arguments; in particular it does not
   * copy {@code graph} or {@code vectors}. Do NOT mutate either through any other reference once
   * this factory returns.
   *
   * @throws NullPointerException if any argument is null
   */
  public static HnswIndex ofPrebuilt(
      HnswGraph graph, RandomAccessVectors vectors, SimilarityFunction similarityFunction) {
    java.util.Objects.requireNonNull(graph, "graph must not be null");
    java.util.Objects.requireNonNull(vectors, "vectors must not be null");
    java.util.Objects.requireNonNull(similarityFunction, "similarityFunction must not be null");
    return new HnswIndex(graph, vectors, similarityFunction);
  }

  /** Builder for configuring and constructing an HNSW index. */
  public static final class Builder {

    private final RandomAccessVectors vectors;
    private final SimilarityFunction similarityFunction;
    private int maxConnections = 16;
    private int efConstruction = 200;
    private long seed = System.nanoTime();
    private int parallelism = 1;

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

    /**
     * Sets the number of worker threads used during graph construction. Default 1 routes to the
     * deterministic {@link HnswGraphBuilder}; values {@code > 1} route to {@link
     * ConcurrentHnswGraphBuilder}, which produces valid (but non-deterministic) graphs with
     * equivalent recall. Must be {@code >= 1}.
     */
    public Builder parallelism(int threads) {
      if (threads < 1) {
        throw new IllegalArgumentException("parallelism must be >= 1: " + threads);
      }
      this.parallelism = threads;
      return this;
    }

    /** Builds the HNSW index by inserting all vectors. */
    public HnswIndex build() {
      HnswGraph graph;
      if (parallelism <= 1) {
        graph =
            HnswGraphBuilder.create(
                    maxConnections, efConstruction, vectors, similarityFunction, seed)
                .build();
      } else {
        graph =
            ConcurrentHnswGraphBuilder.create(
                    maxConnections, efConstruction, vectors, similarityFunction, seed)
                .build(parallelism);
      }
      return new HnswIndex(graph, vectors, similarityFunction);
    }
  }
}
