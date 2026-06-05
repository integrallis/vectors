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
package com.integrallis.vectors.vamana;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.quantization.CompressedVectors;
import com.integrallis.vectors.quantization.ScoreFunction;
import java.util.Objects;

/**
 * Thread-safe Vamana index with builder pattern API.
 *
 * <p>Uses {@link ThreadLocal} searchers for lock-free concurrent search. The builder constructs the
 * graph before publishing the index.
 *
 * <p>Supports optional quantization via {@link #enableQuantization(CompressedVectors)} for two-pass
 * search: a fast coarse pass using quantized scoring followed by full-precision rescoring.
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
 * // Full-precision search (thread-safe)
 * SearchResult result = index.search(query, 10);
 *
 * // Two-pass quantized search (thread-safe)
 * index.enableQuantization(compressedVectors);
 * SearchResult twoPass = index.searchTwoPass(query, 10);
 * }</pre>
 */
public final class VamanaIndex {

  // Topology the searcher traverses — either the heap VamanaGraph or a paged disk-resident view.
  private final VamanaTopology topology;
  // The concrete heap graph, or null when this index wraps a paged topology. Exposed by graph()
  // for the commit/encode path (which only ever runs on a heap-built index, never a paged one).
  private final VamanaGraph graph;
  private final RandomAccessVectors vectors;
  private final SimilarityFunction sim;
  private final ThreadLocal<VamanaSearcher> threadLocalSearcher;

  /**
   * Holds both fields of quantization state in a single record so they can be published atomically
   * via one {@code volatile} write to {@link #quantizationState}. Two separate {@code volatile}
   * fields would allow a concurrent reader to observe a torn state (e.g., {@code compressed} from
   * one call and the old {@code searchers} from before).
   */
  private record QuantizationState(
      CompressedVectors compressed, ThreadLocal<VamanaSearcher> searchers) {}

  // Single volatile reference — readers always see a consistent (compressed, searchers) pair.
  private volatile QuantizationState quantizationState;

  private VamanaIndex(
      VamanaTopology topology,
      VamanaGraph graph,
      RandomAccessVectors vectors,
      SimilarityFunction sim) {
    this.topology = Objects.requireNonNull(topology, "topology must not be null");
    this.graph = graph; // nullable: null when wrapping a paged topology
    this.vectors = Objects.requireNonNull(vectors, "vectors must not be null");
    this.sim = Objects.requireNonNull(sim, "sim must not be null");
    this.threadLocalSearcher =
        ThreadLocal.withInitial(() -> new VamanaSearcher(topology, vectors, sim));
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
    SimilarityFunction localSim = this.sim;
    ThreadLocal<VamanaSearcher> tl =
        ThreadLocal.withInitial(
            () -> {
              NodeScorerFactory quantizedFactory =
                  query -> {
                    ScoreFunction sf = compressed.scoreFunctionFor(query, localSim);
                    return sf::score;
                  };
              return new VamanaSearcher(topology, vectors, localSim, quantizedFactory);
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
   * Pre-filtered search (ACORN): admits only ordinals accepted by {@code predicate} into the
   * results while still traversing through non-matching nodes, keeping recall high for
   * high-selectivity filters (I.5). Uses full-precision scoring.
   *
   * @param query the query vector
   * @param k number of results to return
   * @param searchListSize beam width L
   * @param predicate accepts an ordinal iff it is eligible for the result set
   * @return search result with ranked node IDs and scores (only predicate-accepted ordinals)
   */
  public SearchResult searchFiltered(
      float[] query, int k, int searchListSize, java.util.function.IntPredicate predicate) {
    return threadLocalSearcher.get().searchFiltered(query, k, searchListSize, predicate);
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
   * @param searchListSize beam width for the coarse pass
   * @param overQueryFactor multiplier for coarse-pass k (e.g., 2.0 retrieves 2*k candidates)
   * @return the top-k results after rescoring, sorted by score descending
   */
  public SearchResult searchTwoPass(
      float[] query, int k, int searchListSize, float overQueryFactor) {
    // Single volatile read — guaranteed to see a consistent (compressed, searchers) pair.
    QuantizationState qs = this.quantizationState;
    VamanaSearcher searcher = qs != null ? qs.searchers().get() : threadLocalSearcher.get();
    return searcher.searchTwoPass(query, k, searchListSize, overQueryFactor);
  }

  /** Two-pass search with default searchListSize=max(k,100) and overQueryFactor=2.0. */
  public SearchResult searchTwoPass(float[] query, int k) {
    return searchTwoPass(query, k, Math.max(k, 100), 2.0f);
  }

  /**
   * Creates a new full-precision searcher instance with its own scratch buffers.
   *
   * <p><b>Always returns a full-precision searcher</b>, regardless of whether quantization is
   * enabled. For quantized two-pass search, call {@link #searchTwoPass} directly instead.
   */
  public VamanaSearcher searcher() {
    return new VamanaSearcher(graph, vectors, sim);
  }

  /**
   * Returns the underlying heap graph for testing, inspection, and serialization, or {@code null}
   * when this index wraps a paged disk-resident topology (which has no heap graph to hand back).
   *
   * <p><b>Warning:</b> when non-null, the returned {@link VamanaGraph} is the live graph backing
   * this index. Mutating it (e.g., calling {@link VamanaGraph#getNeighbors(int)} and clearing the
   * returned array) will permanently corrupt the index. Use only for read-only inspection.
   */
  public VamanaGraph graph() {
    return graph;
  }

  /** Returns the number of vectors in the index. */
  public int size() {
    return topology.size();
  }

  /** Returns the vector dimension. */
  public int dimension() {
    return vectors.dimension();
  }

  /** Returns the similarity function used by this index. */
  public SimilarityFunction similarityFunction() {
    return sim;
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

  /**
   * Wraps a pre-built {@link VamanaGraph} together with its backing {@link RandomAccessVectors} and
   * similarity function into a searchable {@link VamanaIndex} WITHOUT running any graph
   * construction. The caller is responsible for ensuring that {@code graph}, {@code vectors}, and
   * {@code sim} are mutually consistent (same vector count, same metric used when the graph was
   * originally built, etc.).
   *
   * <p>This factory wraps a decoded graph for the persistence path in {@code vectors-db}: a decoded
   * graph from {@code VamanaGraphCodec.decode(byte[])} is paired with a {@code
   * MemorySegmentRandomAccessVectors} view of the mmap'd {@code vectors.bin} file, and the
   * resulting index is wrapped in {@code MappedVamanaPagedIndexAdapter} for read-only search. There
   * is no "rebuild-on-commit" concern here — the graph is loaded from disk and the index is never
   * mutated after construction.
   *
   * <p>The returned index shares its internal state with the arguments; in particular it does not
   * copy {@code graph} or {@code vectors}. Do NOT mutate either through any other reference once
   * this factory returns.
   *
   * @throws NullPointerException if any argument is null
   */
  public static VamanaIndex ofPrebuilt(
      VamanaGraph graph, RandomAccessVectors vectors, SimilarityFunction sim) {
    Objects.requireNonNull(graph, "graph must not be null");
    Objects.requireNonNull(vectors, "vectors must not be null");
    Objects.requireNonNull(sim, "sim must not be null");
    return new VamanaIndex(graph, graph, vectors, sim);
  }

  /**
   * Wraps a pre-built {@link VamanaTopology} (e.g. a disk-resident paged view of {@code graph.bin})
   * together with its backing {@link RandomAccessVectors} and similarity function, WITHOUT a heap
   * graph. {@link #graph()} returns {@code null} for the resulting index — it is read-only and is
   * never re-encoded. Used by the persistence path to serve search from an mmap'd graph without
   * inflating the full adjacency into heap.
   *
   * @throws NullPointerException if any argument is null
   */
  public static VamanaIndex ofPrebuilt(
      VamanaTopology topology, RandomAccessVectors vectors, SimilarityFunction sim) {
    Objects.requireNonNull(topology, "topology must not be null");
    Objects.requireNonNull(vectors, "vectors must not be null");
    Objects.requireNonNull(sim, "sim must not be null");
    return new VamanaIndex(topology, null, vectors, sim);
  }

  /** Builder for {@link VamanaIndex}. */
  public static final class Builder {

    private final RandomAccessVectors vectors;
    private final SimilarityFunction sim;
    private int maxDegree = 64;
    private int searchListSize = 128;
    private float alpha = 1.2f;
    private long seed = System.nanoTime();
    private int buildThreads = 1;

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

    /**
     * Sets the number of threads used for graph construction (default: 1 = deterministic,
     * sequential). When {@code threads > 1}, construction is routed through {@link
     * ConcurrentVamanaGraphBuilder} which partitions the two-pass insert across worker threads
     * guarded by per-node {@link java.util.concurrent.locks.ReentrantLock ReentrantLock}s. Search
     * and graph topology remain unchanged; recall is preserved within statistical noise.
     */
    public Builder buildThreads(int threads) {
      if (threads < 1) {
        throw new IllegalArgumentException("buildThreads must be >= 1: " + threads);
      }
      this.buildThreads = threads;
      return this;
    }

    /** Builds the index. */
    public VamanaIndex build() {
      VamanaGraph graph;
      if (buildThreads > 1) {
        graph =
            ConcurrentVamanaGraphBuilder.create(
                    maxDegree, searchListSize, alpha, vectors, sim, seed)
                .build(buildThreads);
      } else {
        graph =
            VamanaGraphBuilder.create(maxDegree, searchListSize, alpha, vectors, sim, seed).build();
      }
      return new VamanaIndex(graph, graph, vectors, sim);
    }
  }
}
