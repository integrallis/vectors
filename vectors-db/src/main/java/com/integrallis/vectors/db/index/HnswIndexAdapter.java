package com.integrallis.vectors.db.index;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.hnsw.HnswGraph;
import com.integrallis.vectors.hnsw.HnswGraphMerger;
import com.integrallis.vectors.hnsw.HnswIndex;
import com.integrallis.vectors.hnsw.InMemoryVectors;
import com.integrallis.vectors.hnsw.SearchResult;
import com.integrallis.vectors.quantization.CompressedVectors;
import java.util.Objects;
import java.util.function.IntPredicate;

/**
 * {@link IndexSpi} backed by an on-heap {@link HnswIndex}. Used by {@link
 * com.integrallis.vectors.db.VectorCollection} when {@code indexType == HNSW} in both in-memory
 * mode and as the staging builder during persistent commits (the commit pipeline calls {@link
 * #build(float[][], SimilarityFunction)} to construct a fresh graph from the successor vector
 * matrix, extracts the resulting {@link HnswGraph} via {@link #graph()}, and serializes it to
 * {@code graph.bin}).
 *
 * <p><b>Rebuild-on-commit.</b> Every call to {@link #build(float[][], SimilarityFunction)}
 * reconstructs the graph from scratch. This mirrors the flat-scan precedent ({@link
 * FlatScanAdapter#build}) and keeps the commit pipeline uniform — there is no incremental insertion
 * path in Step 4b. The caller guarantees that concurrent searches never overlap with a rebuild
 * (enforced by {@link com.integrallis.vectors.db.VectorCollection} via its writer lock).
 *
 * <p><b>Empty inputs.</b> {@link HnswIndex} cannot be built from zero vectors. When {@link
 * #build(float[][], SimilarityFunction)} receives an empty array, the adapter records the empty
 * state and {@link #search(float[], int, int, float) search} returns an empty {@link
 * SearchOutcome}. This is the "bootstrap" case used by {@link
 * com.integrallis.vectors.db.VectorCollection}'s in-memory open.
 *
 * <p><b>Thread-safety.</b> {@link HnswIndex} is thread-safe for concurrent search (its internal
 * {@code ThreadLocal<HnswSearcher>} pool) and single-threaded for construction. This adapter
 * inherits both guarantees: {@link #search} is thread-safe, {@link #build} is not.
 */
public final class HnswIndexAdapter implements IndexSpi {

  private final int maxConnections;
  private final int efConstruction;

  // Null until build() is called, or if build() was called with an empty vector array.
  private HnswIndex index;
  private int dimension;
  private int size;

  /**
   * Creates a new adapter with the given HNSW build parameters.
   *
   * @param maxConnections HNSW {@code M} (must be positive)
   * @param efConstruction beam width during construction (must be {@code >= maxConnections})
   * @throws IllegalArgumentException if either argument violates the contract
   */
  public HnswIndexAdapter(int maxConnections, int efConstruction) {
    if (maxConnections <= 0) {
      throw new IllegalArgumentException("maxConnections must be positive: " + maxConnections);
    }
    if (efConstruction < maxConnections) {
      throw new IllegalArgumentException(
          "efConstruction ("
              + efConstruction
              + ") must be >= maxConnections ("
              + maxConnections
              + ")");
    }
    this.maxConnections = maxConnections;
    this.efConstruction = efConstruction;
  }

  /** Returns the backing {@link HnswIndex} or {@code null} when the collection is empty. */
  HnswIndex index() {
    return index;
  }

  /** Returns {@code true} when no vectors have been indexed. */
  boolean isEmpty() {
    return size == 0 || index == null;
  }

  @Override
  public void build(float[][] vectors, SimilarityFunction metric) {
    Objects.requireNonNull(vectors, "vectors must not be null");
    Objects.requireNonNull(metric, "metric must not be null");
    this.size = vectors.length;
    this.dimension = vectors.length == 0 ? 0 : vectors[0].length;
    if (vectors.length == 0) {
      // HnswIndex rejects empty input in its builder. Record the empty state and short-circuit
      // search() below — matches the bootstrap path for an empty persistent/in-memory collection.
      // The metric is captured inside HnswIndex at build time; we don't need to retain a copy.
      this.index = null;
      return;
    }
    this.index =
        HnswIndex.builder(vectors, metric)
            .maxConnections(maxConnections)
            .efConstruction(efConstruction)
            .build();
  }

  @Override
  public SearchOutcome search(float[] query, int k, int searchListSize, float overQueryFactor) {
    Objects.requireNonNull(query, "query must not be null");
    if (k <= 0) {
      throw new IllegalArgumentException("k must be positive: " + k);
    }
    if (size == 0 || index == null) {
      return new SearchOutcome(new int[0], new float[0]);
    }
    if (query.length != dimension) {
      throw new IllegalArgumentException(
          "Query dimension " + query.length + " does not match index dimension " + dimension);
    }
    // efSearch must be >= k (HnswSearcher contract). searchListSize is the caller-supplied beam
    // width hint — we honor it but clamp to the floor of k.
    int efSearch = Math.max(searchListSize, k);
    SearchResult result;
    if (overQueryFactor > 1.0f && index.isQuantizationEnabled()) {
      // Two-pass: coarse quantized pass + full-precision rescore.
      result = index.searchTwoPass(query, k, efSearch, overQueryFactor);
    } else {
      // Single-pass full-precision search.
      result = index.search(query, k, efSearch);
    }
    // Defensive clone — SearchResult arrays are documented as "must not be mutated", but the
    // SearchOutcome contract does not constrain its ordinals/scores, and downstream callers
    // (post-filter, rescore) may mutate them in-place.
    return new SearchOutcome(result.nodeIds().clone(), result.scores().clone());
  }

  /**
   * ACORN-style pre-filtered search: delegates to {@link HnswIndex#searchFiltered} so the beam
   * search navigates non-matching nodes but only collects matching results.
   */
  @Override
  public SearchOutcome searchWithPredicate(
      float[] query, int k, int searchListSize, float overQueryFactor, IntPredicate predicate) {
    Objects.requireNonNull(query, "query must not be null");
    Objects.requireNonNull(predicate, "predicate must not be null");
    if (k <= 0) throw new IllegalArgumentException("k must be positive: " + k);
    if (size == 0 || index == null) return new SearchOutcome(new int[0], new float[0]);
    if (query.length != dimension) {
      throw new IllegalArgumentException(
          "Query dimension " + query.length + " does not match index dimension " + dimension);
    }
    int efSearch = Math.max(searchListSize, k);
    SearchResult result = index.searchFiltered(query, k, efSearch, predicate);
    return new SearchOutcome(result.nodeIds().clone(), result.scores().clone());
  }

  @Override
  public int size() {
    return size;
  }

  @Override
  public void close() {
    // No resources to release — the underlying HnswIndex holds only on-heap state.
  }

  /**
   * Attaches compressed vectors for two-pass quantized search. When active, {@link #search} with
   * {@code overQueryFactor > 1.0f} delegates to {@link HnswIndex#searchTwoPass}, which runs a
   * coarse quantized first pass followed by a full-precision rescore on the top candidates.
   *
   * @param compressed the compressed vectors produced by a quantizer's {@code encodeAll}
   * @throws IllegalStateException if this adapter has not been built yet
   */
  public void enableQuantization(CompressedVectors compressed) {
    Objects.requireNonNull(compressed, "compressed must not be null");
    if (index == null) {
      throw new IllegalStateException("Cannot enable quantization on an unbuilt or empty adapter");
    }
    index.enableQuantization(compressed);
  }

  /**
   * Returns the underlying {@link HnswGraph}, or {@code null} if this adapter has not been built
   * yet or was built from an empty vector array. Used exclusively by the persistent commit pipeline
   * in {@code VectorCollectionImpl} to extract the graph for serialization via {@code
   * HnswGraphCodec.encode}. Callers must not mutate the returned graph.
   */
  public HnswGraph graph() {
    return index == null ? null : index.graph();
  }

  /**
   * Merges the old HNSW graph into a new index by remapping surviving nodes and repairing
   * under-connected ones, without full reconstruction (IGTM compact path).
   *
   * <p>This is the incremental alternative to {@link #build(float[][], SimilarityFunction)} used by
   * {@code VectorCollectionImpl.compactInMemory} when a pre-existing graph is available. The cost
   * is O(N' · M · d) for edge remapping plus O(R · ef · M) for repair, where R is the number of
   * under-connected nodes after deletion. Both are substantially cheaper than O(N' · log N' · M ·
   * d) for a full rebuild when the deletion fraction is small.
   *
   * @param old the pre-compaction HNSW graph
   * @param newVectors dense array of surviving vectors indexed by new ordinal
   * @param oldToNew mapping: {@code oldToNew[oldOrdinal]} = new ordinal, or {@code -1} if deleted
   * @param metric similarity function (must match the one used to build {@code old})
   */
  public void mergeFrom(
      HnswGraph old, float[][] newVectors, int[] oldToNew, SimilarityFunction metric) {
    Objects.requireNonNull(old, "old must not be null");
    Objects.requireNonNull(newVectors, "newVectors must not be null");
    Objects.requireNonNull(oldToNew, "oldToNew must not be null");
    Objects.requireNonNull(metric, "metric must not be null");
    this.size = newVectors.length;
    this.dimension = newVectors.length == 0 ? 0 : newVectors[0].length;
    if (newVectors.length == 0) {
      this.index = null;
      return;
    }
    HnswGraph merged =
        HnswGraphMerger.merge(old, newVectors, oldToNew, metric, maxConnections, efConstruction);
    if (merged == null) {
      this.index = null;
      return;
    }
    this.index = HnswIndex.ofPrebuilt(merged, new InMemoryVectors(newVectors), metric);
  }

  /** Returns the HNSW {@code M} parameter this adapter was configured with. */
  public int maxConnections() {
    return maxConnections;
  }

  /** Returns the HNSW {@code efConstruction} parameter this adapter was configured with. */
  public int efConstruction() {
    return efConstruction;
  }
}
