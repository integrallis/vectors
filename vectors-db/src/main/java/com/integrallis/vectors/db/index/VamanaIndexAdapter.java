package com.integrallis.vectors.db.index;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.vamana.SearchResult;
import com.integrallis.vectors.vamana.VamanaGraph;
import com.integrallis.vectors.vamana.VamanaIndex;
import java.util.Objects;

/**
 * {@link IndexSpi} backed by an on-heap {@link VamanaIndex}. Used by {@link
 * com.integrallis.vectors.db.VectorCollection} when {@code indexType == VAMANA} in both in-memory
 * mode and as the staging builder during persistent commits (the commit pipeline calls {@link
 * #build(float[][], SimilarityFunction)} to construct a fresh graph from the successor vector
 * matrix, extracts the resulting {@link VamanaGraph} via {@link #graph()}, and serializes it to
 * {@code graph.bin}).
 *
 * <p><b>Rebuild-on-commit.</b> Every call to {@link #build(float[][], SimilarityFunction)}
 * reconstructs the graph from scratch. This mirrors {@link HnswIndexAdapter} and {@link
 * FlatScanAdapter#build}, keeping the commit pipeline uniform — there is no incremental insertion
 * path in Step 4c. The caller guarantees that concurrent searches never overlap with a rebuild
 * (enforced by {@link com.integrallis.vectors.db.VectorCollection} via its writer lock).
 *
 * <p><b>Empty inputs.</b> {@link VamanaIndex} cannot be built from zero vectors. When {@link
 * #build(float[][], SimilarityFunction)} receives an empty array, the adapter records the empty
 * state and {@link #search(float[], int, int, float) search} returns an empty {@link
 * SearchOutcome}. This is the "bootstrap" case used by {@link
 * com.integrallis.vectors.db.VectorCollection}'s in-memory open.
 *
 * <p><b>Thread-safety.</b> {@link VamanaIndex} is thread-safe for concurrent search (its internal
 * {@code ThreadLocal<VamanaSearcher>} pool) and single-threaded for construction. This adapter
 * inherits both guarantees: {@link #search} is thread-safe, {@link #build} is not.
 */
public final class VamanaIndexAdapter implements IndexSpi {

  private final int maxDegree;
  private final int searchListSize;
  private final float alpha;
  private final long seed;

  // Null until build() is called, or if build() was called with an empty vector array.
  private VamanaIndex index;
  private int dimension;
  private int size;

  /**
   * Creates a new adapter with the given Vamana build parameters.
   *
   * @param maxDegree Vamana {@code R} (must be positive)
   * @param searchListSize Vamana {@code L} — beam width during construction (must be {@code >=
   *     maxDegree})
   * @param alpha diversity parameter (must be {@code >= 1.0})
   * @param seed random seed for deterministic construction
   * @throws IllegalArgumentException if any argument violates its contract
   */
  public VamanaIndexAdapter(int maxDegree, int searchListSize, float alpha, long seed) {
    if (maxDegree <= 0) {
      throw new IllegalArgumentException("maxDegree must be positive: " + maxDegree);
    }
    if (searchListSize < maxDegree) {
      throw new IllegalArgumentException(
          "searchListSize (" + searchListSize + ") must be >= maxDegree (" + maxDegree + ")");
    }
    if (alpha < 1.0f) {
      throw new IllegalArgumentException("alpha must be >= 1.0: " + alpha);
    }
    this.maxDegree = maxDegree;
    this.searchListSize = searchListSize;
    this.alpha = alpha;
    this.seed = seed;
  }

  @Override
  public void build(float[][] vectors, SimilarityFunction metric) {
    Objects.requireNonNull(vectors, "vectors must not be null");
    Objects.requireNonNull(metric, "metric must not be null");
    this.size = vectors.length;
    this.dimension = vectors.length == 0 ? 0 : vectors[0].length;
    if (vectors.length == 0) {
      // VamanaIndex rejects empty input in its builder. Record the empty state and short-circuit
      // search() below — matches the bootstrap path for an empty persistent/in-memory collection.
      // The metric is captured inside VamanaIndex at build time; we don't need to retain a copy.
      this.index = null;
      return;
    }
    this.index =
        VamanaIndex.builder(vectors, metric)
            .maxDegree(maxDegree)
            .searchListSize(searchListSize)
            .alpha(alpha)
            .seed(seed)
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
    // L must be >= k (VamanaSearcher contract). searchListSize is the caller-supplied beam width
    // hint — we honor it but clamp to the floor of k. overQueryFactor is ignored on the
    // single-pass full-precision path; the two-pass path is reserved for when quantization lands.
    int searchL = Math.max(searchListSize, k);
    SearchResult result = index.search(query, k, searchL);
    // Defensive clone — SearchResult arrays are documented as "must not be mutated", but the
    // SearchOutcome contract does not constrain its ordinals/scores, and downstream callers
    // (post-filter, rescore) may mutate them in-place.
    return new SearchOutcome(result.nodeIds().clone(), result.scores().clone());
  }

  @Override
  public int size() {
    return size;
  }

  @Override
  public void close() {
    // No resources to release — the underlying VamanaIndex holds only on-heap state.
  }

  /**
   * Returns the underlying {@link VamanaGraph}, or {@code null} if this adapter has not been built
   * yet or was built from an empty vector array. Used exclusively by the persistent commit pipeline
   * in {@code VectorCollectionImpl} to extract the graph for serialization via {@code
   * VamanaGraphCodec.encode}. Callers must not mutate the returned graph.
   */
  public VamanaGraph graph() {
    return index == null ? null : index.graph();
  }

  /** Returns the Vamana {@code R} (maxDegree) parameter this adapter was configured with. */
  public int maxDegree() {
    return maxDegree;
  }

  /** Returns the Vamana {@code L} (searchListSize) parameter this adapter was configured with. */
  public int searchListSize() {
    return searchListSize;
  }

  /** Returns the diversity parameter {@code alpha} this adapter was configured with. */
  public float alpha() {
    return alpha;
  }

  /** Returns the random seed this adapter was configured with. */
  public long seed() {
    return seed;
  }
}
