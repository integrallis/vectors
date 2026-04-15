package com.integrallis.vectors.db.index;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.ivf.IvfBuildParams;
import com.integrallis.vectors.ivf.IvfIndex;
import com.integrallis.vectors.ivf.IvfSearchRequest;
import com.integrallis.vectors.ivf.IvfSearchResult;
import java.util.Objects;

/**
 * In-memory {@link IndexSpi} backed by {@link IvfIndex}.
 *
 * <p>Used when {@code indexType == IVF_FLAT} in both in-memory mode and as the staging builder
 * during persistent commits (the commit pipeline calls {@link #build(float[][],
 * SimilarityFunction)} to construct a fresh IVF index from the successor vector matrix, extracts
 * the encoded bytes via {@link #encodedIndex()}, and serializes them to {@code graph.bin}).
 *
 * <p>Every call to {@link #build(float[][], SimilarityFunction)} reconstructs the index from
 * scratch via K-Means. Concurrent searches are safe (IvfIndex is immutable after construction);
 * concurrent builds are not (single-threaded build contract from IndexSpi).
 */
public final class IvfFlatAdapter implements IndexSpi {

  private final int k;
  private final int nprobe;
  private final int maxIter;
  private final float gamma;
  private final boolean soar;
  private final long seed;

  // Null until build() is called or when built from an empty array.
  private IvfIndex index;
  private int dimension;
  private int size;

  /**
   * Creates an adapter with IVF build and search parameters.
   *
   * @param k number of clusters (must be positive)
   * @param nprobe clusters probed per query (must be in [1, k])
   * @param maxIter maximum K-Means iterations (must be positive)
   * @param gamma SOAR spill ratio at search time
   * @param soar enable SOAR spill at search time
   * @param seed RNG seed for K-Means++
   */
  public IvfFlatAdapter(int k, int nprobe, int maxIter, float gamma, boolean soar, long seed) {
    if (k <= 0) throw new IllegalArgumentException("k must be positive: " + k);
    if (nprobe <= 0) throw new IllegalArgumentException("nprobe must be positive: " + nprobe);
    if (maxIter <= 0) throw new IllegalArgumentException("maxIter must be positive: " + maxIter);
    this.k = k;
    this.nprobe = nprobe;
    this.maxIter = maxIter;
    this.gamma = gamma;
    this.soar = soar;
    this.seed = seed;
  }

  @Override
  public void build(float[][] vectors, SimilarityFunction metric) {
    Objects.requireNonNull(vectors, "vectors must not be null");
    Objects.requireNonNull(metric, "metric must not be null");
    this.size = vectors.length;
    this.dimension = vectors.length == 0 ? 0 : vectors[0].length;
    if (vectors.length == 0) {
      this.index = null;
      return;
    }
    int effectiveK = Math.min(k, vectors.length);
    IvfBuildParams params = new IvfBuildParams(effectiveK, maxIter, 0f, soar, seed);
    this.index = IvfIndex.build(vectors, null, metric, params);
  }

  @Override
  public SearchOutcome search(float[] query, int k, int searchListSize, float overQueryFactor) {
    Objects.requireNonNull(query, "query must not be null");
    if (k <= 0) throw new IllegalArgumentException("k must be positive: " + k);
    if (size == 0 || index == null) {
      return new SearchOutcome(new int[0], new float[0]);
    }
    if (query.length != dimension) {
      throw new IllegalArgumentException(
          "Query dimension " + query.length + " does not match index dimension " + dimension);
    }
    int effectiveNprobe = Math.min(nprobe, index.k());
    IvfSearchRequest req = new IvfSearchRequest(query, k, effectiveNprobe, gamma, -Float.MAX_VALUE);
    IvfSearchResult result = index.search(req);
    int sz = result.hits().size();
    int[] ordinals = new int[sz];
    float[] scores = new float[sz];
    for (int i = 0; i < sz; i++) {
      ordinals[i] = result.hits().get(i).ordinal();
      scores[i] = result.hits().get(i).score();
    }
    return new SearchOutcome(ordinals, scores);
  }

  @Override
  public int size() {
    return size;
  }

  /**
   * Returns the encoded bytes of the underlying {@link IvfIndex}, or {@code null} if the adapter
   * has not been built yet or was built from an empty array. Used by the persistent commit pipeline
   * to write {@code graph.bin}. Callers must not mutate the returned array.
   */
  public byte[] encodedIndex() {
    return index == null ? null : index.encode();
  }

  @Override
  public void close() {
    // no off-heap resources
  }
}
