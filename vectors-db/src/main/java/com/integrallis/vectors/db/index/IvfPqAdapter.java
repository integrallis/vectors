package com.integrallis.vectors.db.index;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.ivf.IvfBuildParams;
import com.integrallis.vectors.ivf.IvfIndex;
import com.integrallis.vectors.ivf.IvfSearchRequest;
import com.integrallis.vectors.ivf.IvfSearchResult;
import java.util.Objects;

/**
 * In-memory {@link IndexSpi} backed by {@link IvfIndex} with product quantisation enabled.
 *
 * <p>Used when {@code indexType == IVF_PQ}. Shares the IVF-flat build/search scaffolding but trains
 * a {@code ProductQuantizer} on per-cluster residuals and searches via asymmetric-distance
 * computation (ADC) with an optional full-precision rescore pass controlled by {@code
 * rescoreFactor}.
 *
 * <p>Every call to {@link #build(float[][], SimilarityFunction)} reconstructs the index from
 * scratch — both K-Means clustering and PQ codebook training. Concurrent searches are safe ({@link
 * IvfIndex} is immutable after construction); concurrent builds are not.
 */
public final class IvfPqAdapter implements IndexSpi {

  private final int k;
  private final int nprobe;
  private final int maxIter;
  private final float gamma;
  private final boolean soar;
  private final long seed;
  private final int pqSubspaces;
  private final int pqClusters;
  private final float pqAnisotropicThreshold;
  private final int rescoreFactor;

  private IvfIndex index;
  private int dimension;
  private int size;

  /**
   * Creates an adapter with IVF + PQ build and search parameters.
   *
   * @param k IVF clusters (must be positive)
   * @param nprobe clusters probed per query (must be positive)
   * @param maxIter max K-Means iterations (must be positive)
   * @param gamma SOAR spill ratio at search time
   * @param soar enable SOAR spill at search time
   * @param seed RNG seed for K-Means++
   * @param pqSubspaces PQ M (must be positive; should divide the vector dimension)
   * @param pqClusters PQ Ks (must be in {@code [2, 256]})
   * @param pqAnisotropicThreshold anisotropic threshold; {@code -1f} for standard PQ, {@code [0,
   *     1]} for anisotropic encoding
   * @param rescoreFactor wide-heap multiplier; must be {@code >= 1}
   */
  public IvfPqAdapter(
      int k,
      int nprobe,
      int maxIter,
      float gamma,
      boolean soar,
      long seed,
      int pqSubspaces,
      int pqClusters,
      float pqAnisotropicThreshold,
      int rescoreFactor) {
    if (k <= 0) throw new IllegalArgumentException("k must be positive: " + k);
    if (nprobe <= 0) throw new IllegalArgumentException("nprobe must be positive: " + nprobe);
    if (maxIter <= 0) throw new IllegalArgumentException("maxIter must be positive: " + maxIter);
    if (pqSubspaces <= 0)
      throw new IllegalArgumentException("pqSubspaces must be positive: " + pqSubspaces);
    if (pqClusters < 2 || pqClusters > 256)
      throw new IllegalArgumentException("pqClusters must be in [2, 256]: " + pqClusters);
    if (rescoreFactor < 1)
      throw new IllegalArgumentException("rescoreFactor must be >= 1: " + rescoreFactor);
    this.k = k;
    this.nprobe = nprobe;
    this.maxIter = maxIter;
    this.gamma = gamma;
    this.soar = soar;
    this.seed = seed;
    this.pqSubspaces = pqSubspaces;
    this.pqClusters = pqClusters;
    this.pqAnisotropicThreshold = pqAnisotropicThreshold;
    this.rescoreFactor = rescoreFactor;
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
    IvfBuildParams base = new IvfBuildParams(effectiveK, maxIter, 0f, soar, seed, 0);
    IvfBuildParams params = base.withPq(pqSubspaces, pqClusters, pqAnisotropicThreshold);
    this.index = IvfIndex.build(vectors, null, metric, params);
  }

  /**
   * Searches the IVF-PQ index using the configured {@code rescoreFactor}. When {@code
   * overQueryFactor > 1.0}, the rescore width is multiplied further, mirroring the {@link
   * com.integrallis.vectors.db.index.HnswIndexAdapter} contract for caller-driven recall/latency
   * tradeoffs.
   */
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

    int effectiveNprobe = (searchListSize > 0) ? searchListSize : nprobe;
    boolean twoPass = overQueryFactor > 1.0f;
    int probeCount =
        twoPass
            ? Math.min((int) Math.ceil(effectiveNprobe * overQueryFactor), index.k())
            : Math.min(effectiveNprobe, index.k());
    int effectiveRescore =
        twoPass
            ? Math.max(rescoreFactor, (int) Math.ceil(rescoreFactor * overQueryFactor))
            : rescoreFactor;

    IvfSearchRequest req =
        new IvfSearchRequest(query, k, probeCount, gamma, -Float.MAX_VALUE, effectiveRescore);
    IvfSearchResult result = index.search(req);

    int sz = Math.min(result.hits().size(), k);
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

  /** Encoded bytes of the underlying IVF-PQ index (codebooks + codes + routing state). */
  public byte[] encodedIndex() {
    return index == null ? null : index.encode();
  }

  @Override
  public void close() {
    // no off-heap resources
  }
}
