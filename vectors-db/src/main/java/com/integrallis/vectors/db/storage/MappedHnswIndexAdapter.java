package com.integrallis.vectors.db.storage;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.index.IndexSpi;
import com.integrallis.vectors.hnsw.HnswGraph;
import com.integrallis.vectors.hnsw.HnswIndex;
import com.integrallis.vectors.hnsw.RandomAccessVectors;
import com.integrallis.vectors.hnsw.SearchResult;
import com.integrallis.vectors.quantization.CompressedVectors;
import java.util.Objects;

/**
 * Read-only {@link IndexSpi} that serves HNSW search from a <b>pre-built</b> {@link HnswGraph}
 * wrapped around a {@link RandomAccessVectors} view of an mmap'd {@code vectors.bin}. This is the
 * persistent-HNSW analogue of {@link com.integrallis.vectors.db.index.HnswIndexAdapter} and the
 * Step 4b Phase 5 open-path terminus: {@code VectorCollectionImpl.openGeneration} decodes {@code
 * graph.bin} via {@link HnswGraphCodec#decode(byte[])}, wraps the per-generation {@link
 * MemorySegmentVectors} in a {@link MemorySegmentRandomAccessVectors}, and hands both to this
 * adapter's constructor.
 *
 * <p><b>Construction.</b> Unlike {@link com.integrallis.vectors.db.index.HnswIndexAdapter}, this
 * adapter does <b>not</b> support {@link IndexSpi#build(float[][], SimilarityFunction)} — its data
 * source is an already-built graph that came out of the Step 4b generation-write pipeline.
 * Construct one via {@link #MappedHnswIndexAdapter(HnswGraph, RandomAccessVectors,
 * SimilarityFunction)}; calling {@link #build} throws {@link UnsupportedOperationException}. This
 * invariant is load-bearing: {@link MemorySegmentRandomAccessVectors} returns a per-thread scratch
 * buffer from {@link RandomAccessVectors#getVector(int)}, and only the HNSW <i>search</i> path
 * (which holds at most one scratch reference per inner iteration) is safe with that contract. The
 * HNSW <i>build</i> path ({@code HnswGraphBuilder.insert}) holds a query vector across many {@code
 * getVector} calls and would corrupt under shared-scratch — so {@code build} must never reach this
 * adapter.
 *
 * <p><b>Thread safety.</b> Safe for concurrent calls to {@link #search(float[], int, int, float)}
 * from any number of threads. The underlying {@link HnswIndex} owns a {@link ThreadLocal} pool of
 * {@code HnswSearcher} instances (one per calling thread), and each searcher's scratch buffers are
 * private. The underlying {@link RandomAccessVectors} is {@link MemorySegmentRandomAccessVectors},
 * which is also thread-safe via per-thread scratch {@code float[]}.
 *
 * <p><b>Close semantics.</b> {@link #close()} is a <b>no-op</b>. The underlying {@link
 * MemorySegmentVectors}' lifetime is owned by the caller-provided per-generation {@code Arena};
 * {@code VectorCollectionImpl} closes that arena exactly once when its refcount drops to zero.
 * {@link HnswGraph} is purely on-heap and holds no releasable resources.
 */
public final class MappedHnswIndexAdapter implements IndexSpi {

  private final HnswIndex index;
  private final int dimension;

  /**
   * Wraps a pre-built {@link HnswGraph} together with its backing {@link RandomAccessVectors} and
   * similarity function. None of the arguments may be {@code null}.
   *
   * @param graph the decoded HNSW graph — do NOT mutate via {@link HnswGraph#initNode} or {@link
   *     HnswGraph#setEntryNode} after this call, doing so would corrupt in-flight searches
   * @param vectors read-only random access to the vectors the graph's node IDs refer to; typically
   *     a {@link MemorySegmentRandomAccessVectors} wrapping the generation's mmap'd {@code
   *     vectors.bin}
   * @param metric the similarity function to use for scoring — must match the metric that was used
   *     when the graph was originally built, or query results will be silently wrong
   * @throws NullPointerException if any argument is null
   */
  public MappedHnswIndexAdapter(
      HnswGraph graph, RandomAccessVectors vectors, SimilarityFunction metric) {
    Objects.requireNonNull(graph, "graph must not be null");
    Objects.requireNonNull(vectors, "vectors must not be null");
    Objects.requireNonNull(metric, "metric must not be null");
    this.index = HnswIndex.ofPrebuilt(graph, vectors, metric);
    this.dimension = vectors.dimension();
  }

  /**
   * Not supported. A {@code MappedHnswIndexAdapter} is always constructed from a pre-built {@link
   * HnswGraph} + mmap'd {@link RandomAccessVectors}; there is no "build from an in-memory {@code
   * float[][]}" path here because the data source is an on-disk file produced by the
   * generation-write pipeline.
   *
   * <p><b>This method must remain unsupported</b>. Routing build through this adapter would invoke
   * {@code HnswGraphBuilder.insert}, which holds a query vector across multiple {@code getVector}
   * calls on the backing {@link RandomAccessVectors}. When that backing store is a {@link
   * MemorySegmentRandomAccessVectors} (the sole production caller), the per-thread scratch
   * invariant would be violated and the graph would be built from silently corrupt data.
   *
   * @throws UnsupportedOperationException always
   */
  @Override
  public void build(float[][] vectors, SimilarityFunction metric) {
    throw new UnsupportedOperationException(
        "MappedHnswIndexAdapter is read-only; use HnswIndexAdapter for in-memory HNSW or"
            + " construct a MappedHnswIndexAdapter directly from a decoded HnswGraph");
  }

  @Override
  public SearchOutcome search(float[] query, int k, int searchListSize, float overQueryFactor) {
    Objects.requireNonNull(query, "query must not be null");
    if (k <= 0) {
      throw new IllegalArgumentException("k must be positive: " + k);
    }
    int size = index.size();
    if (size == 0) {
      return new SearchOutcome(new int[0], new float[0]);
    }
    if (query.length != dimension) {
      throw new IllegalArgumentException(
          "Query dimension " + query.length + " does not match index dimension " + dimension);
    }
    int efSearch = Math.max(searchListSize, k);
    SearchResult result;
    if (overQueryFactor > 1.0f && index.isQuantizationEnabled()) {
      result = index.searchTwoPass(query, k, efSearch, overQueryFactor);
    } else {
      result = index.search(query, k, efSearch);
    }
    return new SearchOutcome(result.nodeIds().clone(), result.scores().clone());
  }

  /**
   * Attaches compressed vectors for quantized two-pass search. Once enabled, {@link
   * #search(float[], int, int, float)} will delegate to {@link HnswIndex#searchTwoPass} when {@code
   * overQueryFactor > 1.0f}.
   *
   * @param compressed the compressed vectors — must match the index's size and dimension
   * @throws NullPointerException if compressed is null
   */
  public void enableQuantization(CompressedVectors compressed) {
    Objects.requireNonNull(compressed, "compressed must not be null");
    index.enableQuantization(compressed);
  }

  @Override
  public int size() {
    return index.size();
  }

  /**
   * Returns the underlying {@link HnswGraph}. Used by the compaction pipeline in {@code
   * VectorCollectionImpl.compactPersistent} to extract the pre-built graph for incremental merge
   * via {@link com.integrallis.vectors.hnsw.HnswGraphMerger}. Callers must not mutate the returned
   * graph via {@link HnswGraph#initNode} or {@link HnswGraph#setEntryNode}.
   */
  public HnswGraph graph() {
    return index.graph();
  }

  /**
   * No-op. The underlying {@link MemorySegmentVectors}'s lifetime is tied to its per-generation
   * {@link java.lang.foreign.Arena}; {@code VectorCollectionImpl} closes that arena exactly once
   * per retired generation. The {@link HnswGraph} and {@link HnswIndex} are purely on-heap and need
   * no explicit release.
   */
  @Override
  public void close() {
    // no-op — arena owns the mmap lifetime; graph is on-heap
  }
}
