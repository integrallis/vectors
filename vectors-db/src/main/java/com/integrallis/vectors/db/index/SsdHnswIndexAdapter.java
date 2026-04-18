package com.integrallis.vectors.db.index;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.hnsw.AsyncVectorPrefetcher;
import com.integrallis.vectors.hnsw.HnswIndex;
import com.integrallis.vectors.hnsw.RandomAccessVectors;
import com.integrallis.vectors.hnsw.SearchResult;
import com.integrallis.vectors.hnsw.SsdHnswConfig;
import java.util.Objects;
import java.util.function.IntPredicate;

/**
 * SSD-aware HNSW index adapter for {@link com.integrallis.vectors.db.VectorCollection}.
 *
 * <p>Extends the standard in-memory HNSW search path with prefetch-aware traversal suitable for
 * datasets whose vectors live in a mmap-backed {@link
 * com.integrallis.vectors.storage.MappedVectorStore}.
 *
 * <p><b>How it works</b>: During beam search, before scoring each candidate's neighbor array, the
 * {@link AsyncVectorPrefetcher} submits async touch-reads for all neighbor ordinals. For
 * mmap-backed stores this causes the OS to populate the corresponding 4 KiB pages in the page cache
 * concurrently with the current scoring work, hiding read latency behind computation.
 *
 * <p><b>Fallback</b>: When the underlying vector source is already fully in RAM, the touch-reads
 * are cheap array accesses and the adapter behaves identically to {@link HnswIndexAdapter}.
 *
 * <p>The prefetcher thread pool is shut down via {@link #close()}.
 */
public final class SsdHnswIndexAdapter implements IndexSpi, AutoCloseable {

  private final HnswIndexAdapter delegate;
  private final SsdHnswConfig ssdConfig;
  private AsyncVectorPrefetcher prefetcher;

  /**
   * Constructs an SSD HNSW adapter.
   *
   * @param maxConnections HNSW {@code M} parameter
   * @param efConstruction HNSW construction beam width
   * @param ssdConfig SSD prefetch configuration
   */
  public SsdHnswIndexAdapter(int maxConnections, int efConstruction, SsdHnswConfig ssdConfig) {
    this.delegate = new HnswIndexAdapter(maxConnections, efConstruction);
    this.ssdConfig = Objects.requireNonNull(ssdConfig, "ssdConfig must not be null");
  }

  // ---------------------------------------------------------------------------
  // IndexSpi implementation
  // ---------------------------------------------------------------------------

  @Override
  public void build(float[][] vectors, SimilarityFunction metric) {
    delegate.build(vectors, metric);
    // Wire the prefetcher against the same RandomAccessVectors that backs the delegate index.
    // For mmap-backed stores, swap in the mapped source before calling build().
    HnswIndex idx = delegate.index();
    if (idx != null) {
      RandomAccessVectors src = idx.vectorSource();
      prefetcher = new AsyncVectorPrefetcher(src, ssdConfig.ioThreads());
    }
  }

  @Override
  public SearchOutcome search(float[] query, int k, int searchListSize, float overQueryFactor) {
    Objects.requireNonNull(query, "query must not be null");
    if (k <= 0) throw new IllegalArgumentException("k must be positive: " + k);
    if (delegate.isEmpty()) return new SearchOutcome(new int[0], new float[0]);

    int efSearch = Math.max(searchListSize, k);
    SearchResult result;
    if (prefetcher != null) {
      result = delegate.index().searchWithPrefetch(query, k, efSearch, prefetcher);
    } else {
      result = delegate.index().search(query, k, efSearch);
    }
    return new SearchOutcome(
        result.nodeIds().clone(), result.scores().clone()); // nodeIds in hnsw = ordinals in db
  }

  @Override
  public SearchOutcome searchWithPredicate(
      float[] query, int k, int searchListSize, float overQueryFactor, IntPredicate predicate) {
    // Predicate-filtered search delegates to the standard adapter (no prefetch).
    return delegate.searchWithPredicate(query, k, searchListSize, overQueryFactor, predicate);
  }

  @Override
  public int size() {
    return delegate.size();
  }

  /** Returns the number of prefetch requests submitted since the last build. */
  public long prefetchCount() {
    return prefetcher != null ? prefetcher.submittedCount() : 0L;
  }

  /** Returns the underlying {@link SsdHnswConfig}. */
  public SsdHnswConfig ssdConfig() {
    return ssdConfig;
  }

  @Override
  public void close() {
    if (prefetcher != null) {
      prefetcher.close();
      prefetcher = null;
    }
    delegate.close();
  }
}
