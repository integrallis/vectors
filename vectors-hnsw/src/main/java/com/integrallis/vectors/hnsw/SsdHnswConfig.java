package com.integrallis.vectors.hnsw;

/**
 * Configuration for SSD-resident HNSW graph traversal.
 *
 * <p>When a vector dataset is larger than available RAM, vectors are stored on an NVMe or SSD via
 * {@code MemorySegment} mmap. Each distance computation during graph traversal triggers a potential
 * OS page fault. {@link AsyncVectorPrefetcher} uses a small I/O thread pool to issue touch-reads
 * (page-ins) for the <em>next</em> batch of neighbor vectors before the current batch is scored,
 * hiding I/O latency behind computation.
 *
 * <h3>Parameter guidance</h3>
 *
 * <ul>
 *   <li>{@code ioThreads} — typically 2–8; one thread can sustain ~500 MB/s sequential reads on
 *       NVMe; set higher only when IOPS, not throughput, is the bottleneck.
 *   <li>{@code prefetchWindowSize} — number of graph hops to prefetch in advance; 1 (immediate
 *       neighbors only) is the default and safe choice.
 * </ul>
 *
 * @param ioThreads number of daemon threads in the prefetch thread pool
 * @param prefetchWindowSize number of neighbor batches to prefetch ahead (currently 1 is used)
 */
public record SsdHnswConfig(int ioThreads, int prefetchWindowSize) {

  /** Validates parameters at construction time. */
  public SsdHnswConfig {
    if (ioThreads < 1)
      throw new IllegalArgumentException("ioThreads must be >= 1, got " + ioThreads);
    if (prefetchWindowSize < 1)
      throw new IllegalArgumentException(
          "prefetchWindowSize must be >= 1, got " + prefetchWindowSize);
  }

  /**
   * Sensible defaults: 4 I/O threads, prefetch 1 hop ahead.
   *
   * <p>Works well on typical NVMe hardware with HNSW {@code M = 16}.
   */
  public static SsdHnswConfig defaults() {
    return new SsdHnswConfig(4, 1);
  }
}
