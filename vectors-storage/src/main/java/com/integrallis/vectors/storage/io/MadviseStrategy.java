package com.integrallis.vectors.storage.io;

/**
 * Memory-mapped I/O advisory strategies. Applied via {@code posix_madvise()} on platforms that
 * support it (Linux, macOS). Silently ignored on unsupported platforms (Windows).
 */
public enum MadviseStrategy {

  /** Random access pattern. Best for ANN search with random graph traversal. Default strategy. */
  RANDOM,

  /** Sequential access pattern. Best for bulk writes, full scans, and compaction. */
  SEQUENTIAL,

  /** Advise the OS to preload pages. Best for hot data like upper HNSW layers. */
  WILLNEED,

  /** No advisory. Let the OS use its default page replacement policy. */
  NONE
}
