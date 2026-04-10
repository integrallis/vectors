package com.integrallis.vectors.db;

/**
 * Index backend selector. Step 2 only wires {@link #FLAT}; other values are reserved for subsequent
 * steps and throw {@link UnsupportedOperationException} at build time.
 */
public enum IndexType {
  /** Brute-force linear scan. Reference implementation, always available. */
  FLAT,

  /** Hierarchical Navigable Small World (HNSW) graph index. Deferred to Step 4. */
  HNSW,

  /** Vamana/DiskANN graph index. Deferred to Step 6. */
  VAMANA,

  /** Inverted-file (flat) clustering index. Deferred to a later step. */
  IVF_FLAT
}
