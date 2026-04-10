/**
 * Embedded vector database facade. Provides {@link com.integrallis.vectors.db.VectorCollection},
 * {@link com.integrallis.vectors.db.Document}, {@link com.integrallis.vectors.db.MetadataValue},
 * {@link com.integrallis.vectors.db.SearchRequest}, and {@link
 * com.integrallis.vectors.db.SearchResult}.
 *
 * <p>Step 2 of the v0.1 roadmap: in-memory only, flat-scan reference backend, no persistence, no
 * filter execution, no HNSW/Vamana adapters. Subsequent steps fill in the rest of the design doc
 * (see {@code design-embedded-vector-db.md}).
 */
package com.integrallis.vectors.db;
