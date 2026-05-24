/**
 * Embedded vector database facade. Provides {@link com.integrallis.vectors.db.VectorCollection},
 * {@link com.integrallis.vectors.core.Document}, {@link
 * com.integrallis.vectors.core.MetadataValue}, {@link com.integrallis.vectors.db.SearchRequest},
 * and {@link com.integrallis.vectors.db.SearchResult}.
 *
 * <p>Collections support in-memory and mmap-backed persistent storage, exact flat scan, graph and
 * inverted-file indexes, metadata filtering, tombstones, and optional quantized search paths.
 */
package com.integrallis.vectors.db;
