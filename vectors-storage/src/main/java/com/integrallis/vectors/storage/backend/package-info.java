/**
 * Storage backend abstraction for pluggable key-value stores.
 *
 * <ul>
 *   <li>{@link com.integrallis.vectors.storage.backend.StorageBackend} — interface
 *   <li>{@link com.integrallis.vectors.storage.backend.HeapStorageBackend} — in-heap (tests)
 *   <li>{@link com.integrallis.vectors.storage.backend.LocalFileStorageBackend} — local filesystem
 *   <li>{@link com.integrallis.vectors.storage.backend.S3StorageBackend} — S3-compatible object
 *       stores
 * </ul>
 */
package com.integrallis.vectors.storage.backend;
