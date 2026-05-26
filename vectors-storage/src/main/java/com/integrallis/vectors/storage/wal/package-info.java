/**
 * Write-ahead log (WAL) implementations for durable mutation journaling.
 *
 * <ul>
 *   <li>{@link com.integrallis.vectors.storage.wal.WriteAheadLog} — sequence-numbered durability
 *       contract used by storage-backed indexes
 *   <li>{@link com.integrallis.vectors.storage.wal.BackendWriteAheadLog} — namespace-scoped WAL
 *       backed by any {@link com.integrallis.vectors.storage.backend.StorageBackend}
 *   <li>{@link com.integrallis.vectors.storage.wal.SegmentedWriteAheadLog} — local rolling
 *       CRC-checked segment files
 * </ul>
 */
package com.integrallis.vectors.storage.wal;
