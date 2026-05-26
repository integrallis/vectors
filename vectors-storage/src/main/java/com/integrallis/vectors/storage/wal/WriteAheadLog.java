/*
 * Copyright 2025-2026 Integrallis Software, LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.integrallis.vectors.storage.wal;

import java.io.Closeable;
import java.io.IOException;
import java.time.Duration;
import java.util.stream.Stream;

/**
 * Sequence-numbered durable write-ahead log with namespace-prefix layout, designed to run over any
 * {@link com.integrallis.vectors.storage.backend.StorageBackend} (in-heap, local filesystem, or
 * S3).
 *
 * <p>Per-namespace S3 prefix layout (Turbopuffer-aligned; canonical reference is §16.2 of {@code
 * vectors-distributed-design.md}):
 *
 * <pre>
 *   {root}/{namespace_id}/
 *       wal/
 *           000000000001.log     ← closed, indexed (■)
 *           000000000002.log     ← closed, indexed (■)
 *           000000000003.log     ← closed, NOT YET indexed (◈)
 *           000000000004.log     ← active, group-commit buffer; flushed every 1 s
 *       index/
 *           000000000001.idx     ← immutable run produced by the Indexer
 *           ...
 *       manifest.json            ← which wal/* are ■ vs ◈; current index epoch
 * </pre>
 *
 * <p>A query node scans the unindexed tail (◈ files) exhaustively for strong consistency, and uses
 * {@code index/*} for everything older.
 */
public interface WriteAheadLog extends Closeable {

  /**
   * Appends {@code entry} and returns the assigned sequence number once the entry is durable (its
   * enclosing WAL object has been written through to the underlying {@link
   * com.integrallis.vectors.storage.backend.StorageBackend}).
   *
   * <p>Concurrent {@code append} calls within a {@link #groupCommitInterval} window are coalesced
   * into a single WAL object to amortise PUT cost. Callers block until that single PUT completes.
   *
   * @param entry payload bytes; framed in storage with a 4-byte length prefix and 4-byte CRC32
   * @return monotonically increasing sequence number assigned to this entry
   * @throws IOException on storage failure
   */
  long append(byte[] entry) throws IOException;

  /**
   * Returns a stream of all entries with sequence numbers {@code >= fromSeqInclusive}, in order.
   * The stream may include in-flight (active-segment) entries that have already been durably
   * persisted by the time {@code readFrom} is invoked.
   *
   * <p>The returned {@link Stream} owns underlying resources and must be closed by the caller (use
   * try-with-resources).
   */
  Stream<WalEntry> readFrom(long fromSeqInclusive) throws IOException;

  /** Returns the last assigned sequence number, or {@code -1} if the log is empty. */
  long lastSequenceNumber();

  /**
   * Returns the sequence-number range descriptors for closed-but-unindexed (◈) segments plus the
   * active segment. A query node MUST scan these for strong-consistency reads. The array length is
   * two longs per range: {@code [start0, end0, start1, end1, ...]}.
   */
  long[] unindexedTailSeqs();

  /**
   * Marks all closed segments whose sequence range is fully contained in {@code [seqStartInclusive,
   * seqEndInclusive]} as indexed (■). Called by the Indexer after publishing the corresponding
   * {@code index/NNN.idx} run.
   *
   * @throws IOException on storage / manifest-write failure
   */
  void markIndexed(long seqStartInclusive, long seqEndInclusive) throws IOException;

  /**
   * Forces any buffered entries through the group-commit path immediately. Returns when the
   * resulting PUT (if any) has completed.
   */
  void flush() throws IOException;

  /**
   * The maximum write-side latency before buffered entries are flushed. Default 1 second; smaller
   * values give faster durability acks at higher PUT cost.
   */
  Duration groupCommitInterval();

  /** A single WAL entry with its assigned sequence number and original payload bytes. */
  record WalEntry(long sequenceNumber, byte[] data) {}
}
