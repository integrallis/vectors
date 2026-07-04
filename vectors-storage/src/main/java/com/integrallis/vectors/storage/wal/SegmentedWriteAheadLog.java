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

import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;
import java.util.stream.Stream;
import java.util.zip.CRC32;

/**
 * Append-only write-ahead log backed by rolling segment files on the local filesystem.
 *
 * <p>Binary entry format (per append):
 *
 * <pre>
 *   [4 bytes: payload length, big-endian int]
 *   [N bytes: payload]
 *   [4 bytes: CRC32 of payload, big-endian int]
 * </pre>
 *
 * <p>Segment files are named {@code wal-<20-digit-seq>.seg} where {@code seq} is the sequence
 * number of the first entry in that segment. Segments are created <i>lazily</i>: a new segment file
 * appears on disk only on the first {@link #append} after construction or after {@link #seal()}.
 * This guarantees that every segment file on disk holds at least one entry — there are never
 * zero-byte trailing segments left behind by a {@code seal()}-then-{@code close()} with no
 * intervening append.
 *
 * <p>This class is thread-compatible: external synchronisation is required for concurrent appends.
 */
public final class SegmentedWriteAheadLog implements WriteAheadLog {

  private static final String SEGMENT_PREFIX = "wal-";
  private static final String SEGMENT_SUFFIX = ".seg";
  private static final Duration GROUP_COMMIT_INTERVAL = Duration.ZERO;

  private final Path walDir;
  private final Duration groupCommitInterval;

  /** The open segment channel, or {@code null} when no segment is currently open. */
  private FileChannel current;

  /** First sequence number of {@link #current}. Meaningful only while {@code current != null}. */
  private long currentSegStart;

  private long nextSeq;
  private long indexedThrough = -1L;

  /**
   * Opens (or creates) a WAL in {@code walDir}. Existing segment files are discovered and their
   * sequence numbers are used to initialise {@link #nextSeq()}. No segment file is opened or
   * created here — the first {@link #append} does that.
   */
  public SegmentedWriteAheadLog(Path walDir) throws IOException {
    this(walDir, GROUP_COMMIT_INTERVAL);
  }

  /**
   * Opens (or creates) a WAL in {@code walDir} with an explicit group-commit interval reported by
   * {@link #groupCommitInterval()}. Durability semantics are identical to {@link
   * #SegmentedWriteAheadLog(Path)}.
   */
  public SegmentedWriteAheadLog(Path walDir, Duration groupCommitInterval) throws IOException {
    this.walDir = walDir;
    this.groupCommitInterval = groupCommitInterval;
    Files.createDirectories(walDir);
    List<Long> starts = existingSegmentStarts();
    nextSeq = starts.isEmpty() ? 0L : lastSeqOf(starts.get(starts.size() - 1));
    currentSegStart = nextSeq;
    current = null; // lazily opened on the first append
  }

  /**
   * Appends {@code entry} to the current segment and returns the sequence number assigned to this
   * entry. Lazily opens a new segment file if none is currently open (after construction or after
   * {@link #seal()}).
   */
  public synchronized long append(byte[] entry) throws IOException {
    ensureCurrentOpen();
    long seq = nextSeq++;
    writeFully(current, ByteBuffer.wrap(buildFrame(entry)));
    current.force(true);
    return seq;
  }

  /**
   * Group-commit fast path: writes every entry in {@code entries} and forces the segment to disk
   * exactly once, instead of once per entry. On return the whole batch is durable. Crash
   * consistency is preserved per batch: a crash mid-batch leaves a prefix of intact, CRC-verified
   * frames followed by at most one torn trailing frame, which replay rejects (EOF / CRC mismatch).
   *
   * @param entries payloads to append, in order
   * @return the sequence numbers assigned to {@code entries}, in the same order
   * @throws IOException on storage failure
   */
  public synchronized long[] appendBatch(List<byte[]> entries) throws IOException {
    int n = entries.size();
    long[] seqs = new long[n];
    if (n == 0) {
      return seqs;
    }
    ensureCurrentOpen();
    int total = 0;
    for (byte[] e : entries) {
      total += Integer.BYTES + e.length + Integer.BYTES;
    }
    ByteBuffer batch = ByteBuffer.allocate(total).order(ByteOrder.BIG_ENDIAN);
    CRC32 crc = new CRC32();
    for (int i = 0; i < n; i++) {
      byte[] e = entries.get(i);
      seqs[i] = nextSeq++;
      crc.reset();
      crc.update(e);
      batch.putInt(e.length);
      batch.put(e);
      batch.putInt((int) crc.getValue());
    }
    batch.flip();
    writeFully(current, batch);
    current.force(true);
    return seqs;
  }

  private void ensureCurrentOpen() throws IOException {
    if (current == null) {
      currentSegStart = nextSeq;
      current = openSegment(currentSegStart);
    }
  }

  /**
   * Closes the current segment file. The next {@link #append} lazily starts a new segment whose
   * first sequence number is {@link #nextSeq()} at that time. Calling {@code seal()} with no open
   * segment — twice in a row, or before any append — is a no-op, so no empty segment is created.
   */
  public synchronized void seal() throws IOException {
    if (current != null) {
      current.close();
      current = null;
    }
  }

  /**
   * Replays all entries from all sealed segments (in order). The current open segment, if any, is
   * not replayed — call {@link #seal()} first if its entries are needed.
   *
   * <p>Synchronized so that the reads of {@link #current}/{@link #currentSegStart} are consistent
   * with the writes in {@link #append}/{@link #seal}. Replay is a recovery-time operation that is
   * not expected to run concurrently with appends; holding the lock for its duration is harmless.
   */
  public synchronized void replay(Consumer<byte[]> handler) throws IOException {
    for (long start : existingSegmentStarts()) {
      if (current != null && start == currentSegStart) continue; // skip the open segment
      replaySegment(segmentPath(start), handler);
    }
  }

  @Override
  public synchronized Stream<WalEntry> readFrom(long fromSeqInclusive) throws IOException {
    List<WalEntry> entries = new ArrayList<>();
    for (long start : existingSegmentStarts()) {
      readSegment(segmentPath(start), start, fromSeqInclusive, entries);
    }
    return entries.stream();
  }

  @Override
  public synchronized long lastSequenceNumber() {
    return nextSeq - 1;
  }

  @Override
  public synchronized long[] unindexedTailSeqs() {
    long lastSeq = lastSequenceNumber();
    if (indexedThrough >= lastSeq) {
      return new long[0];
    }
    return new long[] {indexedThrough + 1, lastSeq};
  }

  @Override
  public synchronized void markIndexed(long seqStartInclusive, long seqEndInclusive) {
    if (seqStartInclusive <= indexedThrough + 1 && seqEndInclusive > indexedThrough) {
      indexedThrough = seqEndInclusive;
    }
  }

  @Override
  public synchronized void flush() throws IOException {
    if (current != null) {
      current.force(true);
    }
  }

  @Override
  public Duration groupCommitInterval() {
    return groupCommitInterval;
  }

  /** The sequence number that will be assigned to the next {@link #append} call. */
  public synchronized long nextSeq() {
    return nextSeq;
  }

  @Override
  public synchronized void close() throws IOException {
    if (current != null) {
      current.close();
      current = null;
    }
  }

  // --- internals ---

  private FileChannel openSegment(long seqStart) throws IOException {
    return FileChannel.open(
        segmentPath(seqStart),
        StandardOpenOption.CREATE,
        StandardOpenOption.WRITE,
        StandardOpenOption.APPEND);
  }

  private Path segmentPath(long seqStart) {
    return walDir.resolve(String.format("%s%020d%s", SEGMENT_PREFIX, seqStart, SEGMENT_SUFFIX));
  }

  private List<Long> existingSegmentStarts() throws IOException {
    List<Long> starts = new ArrayList<>();
    try (var stream = Files.list(walDir)) {
      stream
          .filter(
              p ->
                  p.getFileName().toString().startsWith(SEGMENT_PREFIX)
                      && p.getFileName().toString().endsWith(SEGMENT_SUFFIX))
          .map(p -> parseSeqStart(p.getFileName().toString()))
          .sorted()
          .forEach(starts::add);
    }
    return starts;
  }

  private static long parseSeqStart(String filename) {
    String digits =
        filename.substring(SEGMENT_PREFIX.length(), filename.length() - SEGMENT_SUFFIX.length());
    return Long.parseLong(digits);
  }

  private long lastSeqOf(long segStart) throws IOException {
    long seq = segStart;
    try (InputStream in = Files.newInputStream(segmentPath(segStart))) {
      byte[] lenBuf = new byte[4];
      while (true) {
        int r = in.readNBytes(lenBuf, 0, 4);
        if (r == 0) break;
        if (r < 4) throw new EOFException("truncated WAL entry length at seq " + seq);
        int len = ByteBuffer.wrap(lenBuf).order(ByteOrder.BIG_ENDIAN).getInt();
        in.skipNBytes(len + 4); // payload + CRC
        seq++;
      }
    }
    return seq;
  }

  private void replaySegment(Path segPath, Consumer<byte[]> handler) throws IOException {
    try (InputStream in = Files.newInputStream(segPath)) {
      byte[] lenBuf = new byte[4];
      while (true) {
        byte[] payload = readPayload(in, lenBuf, segPath);
        if (payload == null) return;
        handler.accept(payload);
      }
    }
  }

  private void readSegment(
      Path segPath, long firstSeq, long fromSeqInclusive, List<WalEntry> entries)
      throws IOException {
    try (InputStream in = Files.newInputStream(segPath)) {
      long seq = firstSeq;
      byte[] lenBuf = new byte[4];
      while (true) {
        byte[] payload = readPayload(in, lenBuf, segPath);
        if (payload == null) return;
        if (seq >= fromSeqInclusive) {
          entries.add(new WalEntry(seq, payload));
        }
        seq++;
      }
    }
  }

  private static byte[] readPayload(InputStream in, byte[] lenBuf, Path segPath)
      throws IOException {
    int r = in.readNBytes(lenBuf, 0, 4);
    if (r == 0) return null;
    if (r < 4) throw new EOFException("truncated WAL entry in " + segPath);
    int len = ByteBuffer.wrap(lenBuf).order(ByteOrder.BIG_ENDIAN).getInt();
    if (len < 0) {
      throw new IOException("negative WAL entry length in " + segPath);
    }
    byte[] payload = in.readNBytes(len);
    if (payload.length < len) {
      throw new EOFException("truncated WAL payload in " + segPath);
    }
    byte[] crcBuf = in.readNBytes(4);
    if (crcBuf.length < 4) {
      throw new EOFException("truncated WAL checksum in " + segPath);
    }
    int storedCrc = ByteBuffer.wrap(crcBuf).order(ByteOrder.BIG_ENDIAN).getInt();
    CRC32 crc = new CRC32();
    crc.update(payload);
    if ((int) crc.getValue() != storedCrc) {
      throw new IOException("CRC mismatch in WAL segment " + segPath);
    }
    return payload;
  }

  private static byte[] buildFrame(byte[] payload) {
    CRC32 crc = new CRC32();
    crc.update(payload);
    ByteBuffer buf = ByteBuffer.allocate(4 + payload.length + 4).order(ByteOrder.BIG_ENDIAN);
    buf.putInt(payload.length);
    buf.put(payload);
    buf.putInt((int) crc.getValue());
    return buf.array();
  }

  private static void writeFully(FileChannel channel, ByteBuffer buffer) throws IOException {
    while (buffer.hasRemaining()) {
      channel.write(buffer);
    }
  }
}
