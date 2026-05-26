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

import com.integrallis.vectors.storage.backend.StorageBackend;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.time.Duration;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;
import java.util.stream.Stream;
import java.util.zip.CRC32;

/**
 * {@link WriteAheadLog} implementation backed by any {@link StorageBackend}, with 1-second group
 * commit and 512 MB segment caps per §6.2 / §16.2 of {@code vectors-distributed-design.md}.
 *
 * <p>Per-namespace prefix layout: {@code {root}/{namespace}/wal/NNNNNNNNNNNN.log} for segment
 * objects and {@code {root}/{namespace}/manifest.json} for the segment-state manifest.
 *
 * <p>Each segment object contains a sequence of length-prefixed, CRC-checked frames:
 *
 * <pre>
 *   [4B big-endian payload length] [payload bytes] [4B big-endian CRC32 of payload]
 * </pre>
 *
 * <p>Concurrent {@code append} calls within a {@link #groupCommitInterval} window are coalesced
 * into a single {@link StorageBackend#put} call. Callers block until that PUT completes; the
 * returned sequence number is therefore guaranteed durable.
 */
public final class BackendWriteAheadLog implements WriteAheadLog {

  public static final int DEFAULT_MAX_SEGMENT_BYTES = 512 * 1024 * 1024;
  public static final Duration DEFAULT_GROUP_COMMIT_INTERVAL = Duration.ofSeconds(1);

  private final StorageBackend backend;
  private final String namespace;
  private final Duration groupCommitInterval;
  private final int maxSegmentBytes;

  private final ReentrantLock lock = new ReentrantLock();
  private final Condition durable = lock.newCondition();
  private final Deque<Pending> pending = new ArrayDeque<>();
  private final ScheduledExecutorService scheduler;
  private ScheduledFuture<?> scheduledFlush;
  private boolean closed;
  private IOException commitFailure;

  private final AtomicLong putCount = new AtomicLong();

  private long nextSeq;
  private long lastDurableSeq = -1L;
  private int compactedSegmentIndex;
  private long compactedThroughSeq = -1L;
  private int activeSegmentIndex;
  private long activeFirstSeq;
  private byte[] activeBytes = new byte[0];
  private final List<SegmentMeta> closedSegments = new ArrayList<>();

  /** Convenience constructor with default 1 s group-commit interval and 512 MB segment cap. */
  public BackendWriteAheadLog(StorageBackend backend, String namespace) throws IOException {
    this(backend, namespace, DEFAULT_GROUP_COMMIT_INTERVAL, DEFAULT_MAX_SEGMENT_BYTES);
  }

  public BackendWriteAheadLog(
      StorageBackend backend, String namespace, Duration groupCommitInterval, int maxSegmentBytes)
      throws IOException {
    if (groupCommitInterval.isNegative() || groupCommitInterval.isZero()) {
      throw new IllegalArgumentException("groupCommitInterval must be positive");
    }
    if (maxSegmentBytes <= 0) {
      throw new IllegalArgumentException("maxSegmentBytes must be > 0");
    }
    this.backend = backend;
    this.namespace = namespace;
    this.groupCommitInterval = groupCommitInterval;
    this.maxSegmentBytes = maxSegmentBytes;
    this.scheduler =
        Executors.newSingleThreadScheduledExecutor(
            r -> {
              Thread t = new Thread(r, "wal-commit-" + namespace);
              t.setDaemon(true);
              return t;
            });
    loadManifest();
  }

  // ─── public API ────────────────────────────────────────────────────────────

  @Override
  public long append(byte[] entry) throws IOException {
    long seq;
    lock.lock();
    try {
      if (closed) throw new IOException("WAL closed");
      throwIfCommitFailed();
      seq = nextSeq++;
      pending.addLast(new Pending(seq, entry.clone()));
      if (scheduledFlush == null) {
        scheduledFlush =
            scheduler.schedule(
                this::scheduledFlush, groupCommitInterval.toNanos(), TimeUnit.NANOSECONDS);
      }
      while (lastDurableSeq < seq) {
        durable.awaitUninterruptibly();
        throwIfCommitFailed();
        if (closed && lastDurableSeq < seq) {
          throw new IOException("WAL closed before append durable; seq=" + seq);
        }
      }
    } finally {
      lock.unlock();
    }
    return seq;
  }

  @Override
  public long lastSequenceNumber() {
    lock.lock();
    try {
      return lastDurableSeq;
    } finally {
      lock.unlock();
    }
  }

  @Override
  public long[] unindexedTailSeqs() {
    lock.lock();
    try {
      List<long[]> ranges = new ArrayList<>();
      for (SegmentMeta s : closedSegments) {
        if (!s.indexed) ranges.add(new long[] {s.firstSeq, s.lastSeq});
      }
      if (lastDurableSeq >= activeFirstSeq) {
        ranges.add(new long[] {activeFirstSeq, lastDurableSeq});
      }
      long[] out = new long[ranges.size() * 2];
      int i = 0;
      for (long[] r : ranges) {
        out[i++] = r[0];
        out[i++] = r[1];
      }
      return out;
    } finally {
      lock.unlock();
    }
  }

  @Override
  public void markIndexed(long seqStartInclusive, long seqEndInclusive) throws IOException {
    lock.lock();
    try {
      for (SegmentMeta s : closedSegments) {
        if (s.firstSeq >= seqStartInclusive && s.lastSeq <= seqEndInclusive) {
          s.indexed = true;
        }
      }
      compactIndexedPrefix();
      writeManifest();
    } finally {
      lock.unlock();
    }
  }

  @Override
  public void flush() throws IOException {
    lock.lock();
    try {
      throwIfCommitFailed();
      if (pending.isEmpty()) return;
      if (scheduledFlush != null) {
        scheduledFlush.cancel(false);
        scheduledFlush = null;
      }
      try {
        doCommit();
      } catch (IOException e) {
        recordCommitFailure(e);
        throw e;
      }
    } finally {
      lock.unlock();
    }
  }

  @Override
  public Duration groupCommitInterval() {
    return groupCommitInterval;
  }

  @Override
  public void close() throws IOException {
    boolean shutdownScheduler = false;
    IOException closeFailure = null;
    lock.lock();
    try {
      if (closed) return;
      shutdownScheduler = true;
      try {
        if (!pending.isEmpty() && commitFailure == null) {
          if (scheduledFlush != null) {
            scheduledFlush.cancel(false);
            scheduledFlush = null;
          }
          doCommit();
        }
        // Seal the active segment if it has any entries so on reopen the next segment
        // index advances and the previously active object is treated as a closed (◈) segment.
        if (lastDurableSeq >= activeFirstSeq && commitFailure == null) {
          sealActiveSegment();
          writeManifest();
        }
      } catch (IOException e) {
        recordCommitFailure(e);
        closeFailure = e;
      } finally {
        closed = true;
        durable.signalAll();
      }
    } finally {
      lock.unlock();
      if (shutdownScheduler) {
        scheduler.shutdownNow();
      }
    }
    if (closeFailure != null) {
      throw closeFailure;
    }
  }

  /** Total number of {@link StorageBackend#put} calls issued for segment data; for tests only. */
  long putCount() {
    return putCount.get();
  }

  // ─── commit path ───────────────────────────────────────────────────────────

  private void scheduledFlush() {
    lock.lock();
    try {
      scheduledFlush = null;
      if (pending.isEmpty() || closed) return;
      doCommit();
    } catch (IOException e) {
      recordCommitFailure(e);
      // The scheduler swallows checked exceptions; surface as an unchecked one so it
      // shows up in test logs. Awaiting appenders are released before this is thrown.
      throw new UncheckedIOException(e);
    } finally {
      lock.unlock();
    }
  }

  private void throwIfCommitFailed() throws IOException {
    if (commitFailure != null) {
      throw new IOException("WAL commit failed", commitFailure);
    }
  }

  private void recordCommitFailure(IOException failure) {
    if (commitFailure == null) {
      commitFailure = failure;
    }
    durable.signalAll();
  }

  /**
   * Drains {@link #pending} and writes one or more segment objects covering all entries.
   *
   * <p>Frames are processed one at a time so that the seq of the last frame written into the active
   * segment is always known precisely; this is required to seal segments mid-commit with correct
   * {@code [firstSeq..lastSeq]} ranges.
   */
  private void doCommit() throws IOException {
    assert lock.isHeldByCurrentThread();
    // Last seq covered by the bytes currently in `activeBytes`. -1 means the active segment is
    // empty (no frames yet). Initialised from the durable state on entry.
    long lastSeqInActive = (lastDurableSeq >= activeFirstSeq) ? lastDurableSeq : -1L;
    boolean activeDirty = false;
    long lastFlushed = lastDurableSeq;
    ByteArrayOutputStream activeBuffer = new ByteArrayOutputStream(activeBytes.length);
    activeBuffer.writeBytes(activeBytes);
    int activeLength = activeBytes.length;

    for (Pending p : pending) {
      byte[] frame = encodeFrame(p.payload);
      // Seal before adding when the new frame would push the active segment past its cap.
      if (activeLength > 0 && activeLength + frame.length > maxSegmentBytes) {
        if (activeDirty) {
          activeBytes = activeBuffer.toByteArray();
          backend.put(segmentKey(activeSegmentIndex), activeBytes);
          putCount.incrementAndGet();
          activeDirty = false;
        }
        closedSegments.add(
            new SegmentMeta(activeSegmentIndex, activeFirstSeq, lastSeqInActive, false));
        activeSegmentIndex++;
        activeBytes = new byte[0];
        activeBuffer = new ByteArrayOutputStream(frame.length);
        activeLength = 0;
        lastSeqInActive = -1L;
      }
      if (activeLength == 0) {
        activeFirstSeq = p.seq;
      }
      activeBuffer.writeBytes(frame);
      activeLength += frame.length;
      lastSeqInActive = p.seq;
      lastFlushed = p.seq;
      activeDirty = true;
    }
    pending.clear();

    if (activeDirty) {
      activeBytes = activeBuffer.toByteArray();
      backend.put(segmentKey(activeSegmentIndex), activeBytes);
      putCount.incrementAndGet();
    }

    lastDurableSeq = lastFlushed;
    writeManifest();
    durable.signalAll();
  }

  /** Closes the active segment (if non-empty) and advances to the next segment index. */
  private void sealActiveSegment() {
    if (activeBytes.length == 0) return;
    closedSegments.add(new SegmentMeta(activeSegmentIndex, activeFirstSeq, lastDurableSeq, false));
    activeSegmentIndex++;
    activeFirstSeq = lastDurableSeq + 1;
    activeBytes = new byte[0];
  }

  // ─── readFrom ──────────────────────────────────────────────────────────────

  @Override
  public Stream<WalEntry> readFrom(long fromSeqInclusive) throws IOException {
    List<SegmentMeta> closedSnapshot;
    int activeIdxSnapshot;
    long activeFirstSnapshot;
    long lastDurableSnapshot;
    long compactedThroughSnapshot;
    lock.lock();
    try {
      closedSnapshot = new ArrayList<>(closedSegments);
      activeIdxSnapshot = activeSegmentIndex;
      activeFirstSnapshot = activeFirstSeq;
      lastDurableSnapshot = lastDurableSeq;
      compactedThroughSnapshot = compactedThroughSeq;
    } finally {
      lock.unlock();
    }
    if (fromSeqInclusive <= compactedThroughSnapshot) {
      throw new IOException("WAL entries compacted through seq " + compactedThroughSnapshot);
    }
    List<SegmentReadPlan> plans = new ArrayList<>();
    for (SegmentMeta s : closedSnapshot) {
      if (s.lastSeq < fromSeqInclusive) continue;
      plans.add(new SegmentReadPlan(segmentKey(s.index), s.firstSeq, fromSeqInclusive, s.lastSeq));
    }
    if (lastDurableSnapshot >= activeFirstSnapshot && lastDurableSnapshot >= fromSeqInclusive) {
      plans.add(
          new SegmentReadPlan(
              segmentKey(activeIdxSnapshot),
              activeFirstSnapshot,
              fromSeqInclusive,
              lastDurableSnapshot));
    }
    return java.util.stream.StreamSupport.stream(
        java.util.Spliterators.spliteratorUnknownSize(
            new WalEntryIterator(plans),
            java.util.Spliterator.ORDERED | java.util.Spliterator.NONNULL),
        false);
  }

  private List<WalEntry> readSegment(SegmentReadPlan plan) throws IOException {
    byte[] data = backend.get(plan.key());
    if (data == null) throw new IOException("WAL segment missing: " + plan.key());
    List<WalEntry> entries = new ArrayList<>();
    long seq = plan.firstSeq();
    int off = 0;
    while (off < data.length && seq <= plan.lastSeqInSegment()) {
      if (off + 4 > data.length) throw new IOException("truncated WAL segment: " + plan.key());
      int len = ByteBuffer.wrap(data, off, 4).order(ByteOrder.BIG_ENDIAN).getInt();
      off += 4;
      if (len < 0 || off + len + 4 > data.length) {
        throw new IOException("corrupt WAL frame in " + plan.key() + " at seq " + seq);
      }
      byte[] payload = new byte[len];
      System.arraycopy(data, off, payload, 0, len);
      off += len;
      int storedCrc = ByteBuffer.wrap(data, off, 4).order(ByteOrder.BIG_ENDIAN).getInt();
      off += 4;
      CRC32 crc = new CRC32();
      crc.update(payload);
      if ((int) crc.getValue() != storedCrc) {
        throw new IOException("CRC mismatch in WAL segment " + plan.key() + " at seq " + seq);
      }
      if (seq >= plan.fromSeqInclusive()) entries.add(new WalEntry(seq, payload));
      seq++;
    }
    return entries;
  }

  private static long lastSeqInSegment(String key, long firstSeq, byte[] data) throws IOException {
    long seq = firstSeq;
    int off = 0;
    while (off < data.length) {
      if (off + 4 > data.length) throw new IOException("truncated WAL segment: " + key);
      int len = ByteBuffer.wrap(data, off, 4).order(ByteOrder.BIG_ENDIAN).getInt();
      off += 4;
      if (len < 0 || off + len + 4 > data.length) {
        throw new IOException("corrupt WAL frame in " + key + " at seq " + seq);
      }
      byte[] payload = new byte[len];
      System.arraycopy(data, off, payload, 0, len);
      off += len;
      int storedCrc = ByteBuffer.wrap(data, off, 4).order(ByteOrder.BIG_ENDIAN).getInt();
      off += 4;
      CRC32 crc = new CRC32();
      crc.update(payload);
      if ((int) crc.getValue() != storedCrc) {
        throw new IOException("CRC mismatch in WAL segment " + key + " at seq " + seq);
      }
      seq++;
    }
    return seq - 1;
  }

  private final class WalEntryIterator implements Iterator<WalEntry> {
    private final Iterator<SegmentReadPlan> plans;
    private Iterator<WalEntry> current = List.<WalEntry>of().iterator();

    WalEntryIterator(List<SegmentReadPlan> plans) {
      this.plans = plans.iterator();
    }

    @Override
    public boolean hasNext() {
      while (!current.hasNext()) {
        if (!plans.hasNext()) {
          return false;
        }
        try {
          current = readSegment(plans.next()).iterator();
        } catch (IOException e) {
          throw new UncheckedIOException(e);
        }
      }
      return true;
    }

    @Override
    public WalEntry next() {
      if (!hasNext()) {
        throw new NoSuchElementException();
      }
      return current.next();
    }
  }

  // ─── framing ───────────────────────────────────────────────────────────────

  /** Encodes one WAL frame: {@code [4B BE length][payload][4B BE CRC32(payload)]}. */
  private static byte[] encodeFrame(byte[] payload) {
    CRC32 crc = new CRC32();
    crc.update(payload);
    byte[] frame = new byte[4 + payload.length + 4];
    ByteBuffer.wrap(frame, 0, 4).order(ByteOrder.BIG_ENDIAN).putInt(payload.length);
    System.arraycopy(payload, 0, frame, 4, payload.length);
    ByteBuffer.wrap(frame, 4 + payload.length, 4)
        .order(ByteOrder.BIG_ENDIAN)
        .putInt((int) crc.getValue());
    return frame;
  }

  // ─── manifest I/O ──────────────────────────────────────────────────────────

  private String manifestKey() {
    return namespace + "/manifest.json";
  }

  private String segmentKey(int index) {
    return String.format("%s/wal/%012d.log", namespace, index);
  }

  private void loadManifest() throws IOException {
    byte[] raw = backend.get(manifestKey());
    Manifest m =
        raw == null
            ? new Manifest(-1L, 0, -1L, List.of())
            : Manifest.parse(new String(raw, java.nio.charset.StandardCharsets.UTF_8));
    compactedSegmentIndex = m.compactedSegmentIndex;
    compactedThroughSeq = m.compactedThroughSeq;
    closedSegments.addAll(m.segments);
    int maxIdx = compactedSegmentIndex;
    long maxLastSeq = compactedThroughSeq;
    for (SegmentMeta s : m.segments) {
      if (s.index > maxIdx) maxIdx = s.index;
      if (s.lastSeq > maxLastSeq) maxLastSeq = s.lastSeq;
    }
    List<RecoveredSegment> recovered = new ArrayList<>();
    int candidateIndex = maxIdx + 1;
    long candidateFirstSeq = maxLastSeq + 1;
    while (true) {
      String key = segmentKey(candidateIndex);
      byte[] data = backend.get(key);
      if (data == null) break;
      long candidateLastSeq = lastSeqInSegment(key, candidateFirstSeq, data);
      recovered.add(
          new RecoveredSegment(candidateIndex, candidateFirstSeq, candidateLastSeq, data));
      candidateIndex++;
      candidateFirstSeq = candidateLastSeq + 1;
    }

    if (recovered.isEmpty()) {
      if (m.lastSeq > maxLastSeq) {
        throw new IOException(
            "WAL manifest references missing active segment: " + segmentKey(maxIdx + 1));
      }
      activeSegmentIndex = maxIdx + 1;
      nextSeq = maxLastSeq + 1;
      lastDurableSeq = maxLastSeq;
      activeFirstSeq = nextSeq;
      return;
    }

    for (int i = 0; i < recovered.size() - 1; i++) {
      RecoveredSegment s = recovered.get(i);
      closedSegments.add(new SegmentMeta(s.index(), s.firstSeq(), s.lastSeq(), false));
    }
    RecoveredSegment active = recovered.getLast();
    activeSegmentIndex = active.index();
    activeFirstSeq = active.firstSeq();
    activeBytes = active.bytes();
    lastDurableSeq = active.lastSeq();
    if (m.lastSeq > lastDurableSeq) {
      throw new IOException("WAL manifest lastSeq exceeds recovered segments: " + m.lastSeq);
    }
    nextSeq = lastDurableSeq + 1;
  }

  private void writeManifest() throws IOException {
    Manifest m =
        new Manifest(lastDurableSeq, compactedSegmentIndex, compactedThroughSeq, closedSegments);
    backend.put(manifestKey(), m.toJson().getBytes(java.nio.charset.StandardCharsets.UTF_8));
  }

  private void compactIndexedPrefix() {
    Iterator<SegmentMeta> it = closedSegments.iterator();
    while (it.hasNext()) {
      SegmentMeta s = it.next();
      if (!s.indexed) {
        return;
      }
      compactedSegmentIndex = s.index;
      compactedThroughSeq = s.lastSeq;
      it.remove();
    }
  }

  // ─── records ───────────────────────────────────────────────────────────────

  private record Pending(long seq, byte[] payload) {}

  private record SegmentReadPlan(
      String key, long firstSeq, long fromSeqInclusive, long lastSeqInSegment) {}

  private record RecoveredSegment(int index, long firstSeq, long lastSeq, byte[] bytes) {}

  static final class SegmentMeta {
    final int index;
    final long firstSeq;
    final long lastSeq;
    boolean indexed;

    SegmentMeta(int index, long firstSeq, long lastSeq, boolean indexed) {
      this.index = index;
      this.firstSeq = firstSeq;
      this.lastSeq = lastSeq;
      this.indexed = indexed;
    }
  }
}
