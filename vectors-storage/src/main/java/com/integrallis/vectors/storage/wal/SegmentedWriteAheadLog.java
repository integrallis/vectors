package com.integrallis.vectors.storage.wal;

import java.io.Closeable;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;
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
 * number of the first entry in that segment. A new segment is created by {@link #seal()} or
 * automatically on open when no segment exists yet.
 *
 * <p>This class is thread-compatible: external synchronisation is required for concurrent appends.
 */
public final class SegmentedWriteAheadLog implements Closeable {

  private static final String SEGMENT_PREFIX = "wal-";
  private static final String SEGMENT_SUFFIX = ".seg";

  private final Path walDir;
  private OutputStream current;
  private long currentSegStart;
  private long nextSeq;

  /**
   * Opens (or creates) a WAL in {@code walDir}. Existing segment files are discovered and their
   * sequence numbers are used to initialise {@link #nextSeq()}.
   */
  public SegmentedWriteAheadLog(Path walDir) throws IOException {
    this.walDir = walDir;
    Files.createDirectories(walDir);
    List<Long> starts = existingSegmentStarts();
    nextSeq = starts.isEmpty() ? 0L : lastSeqOf(starts.get(starts.size() - 1));
    currentSegStart = nextSeq;
    current = openSegment(currentSegStart);
  }

  /**
   * Appends {@code entry} to the current segment and returns the sequence number assigned to this
   * entry.
   */
  public synchronized long append(byte[] entry) throws IOException {
    long seq = nextSeq++;
    byte[] frame = buildFrame(entry);
    current.write(frame);
    current.flush();
    return seq;
  }

  /**
   * Closes the current segment file and starts a new one. The new segment's first sequence number
   * is {@link #nextSeq()}.
   */
  public synchronized void seal() throws IOException {
    current.close();
    currentSegStart = nextSeq;
    current = openSegment(currentSegStart);
  }

  /**
   * Replays all entries from all sealed segments (in order). The current open segment is not
   * replayed — call {@link #seal()} first if needed.
   */
  public void replay(Consumer<byte[]> handler) throws IOException {
    for (long start : existingSegmentStarts()) {
      if (start == currentSegStart) continue; // skip the open segment
      replaySegment(segmentPath(start), handler);
    }
  }

  /** The sequence number that will be assigned to the next {@link #append} call. */
  public synchronized long nextSeq() {
    return nextSeq;
  }

  @Override
  public synchronized void close() throws IOException {
    current.close();
  }

  // --- internals ---

  private OutputStream openSegment(long seqStart) throws IOException {
    return Files.newOutputStream(
        segmentPath(seqStart), StandardOpenOption.CREATE, StandardOpenOption.APPEND);
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
        int r = in.readNBytes(lenBuf, 0, 4);
        if (r == 0) return;
        if (r < 4) throw new EOFException("truncated WAL entry in " + segPath);
        int len = ByteBuffer.wrap(lenBuf).order(ByteOrder.BIG_ENDIAN).getInt();
        byte[] payload = in.readNBytes(len);
        byte[] crcBuf = in.readNBytes(4);
        int storedCrc = ByteBuffer.wrap(crcBuf).order(ByteOrder.BIG_ENDIAN).getInt();
        CRC32 crc = new CRC32();
        crc.update(payload);
        if ((int) crc.getValue() != storedCrc)
          throw new IOException("CRC mismatch in WAL segment " + segPath);
        handler.accept(payload);
      }
    }
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
}
