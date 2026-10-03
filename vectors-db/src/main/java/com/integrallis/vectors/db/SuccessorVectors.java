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
package com.integrallis.vectors.db;

import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.db.storage.MemorySegmentVectors;
import com.integrallis.vectors.hnsw.RandomAccessVectors;
import com.integrallis.vectors.storage.memory.AlignmentUtil;
import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.FileSystemException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.Objects;
import java.util.zip.CRC32;

/**
 * The next generation's {@code vectors.bin} as something to read and stream, rather than a copy of
 * the whole collection on the heap.
 *
 * <p>A commit publishes a generation holding every live vector plus the staged ones. Building that
 * image in memory first cost two allocations proportional to the <b>entire collection</b> on every
 * commit — a {@code byte[]} of the finished file and a {@code float[][]} of every vector — so the
 * heap a commit needed grew with the collection and an ingest died of {@link OutOfMemoryError} long
 * before the disk filled. At 512 dimensions each was about 2 KiB per document, so a
 * 460,000-document collection needed roughly 1.9 GiB of fresh heap to commit, and the {@code
 * byte[]} also capped {@code vectors.bin} at 2 GiB — about one million documents at 512 dimensions
 * — regardless of how much disk or memory the host had.
 *
 * <p>This class holds the carried-over generation's mapped vectors and the staged documents, and
 * produces what each caller actually needs:
 *
 * <ul>
 *   <li>{@link #length()} and {@link #crc32()} for the manifest, the latter streamed in chunks.
 *   <li>{@link #writeTo(Path)} to emit the file, copying the carried-over bytes straight from the
 *       predecessor's mapping.
 *   <li>{@link #asVectors()} for an index build, a read-through view with no copy.
 * </ul>
 *
 * <p>Nothing here is sized by the collection: the only buffer is {@link #CHUNK_BYTES}.
 */
final class SuccessorVectors {

  /** Streaming buffer for the CRC and write passes. Independent of the collection's size. */
  private static final int CHUNK_BYTES = 1 << 20;

  private final MemorySegmentVectors carried;
  private final int carriedCount;
  private final List<Document> staged;
  private final int dimension;
  private final int stride;
  private final long length;

  /**
   * The predecessor's {@code vectors.bin}, to hard-link rather than copy. Null disables linking.
   */
  private Path carriedFile;

  /** The predecessor's recorded CRC, to combine rather than recompute. Null forces a full pass. */
  private Long carriedCrc;

  SuccessorVectors(
      MemorySegmentVectors carried, int carriedCount, List<Document> staged, int dimension) {
    this.carried = carried;
    this.carriedCount = carriedCount;
    this.staged = Objects.requireNonNull(staged, "staged must not be null");
    this.dimension = dimension;
    long strideL =
        AlignmentUtil.alignUp((long) dimension * Float.BYTES, AlignmentUtil.VECTOR_ALIGNMENT);
    if (strideL > Integer.MAX_VALUE) {
      throw new IllegalStateException("vector stride exceeds 2 GiB: " + strideL);
    }
    this.stride = (int) strideL;
    this.length = strideL * (long) size();
    if (carriedCount > 0 && carried == null) {
      throw new IllegalStateException(
          "cannot carry " + carriedCount + " vectors forward without the predecessor's mapping");
    }
  }

  /**
   * Lets the commit hand over what it knows about the predecessor's payload so this generation can
   * be produced in time proportional to the staged documents instead of the collection.
   *
   * <p>{@code file} is hard-linked instead of copied, so the appended bytes are the only ones
   * written. {@code crc} is combined with the appended bytes' checksum instead of re-reading the
   * file. Both are optional: without them the file is streamed and checksummed in full, which is
   * what a first commit, a compaction, or an unlinkable filesystem does.
   */
  SuccessorVectors carriedFrom(Path file, long crc) {
    this.carriedFile = file;
    this.carriedCrc = crc;
    return this;
  }

  int size() {
    return carriedCount + staged.size();
  }

  int dimension() {
    return dimension;
  }

  /** Length of {@code vectors.bin}, which is no longer bounded by the heap or by 2 GiB. */
  long length() {
    return length;
  }

  /**
   * CRC of the bytes {@link #writeTo} will produce.
   *
   * <p>When the predecessor's CRC is known this is {@code combine(predecessor, staged)} — work
   * proportional to the staged documents. Otherwise every byte is read, which is correct but costs
   * the collection.
   */
  long crc32() {
    if (carriedCrc != null) {
      CRC32 tail = new CRC32();
      byte[] one = new byte[stride];
      for (Document document : staged) {
        encodeStaged(document, one);
        tail.update(one, 0, stride);
      }
      long stagedBytes = (long) stride * staged.size();
      return com.integrallis.vectors.db.storage.Crc32Combine.combine(
          carriedCrc, tail.getValue(), stagedBytes);
    }
    CRC32 crc = new CRC32();
    byte[] chunk = new byte[CHUNK_BYTES];
    long carriedBytes = (long) stride * carriedCount;
    long offset = 0;
    while (offset < carriedBytes) {
      int n = (int) Math.min(CHUNK_BYTES, carriedBytes - offset);
      MemorySegment.copy(carried.segment(), ValueLayout.JAVA_BYTE, offset, chunk, 0, n);
      crc.update(chunk, 0, n);
      offset += n;
    }
    byte[] one = new byte[stride];
    for (Document document : staged) {
      encodeStaged(document, one);
      crc.update(one, 0, stride);
    }
    return crc.getValue();
  }

  /**
   * Produces the file.
   *
   * <p>With a predecessor file available this hard-links it and appends the staged vectors, so the
   * bytes written are the batch and not the collection. {@code vectors.bin} is a bare array of
   * stride-aligned records with no header, which is what makes that sound: the successor's bytes
   * are the predecessor's bytes followed by the new ones, unchanged.
   *
   * <p>The link means the appended bytes land in the predecessor's file too, before this generation
   * is published. That is safe because a generation's extent is its manifest's length, not the
   * file's: a reader of the predecessor maps its own length, recovery checksums its own prefix, and
   * a commit that dies here leaves a tail that the next commit overwrites. The file is truncated to
   * this generation's exact length first, which is what discards such a tail.
   */
  void writeTo(Path destination) throws IOException {
    if (carriedFile != null && carriedCount > 0) {
      try {
        Files.createLink(destination, carriedFile);
        try (FileChannel channel = FileChannel.open(destination, StandardOpenOption.WRITE)) {
          long carriedBytes = (long) stride * carriedCount;
          channel.truncate(carriedBytes); // drop any tail left by a commit that died
          channel.position(carriedBytes);
          ByteBuffer out = ByteBuffer.allocate(CHUNK_BYTES).order(ByteOrder.LITTLE_ENDIAN);
          byte[] one = new byte[stride];
          for (Document document : staged) {
            if (out.remaining() < stride) {
              out.flip();
              writeFully(channel, out);
              out.clear();
            }
            encodeStaged(document, one);
            out.put(one);
          }
          out.flip();
          if (out.hasRemaining()) {
            writeFully(channel, out);
          }
          channel.force(true);
        }
        return;
      } catch (UnsupportedOperationException | FileSystemException e) {
        // Hard links are not available here (a filesystem that cannot, or a cross-device path).
        // Fall through and write the whole file.
        Files.deleteIfExists(destination);
      }
    }
    try (FileChannel channel =
        FileChannel.open(
            destination,
            StandardOpenOption.CREATE,
            StandardOpenOption.WRITE,
            StandardOpenOption.TRUNCATE_EXISTING)) {

      long carriedBytes = (long) stride * carriedCount;
      if (carriedBytes > 0) {
        long offset = 0;
        while (offset < carriedBytes) {
          long n = Math.min(CHUNK_BYTES, carriedBytes - offset);
          MemorySegment slice = carried.segment().asSlice(offset, n);
          writeFully(channel, slice.asByteBuffer());
          offset += n;
        }
      }

      ByteBuffer out = ByteBuffer.allocate(CHUNK_BYTES).order(ByteOrder.LITTLE_ENDIAN);
      byte[] one = new byte[stride];
      for (Document document : staged) {
        if (out.remaining() < stride) {
          out.flip();
          writeFully(channel, out);
          out.clear();
        }
        encodeStaged(document, one);
        out.put(one);
      }
      out.flip();
      if (out.hasRemaining()) {
        writeFully(channel, out);
      }
      channel.force(true);
    }
  }

  /**
   * A read-through view over the carried-over mapping and the staged vectors, for an index build.
   *
   * <p>Carried rows are copied into a per-thread scratch array. Staged rows are read directly. The
   * view declares shared return buffers so builders preserve the insertion query before reading
   * another row. Segment scoring is not exposed: mixing heap-backed staged rows with mapped rows
   * regressed the measured append workload.
   */
  RandomAccessVectors asVectors() {
    return new View();
  }

  private void encodeStaged(Document document, byte[] into) {
    java.util.Arrays.fill(into, (byte) 0);
    ByteBuffer buffer = ByteBuffer.wrap(into).order(ByteOrder.LITTLE_ENDIAN);
    float[] vector = document.vector();
    for (int i = 0; i < dimension; i++) {
      buffer.putFloat(vector[i]);
    }
  }

  private static void writeFully(FileChannel channel, ByteBuffer buffer) throws IOException {
    while (buffer.hasRemaining()) {
      channel.write(buffer);
    }
  }

  private final class View implements RandomAccessVectors {

    private final ThreadLocal<float[]> scratch =
        ThreadLocal.withInitial(() -> new float[dimension]);

    @Override
    public int size() {
      return SuccessorVectors.this.size();
    }

    @Override
    public int dimension() {
      return dimension;
    }

    @Override
    public float[] getVector(int ordinal) {
      if (ordinal >= carriedCount) {
        return staged.get(ordinal - carriedCount).vector();
      }
      float[] out = scratch.get();
      MemorySegment.copy(
          carried.vectorSlice(ordinal), ValueLayout.JAVA_FLOAT, 0L, out, 0, dimension);
      return out;
    }

    @Override
    public boolean sharesReturnBuffer() {
      return true;
    }
  }
}
