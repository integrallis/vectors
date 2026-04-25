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
package com.integrallis.vectors.db.storage;

import com.integrallis.vectors.db.id.IdMapper;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Read-only mmap-backed {@link IdMapper}. Each instance is materialized by {@link #open(Path,
 * Arena)}, which memory-maps {@code idmap.bin} through a caller-supplied {@link Arena} (shared
 * per-generation so every file in one generation unmaps atomically when the arena closes).
 *
 * <p>The file layout is:
 *
 * <pre>
 * Offset  Size  Field                         Notes
 * ------  ----  ----------------------------  --------------------------------
 *   0      4    magic                         {@link FileFormat#MAGIC_IDMAP} ("VIDP")
 *   4      4    version                       {@link FileFormat#VERSION_IDMAP}
 *   8      4    count                         number of (id, ordinal) pairs
 *  12      4    heap byte length              total size of the id-heap
 *  16      8*N  offset table                  int64 per ordinal: offset into heap
 * 16+8*N  -    heap                          per entry: varint length + UTF-8 bytes
 * </pre>
 *
 * <p>Writes go through the nested {@link Writer} — a mapped mapper is strictly immutable once
 * opened, and {@link #put(String)} throws {@link UnsupportedOperationException}. The commit path in
 * {@code VectorCollectionImpl} materializes a successor generation by calling {@link
 * Writer#writeTo(Path, List)} then opening the result through {@link #open(Path, Arena)}.
 *
 * <p><b>Lookup data structures.</b> On {@link #open(Path, Arena)}, the reader walks the heap once
 * and populates two on-heap tables: a {@code String[]} indexed by ordinal for {@link #idOf(int)},
 * and a {@code HashMap<String, Integer>} for {@link #ordinalOf(String)}. For typical corpora
 * (&lt;10M docs) this is a few MB of heap and roughly O(n) open cost dominated by UTF-8 decoding.
 * Lazy-scan alternatives that keep the strings off-heap are possible but not worth the complexity
 * at this scale.
 *
 * <p><b>Close semantics.</b> {@link #close()} is a no-op — the enclosing {@link Arena} owns the
 * mmap lifetime and the owning {@code Generation} closes the arena exactly once when its refcount
 * drops to zero.
 */
public final class MappedIdMapper implements IdMapper {

  /**
   * Size in bytes of the fixed header (magic + version + count + heap length). Delegates to {@link
   * FileFormat#STORE_HEADER_SIZE} so that the layout constant lives in exactly one place — both
   * this class and {@link MappedMetadataStore} share the same 16-byte header shape.
   */
  public static final int HEADER_SIZE = FileFormat.STORE_HEADER_SIZE;

  private static final ValueLayout.OfInt INT_LE =
      ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
  private static final ValueLayout.OfLong LONG_LE =
      ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

  private final int count;
  private final String[] ordinalToId;
  private final Map<String, Integer> idToOrdinal;

  private MappedIdMapper(int count, String[] ordinalToId, Map<String, Integer> idToOrdinal) {
    this.count = count;
    this.ordinalToId = ordinalToId;
    this.idToOrdinal = idToOrdinal;
  }

  /**
   * Opens and validates a persistent id-map file. The caller supplies the {@link Arena} that owns
   * the mmap lifetime — typically a shared arena per generation, closed exactly once by the owning
   * {@code Generation} record.
   *
   * @throws IOException if the file is truncated, has the wrong magic/version, or contains an
   *     out-of-range offset or duplicate id
   */
  public static MappedIdMapper open(Path file, Arena arena) throws IOException {
    Objects.requireNonNull(file, "file must not be null");
    Objects.requireNonNull(arena, "arena must not be null");

    MemorySegment seg;
    long fileSize;
    try (FileChannel ch = FileChannel.open(file, StandardOpenOption.READ)) {
      fileSize = ch.size();
      if (fileSize < HEADER_SIZE) {
        throw new IOException(
            "idmap.bin truncated: expected at least " + HEADER_SIZE + " bytes, got " + fileSize);
      }
      seg = ch.map(FileChannel.MapMode.READ_ONLY, 0, fileSize, arena);
    }

    int magic = seg.get(INT_LE, 0);
    if (magic != FileFormat.MAGIC_IDMAP) {
      throw new IOException(
          String.format(
              "idmap magic mismatch: expected 0x%08x, got 0x%08x", FileFormat.MAGIC_IDMAP, magic));
    }
    int version = seg.get(INT_LE, 4);
    if (version != FileFormat.VERSION_IDMAP) {
      throw new IOException(
          "idmap version mismatch: expected " + FileFormat.VERSION_IDMAP + ", got " + version);
    }
    int count = seg.get(INT_LE, 8);
    int heapByteLength = seg.get(INT_LE, 12);
    if (count < 0) {
      throw new IOException("idmap count must be >= 0: " + count);
    }
    if (heapByteLength < 0) {
      throw new IOException("idmap heap byte length must be >= 0: " + heapByteLength);
    }

    long offsetTableStart = HEADER_SIZE;
    long heapStart = offsetTableStart + (long) count * Long.BYTES;
    long expectedSize = heapStart + heapByteLength;
    if (fileSize < expectedSize) {
      throw new IOException(
          "idmap.bin truncated: header declares " + expectedSize + " bytes, file is " + fileSize);
    }

    String[] ordinalToId = new String[count];
    Map<String, Integer> idToOrdinal = HashMap.newHashMap(count);
    for (int i = 0; i < count; i++) {
      long heapOff = seg.get(LONG_LE, offsetTableStart + (long) i * Long.BYTES);
      if (heapOff < 0 || heapOff >= heapByteLength) {
        throw new IOException("idmap heap offset out of range for ordinal " + i + ": " + heapOff);
      }
      long absOff = heapStart + heapOff;
      long packed = FileFormat.readVarIntFromSegment(seg, absOff);
      int idLen = (int) packed;
      int varintBytes = (int) (packed >>> 32);
      long idEnd = absOff + varintBytes + idLen;
      if (idLen < 0 || idEnd > heapStart + heapByteLength) {
        throw new IOException("idmap heap entry out of range for ordinal " + i + ": len=" + idLen);
      }
      byte[] idBytes = new byte[idLen];
      MemorySegment.copy(seg, ValueLayout.JAVA_BYTE, absOff + varintBytes, idBytes, 0, idLen);
      String id = new String(idBytes, StandardCharsets.UTF_8);
      ordinalToId[i] = id;
      // Allow duplicate ids (upsert: old ordinal is tombstoned, new ordinal is live).
      // The last ordinal wins in the forward mapping — this is correct because the tombstoned
      // ordinal is always earlier and the caller skips tombstoned ordinals on all read paths.
      idToOrdinal.put(id, i);
    }

    return new MappedIdMapper(count, ordinalToId, idToOrdinal);
  }

  /**
   * Unsupported — a mapped mapper is read-only. The commit pipeline materializes a successor
   * generation via {@link Writer#writeTo(Path, List)} instead of per-entry mutation.
   */
  @Override
  public int put(String id) {
    throw new UnsupportedOperationException("MappedIdMapper is read-only");
  }

  @Override
  public boolean contains(String id) {
    return idToOrdinal.containsKey(id);
  }

  @Override
  public int ordinalOf(String id) {
    Integer ord = idToOrdinal.get(id);
    return ord == null ? -1 : ord;
  }

  @Override
  public String idOf(int ordinal) {
    if (ordinal < 0 || ordinal >= count) {
      throw new IndexOutOfBoundsException("ordinal out of range: " + ordinal);
    }
    return ordinalToId[ordinal];
  }

  @Override
  public int size() {
    return count;
  }

  /**
   * No-op — the enclosing {@link Arena} owns the mmap lifetime. The owning {@code Generation}
   * closes the arena exactly once when its refcount drops to zero.
   */
  @Override
  public void close() {
    // no-op
  }

  /** Static serializer that writes the complete {@code idmap.bin} layout. */
  public static final class Writer {

    private Writer() {}

    /**
     * Builds the complete {@code idmap.bin} byte image without touching the file system. The commit
     * pipeline calls this to pre-compute the bytes so it can checksum the content before deciding
     * where to write it. {@link #writeTo(Path, List)} is a thin wrapper that calls this and streams
     * the result to disk.
     *
     * @throws IOException if the heap or total size exceeds 2 GiB or any id is null
     */
    public static byte[] toBytes(List<String> ids) throws IOException {
      Objects.requireNonNull(ids, "ids must not be null");

      int count = ids.size();
      byte[][] idBytes = new byte[count][];
      int[] varIntSizes = new int[count];
      long heapSize = 0;
      for (int i = 0; i < count; i++) {
        String id = ids.get(i);
        if (id == null) {
          throw new IOException("idmap ids[" + i + "] is null");
        }
        idBytes[i] = id.getBytes(StandardCharsets.UTF_8);
        varIntSizes[i] = FileFormat.varIntSize(idBytes[i].length);
        heapSize += (long) varIntSizes[i] + idBytes[i].length;
      }
      if (heapSize > Integer.MAX_VALUE) {
        throw new IOException(
            "idmap heap size exceeds 2 GiB: "
                + heapSize
                + " (raise VDB_MAX_HEAP or split the collection)");
      }

      long totalSize = (long) HEADER_SIZE + (long) count * Long.BYTES + heapSize;
      if (totalSize > Integer.MAX_VALUE) {
        throw new IOException("idmap total size exceeds 2 GiB: " + totalSize);
      }

      byte[] out = new byte[(int) totalSize];
      ByteBuffer buf = ByteBuffer.wrap(out).order(ByteOrder.LITTLE_ENDIAN);
      buf.putInt(FileFormat.MAGIC_IDMAP);
      buf.putInt(FileFormat.VERSION_IDMAP);
      buf.putInt(count);
      buf.putInt((int) heapSize);

      long heapOff = 0;
      for (int i = 0; i < count; i++) {
        buf.putLong(heapOff);
        heapOff += (long) varIntSizes[i] + idBytes[i].length;
      }

      for (int i = 0; i < count; i++) {
        FileFormat.writeVarInt(buf, idBytes[i].length);
        buf.put(idBytes[i]);
      }

      return out;
    }

    /**
     * Writes an id-map to {@code path}, fsyncing the file contents before returning. The file must
     * not already exist ({@link StandardOpenOption#CREATE_NEW}).
     *
     * @param path target file (typically {@code gen-NNNN/idmap.bin} inside an in-flight tmp dir)
     * @param ids id list indexed by ordinal; {@code ids.get(i)} is the id assigned to ordinal
     *     {@code i}
     * @throws IOException if the file cannot be created, the heap exceeds 2 GiB, or any id is null
     */
    public static void writeTo(Path path, List<String> ids) throws IOException {
      Objects.requireNonNull(path, "path must not be null");
      byte[] out = toBytes(ids);
      writeBytesAndFsync(path, out);
    }

    /**
     * Writes the pre-built byte image from {@link #toBytes(List)} to {@code path} and fsyncs.
     * Shared between {@link #writeTo(Path, List)} and the commit pipeline's {@code
     * BufferedGenerationSource}.
     */
    public static void writeBytesAndFsync(Path path, byte[] bytes) throws IOException {
      Objects.requireNonNull(path, "path must not be null");
      Objects.requireNonNull(bytes, "bytes must not be null");
      try (FileChannel ch =
          FileChannel.open(path, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE)) {
        ByteBuffer writeBuf = ByteBuffer.wrap(bytes);
        while (writeBuf.hasRemaining()) {
          ch.write(writeBuf);
        }
        ch.force(true);
      }
    }
  }
}
