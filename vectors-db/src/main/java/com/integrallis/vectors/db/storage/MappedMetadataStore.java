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

import com.integrallis.vectors.db.Document;
import com.integrallis.vectors.db.MetadataValue;
import com.integrallis.vectors.db.metadata.MetadataStore;
import java.io.ByteArrayOutputStream;
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
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Read-only mmap-backed {@link MetadataStore}. Mirrors the shape of {@link MappedIdMapper}: a
 * static {@link Writer#writeTo(Path, List)} serializes a {@code List<Document>} into one sealed
 * {@code metadata.bin} file, and {@link #open(Path, Arena)} mmaps that file through the
 * caller-supplied arena and materializes the entire document table on heap.
 *
 * <p>The file layout is:
 *
 * <pre>
 * Offset  Size  Field                         Notes
 * ------  ----  ----------------------------  --------------------------------
 *   0      4    magic                         {@link FileFormat#MAGIC_METADATA} ("VMDB")
 *   4      4    version                       {@link FileFormat#VERSION_METADATA}
 *   8      4    count                         number of documents
 *  12      4    heap byte length              total size of the document heap
 *  16      8*N  offset table                  int64 per ordinal: offset into heap
 * 16+8*N  -    heap                          per entry: see {@code encodeDocument}
 * </pre>
 *
 * <p>Each heap entry encodes a single {@link Document} as:
 *
 * <pre>
 *   varint len(id)   / utf8 bytes(id)
 *   byte has-text    / (varint len(text) / utf8 bytes(text))?
 *   varint metadataEntryCount
 *   [ varint len(key) / utf8 bytes(key) / byte tag / payload ]*
 * </pre>
 *
 * <p>Tag values: {@code 1 = Str}, {@code 2 = Num}, {@code 3 = Bool}, {@code 4 = Tags}. Payload
 * format per tag:
 *
 * <ul>
 *   <li>Str: varint len + utf8 bytes
 *   <li>Num: 8 bytes IEEE 754 double LE
 *   <li>Bool: 1 byte (0 or 1)
 *   <li>Tags: varint count + (varint len + utf8 bytes)*
 * </ul>
 *
 * <p><b>Vector field.</b> The {@code vector} component of each {@link Document} is <i>never</i>
 * stored in {@code metadata.bin} — it lives in {@code vectors.bin} (managed by {@code
 * MappedVectorStore}). On read, every {@link Document} is materialized with {@code vector = null};
 * search-result projections hydrate the vector out of the vector store in a separate step.
 *
 * <p><b>Writes</b> go through the nested {@link Writer} — a mapped store is strictly immutable once
 * opened, and {@link #put(int, Document)} / {@link #delete(int)} both throw {@link
 * UnsupportedOperationException}. The commit path in {@code VectorCollectionImpl} materializes a
 * successor generation by calling {@link Writer#writeTo(Path, List)} then opening the result
 * through {@link #open(Path, Arena)}.
 *
 * <p><b>Close semantics.</b> {@link #close()} is a no-op — the enclosing {@link Arena} owns the
 * mmap lifetime and the owning {@code Generation} closes the arena exactly once when its refcount
 * drops to zero.
 */
public final class MappedMetadataStore implements MetadataStore {

  /**
   * Size in bytes of the fixed header (magic + version + count + heap length). Delegates to {@link
   * FileFormat#STORE_HEADER_SIZE} so that the layout constant lives in exactly one place — both
   * this class and {@link MappedIdMapper} share the same 16-byte header shape.
   */
  public static final int HEADER_SIZE = FileFormat.STORE_HEADER_SIZE;

  /** Tag byte for {@link MetadataValue.Str}. */
  static final byte TAG_STR = 1;

  /** Tag byte for {@link MetadataValue.Num}. */
  static final byte TAG_NUM = 2;

  /** Tag byte for {@link MetadataValue.Bool}. */
  static final byte TAG_BOOL = 3;

  /** Tag byte for {@link MetadataValue.Tags}. */
  static final byte TAG_TAGS = 4;

  private static final ValueLayout.OfInt INT_LE =
      ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
  private static final ValueLayout.OfLong LONG_LE =
      ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
  private static final ValueLayout.OfDouble DOUBLE_LE =
      ValueLayout.JAVA_DOUBLE_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

  private final int count;
  private final Document[] byOrdinal;

  private MappedMetadataStore(int count, Document[] byOrdinal) {
    this.count = count;
    this.byOrdinal = byOrdinal;
  }

  /**
   * Opens and validates a persistent metadata file. The caller supplies the {@link Arena} that owns
   * the mmap lifetime — typically a shared arena per generation, closed exactly once by the owning
   * {@code Generation} record.
   *
   * @throws IOException if the file is truncated, has the wrong magic/version, contains an
   *     out-of-range offset, or contains a malformed document entry
   */
  public static MappedMetadataStore open(Path file, Arena arena) throws IOException {
    Objects.requireNonNull(file, "file must not be null");
    Objects.requireNonNull(arena, "arena must not be null");

    MemorySegment seg;
    long fileSize;
    try (FileChannel ch = FileChannel.open(file, StandardOpenOption.READ)) {
      fileSize = ch.size();
      if (fileSize < HEADER_SIZE) {
        throw new IOException(
            "metadata.bin truncated: expected at least " + HEADER_SIZE + " bytes, got " + fileSize);
      }
      seg = ch.map(FileChannel.MapMode.READ_ONLY, 0, fileSize, arena);
    }

    int magic = seg.get(INT_LE, 0);
    if (magic != FileFormat.MAGIC_METADATA) {
      throw new IOException(
          String.format(
              "metadata magic mismatch: expected 0x%08x, got 0x%08x",
              FileFormat.MAGIC_METADATA, magic));
    }
    int version = seg.get(INT_LE, 4);
    if (version != FileFormat.VERSION_METADATA) {
      throw new IOException(
          "metadata version mismatch: expected "
              + FileFormat.VERSION_METADATA
              + ", got "
              + version);
    }
    int count = seg.get(INT_LE, 8);
    int heapByteLength = seg.get(INT_LE, 12);
    if (count < 0) {
      throw new IOException("metadata count must be >= 0: " + count);
    }
    if (heapByteLength < 0) {
      throw new IOException("metadata heap byte length must be >= 0: " + heapByteLength);
    }

    long offsetTableStart = HEADER_SIZE;
    long heapStart = offsetTableStart + (long) count * Long.BYTES;
    long heapEnd = heapStart + heapByteLength;
    long expectedSize = heapEnd;
    if (fileSize < expectedSize) {
      throw new IOException(
          "metadata.bin truncated: header declares "
              + expectedSize
              + " bytes, file is "
              + fileSize);
    }

    Document[] docs = new Document[count];
    for (int i = 0; i < count; i++) {
      long heapOff = seg.get(LONG_LE, offsetTableStart + (long) i * Long.BYTES);
      if (heapOff < 0 || heapOff >= heapByteLength) {
        throw new IOException(
            "metadata heap offset out of range for ordinal " + i + ": " + heapOff);
      }
      docs[i] = readDocument(seg, heapStart + heapOff, heapEnd, i);
    }

    return new MappedMetadataStore(count, docs);
  }

  private static Document readDocument(MemorySegment seg, long startOffset, long heapEnd, int ord)
      throws IOException {
    long offset = startOffset;

    // id
    long packed = FileFormat.readVarIntFromSegment(seg, offset);
    int idLen = (int) packed;
    int varintBytes = (int) (packed >>> 32);
    offset += varintBytes;
    if (idLen < 0 || offset + idLen > heapEnd) {
      throw new IOException("metadata id out of range for ordinal " + ord + ": len=" + idLen);
    }
    byte[] idBytes = new byte[idLen];
    MemorySegment.copy(seg, ValueLayout.JAVA_BYTE, offset, idBytes, 0, idLen);
    String id = new String(idBytes, StandardCharsets.UTF_8);
    offset += idLen;

    // has-text flag + optional text
    if (offset >= heapEnd) {
      throw new IOException("metadata truncated before has-text flag for ordinal " + ord);
    }
    byte hasText = seg.get(ValueLayout.JAVA_BYTE, offset);
    offset += 1;
    String text = null;
    if (hasText == 1) {
      packed = FileFormat.readVarIntFromSegment(seg, offset);
      int textLen = (int) packed;
      varintBytes = (int) (packed >>> 32);
      offset += varintBytes;
      if (textLen < 0 || offset + textLen > heapEnd) {
        throw new IOException("metadata text out of range for ordinal " + ord);
      }
      byte[] textBytes = new byte[textLen];
      MemorySegment.copy(seg, ValueLayout.JAVA_BYTE, offset, textBytes, 0, textLen);
      text = new String(textBytes, StandardCharsets.UTF_8);
      offset += textLen;
    } else if (hasText != 0) {
      throw new IOException(
          "metadata has-text flag must be 0 or 1 for ordinal " + ord + ": " + hasText);
    }

    // metadata entries
    packed = FileFormat.readVarIntFromSegment(seg, offset);
    int entryCount = (int) packed;
    varintBytes = (int) (packed >>> 32);
    offset += varintBytes;
    if (entryCount < 0) {
      throw new IOException(
          "metadata entry count out of range for ordinal " + ord + ": " + entryCount);
    }

    Map<String, MetadataValue> metadata;
    if (entryCount == 0) {
      metadata = Map.of();
    } else {
      metadata = LinkedHashMap.newLinkedHashMap(entryCount);
      for (int e = 0; e < entryCount; e++) {
        packed = FileFormat.readVarIntFromSegment(seg, offset);
        int keyLen = (int) packed;
        varintBytes = (int) (packed >>> 32);
        offset += varintBytes;
        if (keyLen < 0 || offset + keyLen > heapEnd) {
          throw new IOException("metadata key out of range for ordinal " + ord);
        }
        byte[] keyBytes = new byte[keyLen];
        MemorySegment.copy(seg, ValueLayout.JAVA_BYTE, offset, keyBytes, 0, keyLen);
        String key = new String(keyBytes, StandardCharsets.UTF_8);
        offset += keyLen;

        if (offset >= heapEnd) {
          throw new IOException("metadata truncated before tag for ordinal " + ord);
        }
        byte tag = seg.get(ValueLayout.JAVA_BYTE, offset);
        offset += 1;

        MetadataValue value;
        switch (tag) {
          case TAG_STR -> {
            packed = FileFormat.readVarIntFromSegment(seg, offset);
            int strLen = (int) packed;
            varintBytes = (int) (packed >>> 32);
            offset += varintBytes;
            if (strLen < 0 || offset + strLen > heapEnd) {
              throw new IOException("metadata str value out of range for ordinal " + ord);
            }
            byte[] strBytes = new byte[strLen];
            MemorySegment.copy(seg, ValueLayout.JAVA_BYTE, offset, strBytes, 0, strLen);
            value = new MetadataValue.Str(new String(strBytes, StandardCharsets.UTF_8));
            offset += strLen;
          }
          case TAG_NUM -> {
            if (offset + 8 > heapEnd) {
              throw new IOException("metadata num value truncated for ordinal " + ord);
            }
            double d = seg.get(DOUBLE_LE, offset);
            offset += 8;
            value = new MetadataValue.Num(d);
          }
          case TAG_BOOL -> {
            if (offset >= heapEnd) {
              throw new IOException("metadata bool value truncated for ordinal " + ord);
            }
            byte b = seg.get(ValueLayout.JAVA_BYTE, offset);
            offset += 1;
            if (b != 0 && b != 1) {
              throw new IOException(
                  "metadata bool value must be 0 or 1 for ordinal " + ord + ": " + b);
            }
            value = new MetadataValue.Bool(b == 1);
          }
          case TAG_TAGS -> {
            packed = FileFormat.readVarIntFromSegment(seg, offset);
            int tagCount = (int) packed;
            varintBytes = (int) (packed >>> 32);
            offset += varintBytes;
            if (tagCount < 0) {
              throw new IOException("metadata tags count out of range for ordinal " + ord);
            }
            List<String> tagList = new ArrayList<>(tagCount);
            for (int t = 0; t < tagCount; t++) {
              packed = FileFormat.readVarIntFromSegment(seg, offset);
              int tlen = (int) packed;
              varintBytes = (int) (packed >>> 32);
              offset += varintBytes;
              if (tlen < 0 || offset + tlen > heapEnd) {
                throw new IOException("metadata tag value out of range for ordinal " + ord);
              }
              byte[] tagBytes = new byte[tlen];
              MemorySegment.copy(seg, ValueLayout.JAVA_BYTE, offset, tagBytes, 0, tlen);
              tagList.add(new String(tagBytes, StandardCharsets.UTF_8));
              offset += tlen;
            }
            value = new MetadataValue.Tags(tagList);
          }
          default ->
              throw new IOException(
                  "metadata unknown tag " + tag + " for ordinal " + ord + " key " + key);
        }
        metadata.put(key, value);
      }
    }

    return new Document(id, null, text, metadata);
  }

  @Override
  public Document get(int ordinal) {
    if (ordinal < 0 || ordinal >= count) {
      return null;
    }
    return byOrdinal[ordinal];
  }

  /**
   * Unsupported — a mapped store is read-only. The commit pipeline materializes a successor
   * generation via {@link Writer#writeTo(Path, List)} instead of per-entry mutation.
   */
  @Override
  public void put(int ordinal, Document document) {
    throw new UnsupportedOperationException("MappedMetadataStore is read-only");
  }

  /**
   * Unsupported — a mapped store is read-only. Deletes land in Step 6 via a tombstone bitmap in a
   * successor generation.
   */
  @Override
  public void delete(int ordinal) {
    throw new UnsupportedOperationException("MappedMetadataStore is read-only");
  }

  @Override
  public Document[] bulkRead(int[] ordinals) {
    Objects.requireNonNull(ordinals, "ordinals must not be null");
    Document[] out = new Document[ordinals.length];
    for (int i = 0; i < ordinals.length; i++) {
      int o = ordinals[i];
      out[i] = (o < 0 || o >= count) ? null : byOrdinal[o];
    }
    return out;
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

  /** Static serializer that writes the complete {@code metadata.bin} layout. */
  public static final class Writer {

    private Writer() {}

    /**
     * Builds the complete {@code metadata.bin} byte image without touching the file system. The
     * commit pipeline calls this to pre-compute the bytes so it can checksum the content before
     * deciding where to write it. {@link #writeTo(Path, List)} is a thin wrapper that calls this
     * and streams the result to disk.
     *
     * <p>The {@code vector} component of each document is deliberately <b>not</b> serialized —
     * vectors live in {@code vectors.bin}, not here.
     *
     * @throws IOException if the heap or total size exceeds 2 GiB or any document is null
     */
    public static byte[] toBytes(List<Document> documents) throws IOException {
      Objects.requireNonNull(documents, "documents must not be null");

      int count = documents.size();
      byte[][] heapEntries = new byte[count][];
      long heapSize = 0;
      for (int i = 0; i < count; i++) {
        Document doc = documents.get(i);
        if (doc == null) {
          throw new IOException("metadata documents[" + i + "] is null");
        }
        byte[] entry = encodeDocument(doc);
        heapEntries[i] = entry;
        heapSize += entry.length;
      }
      if (heapSize > Integer.MAX_VALUE) {
        throw new IOException(
            "metadata heap size exceeds 2 GiB: "
                + heapSize
                + " (raise VDB_MAX_HEAP or split the collection)");
      }

      long totalSize = (long) HEADER_SIZE + (long) count * Long.BYTES + heapSize;
      if (totalSize > Integer.MAX_VALUE) {
        throw new IOException("metadata total size exceeds 2 GiB: " + totalSize);
      }

      byte[] out = new byte[(int) totalSize];
      ByteBuffer buf = ByteBuffer.wrap(out).order(ByteOrder.LITTLE_ENDIAN);
      buf.putInt(FileFormat.MAGIC_METADATA);
      buf.putInt(FileFormat.VERSION_METADATA);
      buf.putInt(count);
      buf.putInt((int) heapSize);

      long heapOff = 0;
      for (int i = 0; i < count; i++) {
        buf.putLong(heapOff);
        heapOff += heapEntries[i].length;
      }
      for (int i = 0; i < count; i++) {
        buf.put(heapEntries[i]);
      }

      return out;
    }

    /**
     * Writes a metadata store to {@code path}, fsyncing the file contents before returning. The
     * file must not already exist ({@link StandardOpenOption#CREATE_NEW}).
     *
     * <p>The {@code vector} component of each document is deliberately <b>not</b> serialized —
     * vectors live in {@code vectors.bin}, not here. Callers may pass documents whose {@code
     * vector} field is either null or non-null; either way the on-disk entry is the same.
     *
     * @param path target file (typically {@code gen-NNNN/metadata.bin} inside an in-flight tmp dir)
     * @param documents documents indexed by ordinal; {@code documents.get(i)} is the doc assigned
     *     to ordinal {@code i}
     * @throws IOException if the file cannot be created, the heap exceeds 2 GiB, or any document is
     *     null
     */
    public static void writeTo(Path path, List<Document> documents) throws IOException {
      Objects.requireNonNull(path, "path must not be null");
      byte[] out = toBytes(documents);
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

    private static byte[] encodeDocument(Document doc) {
      ByteArrayOutputStream baos = new ByteArrayOutputStream(64);

      // id
      byte[] idBytes = doc.id().getBytes(StandardCharsets.UTF_8);
      FileFormat.writeVarInt(baos, idBytes.length);
      baos.writeBytes(idBytes);

      // text
      if (doc.text() != null) {
        baos.write(1);
        byte[] textBytes = doc.text().getBytes(StandardCharsets.UTF_8);
        FileFormat.writeVarInt(baos, textBytes.length);
        baos.writeBytes(textBytes);
      } else {
        baos.write(0);
      }

      // metadata entries
      Map<String, MetadataValue> metadata = doc.metadata();
      FileFormat.writeVarInt(baos, metadata.size());
      for (Map.Entry<String, MetadataValue> e : metadata.entrySet()) {
        byte[] keyBytes = e.getKey().getBytes(StandardCharsets.UTF_8);
        FileFormat.writeVarInt(baos, keyBytes.length);
        baos.writeBytes(keyBytes);
        writeMetadataValue(baos, e.getValue());
      }
      return baos.toByteArray();
    }

    private static void writeMetadataValue(ByteArrayOutputStream out, MetadataValue v) {
      switch (v) {
        case MetadataValue.Str s -> {
          out.write(TAG_STR);
          byte[] b = s.value().getBytes(StandardCharsets.UTF_8);
          FileFormat.writeVarInt(out, b.length);
          out.writeBytes(b);
        }
        case MetadataValue.Num n -> {
          out.write(TAG_NUM);
          long bits = Double.doubleToRawLongBits(n.value());
          // IEEE 754 double, little-endian on disk.
          for (int i = 0; i < 8; i++) {
            out.write((int) ((bits >>> (i * 8)) & 0xFF));
          }
        }
        case MetadataValue.Bool bo -> {
          out.write(TAG_BOOL);
          out.write(bo.value() ? 1 : 0);
        }
        case MetadataValue.Tags t -> {
          out.write(TAG_TAGS);
          FileFormat.writeVarInt(out, t.values().size());
          for (String tag : t.values()) {
            byte[] tb = tag.getBytes(StandardCharsets.UTF_8);
            FileFormat.writeVarInt(out, tb.length);
            out.writeBytes(tb);
          }
        }
      }
    }
  }
}
