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

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.QuantizerKind;
import com.integrallis.vectors.db.VectorCollectionConfig;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Objects;

/**
 * Self-describing header for a persistent generation directory. Every field is stored at a fixed
 * offset so a reader can decide whether the generation is intact by reading exactly {@link
 * #HEADER_SIZE} bytes and validating the self-CRC — no heap allocation, no schema parsing.
 *
 * <p>Layout (little-endian throughout, version 4 — added tombstone fields in Step 6):
 *
 * <pre>
 * Offset  Size  Field                       Notes
 * ------  ----  --------------------------  --------------------------------
 *   0      4    magic                       FileFormat.MAGIC_MANIFEST
 *   4      4    format version              FileFormat.VERSION_MANIFEST (= 4)
 *   8      4    header length               bytes from offset 0 to end of header (= 164)
 *  12      4    flags                        bit 0 = vectorsNormalized (#A cosine unit-norm);
 *                                            all other bits reserved, must be 0
 *  16      4    dimension                   vector dimension
 *  20      4    metric ordinal              SimilarityFunction.ordinal()
 *  24      4    index type ordinal          IndexType.ordinal()
 *  28      4    quantizer kind ordinal      QuantizerKind.ordinal()
 *  32      8    generation number           int64
 *  40      8    created epoch millis        int64
 *  48      8    live count                  int64
 *  56      8    vectors.bin length          int64
 *  64      8    vectors.bin CRC32           uint32 zero-extended
 *  72      8    metadata.bin length         int64
 *  80      8    metadata.bin CRC32          uint32 zero-extended
 *  88      8    idmap.bin length            int64
 *  96      8    idmap.bin CRC32             uint32 zero-extended
 * 104      8    graph.bin length            int64 (0 if no graph file written)
 * 112      8    graph.bin CRC32             uint32 zero-extended, 0 if no graph file
 * 120      8    quantized.bin length        int64 (0 if no quantized file written)
 * 128      8    quantized.bin CRC32         uint32 zero-extended, 0 if no quantized file
 * 136      8    tombstone count             int64 (0 if no tombstones)
 * 144      8    tombstones.bin length       int64 (0 if no tombstones file written)
 * 152      8    tombstones.bin CRC32        uint32 zero-extended, 0 if no tombstones file
 * 160      4    self CRC32                  CRC32 over bytes [0, 160)
 * ------  ----  --------------------------
 * 164      -    (future extension area — version bump required to grow)
 * </pre>
 *
 * <p><b>CRC width asymmetry.</b> The per-file CRCs each occupy 8 bytes on disk even though the
 * current {@link java.util.zip.CRC32} is only 32 bits wide; the upper 32 bits are stored as zero.
 * This leaves room to upgrade the per-file hash to a 64-bit function (xxHash3, CRC64) without
 * changing the header layout or requiring a version bump.
 *
 * <p>Every CRC is {@link java.util.zip.CRC32} (IEEE polynomial). The self-CRC covers every byte
 * before itself, so a corrupted manifest fails validation before any downstream file is opened.
 *
 * <p>Equality and hashing on a {@code Manifest} are based on every on-disk field <i>including</i>
 * the CRCs, so two manifests are equal iff they describe the same on-disk bytes.
 */
public record Manifest(
    int dimension,
    SimilarityFunction metric,
    IndexType indexType,
    QuantizerKind quantizerKind,
    long generationNumber,
    long createdEpochMillis,
    long liveCount,
    long vectorsBinLength,
    long vectorsBinCrc32,
    long metadataBinLength,
    long metadataBinCrc32,
    long idmapBinLength,
    long idmapBinCrc32,
    long graphBinLength,
    long graphBinCrc32,
    long quantizedBinLength,
    long quantizedBinCrc32,
    long tombstoneCount,
    long tombstonesBinLength,
    long tombstonesBinCrc32,
    boolean vectorsNormalized) {

  /** Total fixed header size on disk, including the self CRC. */
  public static final int HEADER_SIZE = 164;

  /** Offset in bytes at which the self-CRC32 word lives. */
  public static final int SELF_CRC_OFFSET = 160;

  /** {@code flags} bit set when {@code vectors.bin} holds L2-unit-normalized vectors (#A). */
  public static final int FLAG_VECTORS_NORMALIZED = 0x1;

  /**
   * Backward-compatible constructor without {@code vectorsNormalized}, defaulting it to {@code
   * false} (the pre-#A behavior: vectors stored verbatim). Preserves every call site written before
   * the #A cosine unit-normalization optimization was added.
   */
  public Manifest(
      int dimension,
      SimilarityFunction metric,
      IndexType indexType,
      QuantizerKind quantizerKind,
      long generationNumber,
      long createdEpochMillis,
      long liveCount,
      long vectorsBinLength,
      long vectorsBinCrc32,
      long metadataBinLength,
      long metadataBinCrc32,
      long idmapBinLength,
      long idmapBinCrc32,
      long graphBinLength,
      long graphBinCrc32,
      long quantizedBinLength,
      long quantizedBinCrc32,
      long tombstoneCount,
      long tombstonesBinLength,
      long tombstonesBinCrc32) {
    this(
        dimension,
        metric,
        indexType,
        quantizerKind,
        generationNumber,
        createdEpochMillis,
        liveCount,
        vectorsBinLength,
        vectorsBinCrc32,
        metadataBinLength,
        metadataBinCrc32,
        idmapBinLength,
        idmapBinCrc32,
        graphBinLength,
        graphBinCrc32,
        quantizedBinLength,
        quantizedBinCrc32,
        tombstoneCount,
        tombstonesBinLength,
        tombstonesBinCrc32,
        false);
  }

  public Manifest {
    if (dimension <= 0) {
      throw new IllegalArgumentException("dimension must be positive: " + dimension);
    }
    Objects.requireNonNull(metric, "metric must not be null");
    Objects.requireNonNull(indexType, "indexType must not be null");
    Objects.requireNonNull(quantizerKind, "quantizerKind must not be null");
    if (generationNumber < 0) {
      throw new IllegalArgumentException("generationNumber must be >= 0: " + generationNumber);
    }
    if (createdEpochMillis < 0) {
      throw new IllegalArgumentException("createdEpochMillis must be >= 0: " + createdEpochMillis);
    }
    if (liveCount < 0) {
      throw new IllegalArgumentException("liveCount must be >= 0: " + liveCount);
    }
    if (vectorsBinLength < 0
        || metadataBinLength < 0
        || idmapBinLength < 0
        || graphBinLength < 0
        || quantizedBinLength < 0
        || tombstonesBinLength < 0) {
      throw new IllegalArgumentException("file lengths must be >= 0");
    }
    if (tombstoneCount < 0) {
      throw new IllegalArgumentException("tombstoneCount must be >= 0: " + tombstoneCount);
    }
    // Cross-field invariants on the tombstone triplet — without these, an inverted manifest
    // could claim "N tombstones in 0 bytes" or "0 tombstones with a non-zero CRC" and still
    // pass the per-field non-negative checks. TombstoneCodec.encode emits zero bytes iff the
    // bitset is empty, so the count and the bin length must agree on emptiness.
    if ((tombstoneCount == 0L) != (tombstonesBinLength == 0L)) {
      throw new IllegalArgumentException(
          "tombstoneCount and tombstonesBinLength must both be zero or both be non-zero: "
              + "tombstoneCount="
              + tombstoneCount
              + ", tombstonesBinLength="
              + tombstonesBinLength);
    }
    if (tombstonesBinLength == 0L && tombstonesBinCrc32 != 0L) {
      throw new IllegalArgumentException(
          "tombstonesBinCrc32 must be 0 when tombstonesBinLength is 0: " + tombstonesBinCrc32);
    }
  }

  /**
   * Builds a manifest record from the current config plus per-file sizes and CRCs, stamping the
   * creation time from the system clock.
   *
   * <p>Prefer the clock-injectable overload from tests that need a deterministic {@code
   * createdEpochMillis}.
   */
  public static Manifest build(
      VectorCollectionConfig config,
      long generationNumber,
      long liveCount,
      long vectorsBinLength,
      long vectorsBinCrc32,
      long metadataBinLength,
      long metadataBinCrc32,
      long idmapBinLength,
      long idmapBinCrc32,
      long graphBinLength,
      long graphBinCrc32,
      long quantizedBinLength,
      long quantizedBinCrc32) {
    return build(
        config,
        generationNumber,
        System.currentTimeMillis(),
        liveCount,
        vectorsBinLength,
        vectorsBinCrc32,
        metadataBinLength,
        metadataBinCrc32,
        idmapBinLength,
        idmapBinCrc32,
        graphBinLength,
        graphBinCrc32,
        quantizedBinLength,
        quantizedBinCrc32,
        0L,
        0L,
        0L);
  }

  /**
   * Clock-injectable variant without tombstone fields. Backwards-compatible with existing call
   * sites that don't need tombstone support.
   */
  public static Manifest build(
      VectorCollectionConfig config,
      long generationNumber,
      long createdEpochMillis,
      long liveCount,
      long vectorsBinLength,
      long vectorsBinCrc32,
      long metadataBinLength,
      long metadataBinCrc32,
      long idmapBinLength,
      long idmapBinCrc32,
      long graphBinLength,
      long graphBinCrc32,
      long quantizedBinLength,
      long quantizedBinCrc32) {
    return build(
        config,
        generationNumber,
        createdEpochMillis,
        liveCount,
        vectorsBinLength,
        vectorsBinCrc32,
        metadataBinLength,
        metadataBinCrc32,
        idmapBinLength,
        idmapBinCrc32,
        graphBinLength,
        graphBinCrc32,
        quantizedBinLength,
        quantizedBinCrc32,
        0L,
        0L,
        0L);
  }

  /**
   * Full clock-injectable variant with tombstone fields. Callers that need reproducible manifests
   * (tests, deterministic fixtures) pass {@code createdEpochMillis} explicitly; production callers
   * use the system-clock overload.
   */
  public static Manifest build(
      VectorCollectionConfig config,
      long generationNumber,
      long createdEpochMillis,
      long liveCount,
      long vectorsBinLength,
      long vectorsBinCrc32,
      long metadataBinLength,
      long metadataBinCrc32,
      long idmapBinLength,
      long idmapBinCrc32,
      long graphBinLength,
      long graphBinCrc32,
      long quantizedBinLength,
      long quantizedBinCrc32,
      long tombstoneCount,
      long tombstonesBinLength,
      long tombstonesBinCrc32) {
    return new Manifest(
        config.dimension(),
        config.metric(),
        config.indexType(),
        config.quantizerKind(),
        generationNumber,
        createdEpochMillis,
        liveCount,
        vectorsBinLength,
        vectorsBinCrc32,
        metadataBinLength,
        metadataBinCrc32,
        idmapBinLength,
        idmapBinCrc32,
        graphBinLength,
        graphBinCrc32,
        quantizedBinLength,
        quantizedBinCrc32,
        tombstoneCount,
        tombstonesBinLength,
        tombstonesBinCrc32,
        // #A: record whether vectors.bin holds unit-normalized vectors so a reopened collection
        // restores the same normalize/DOT-scoring decision. config.metric() stays the TRUE metric.
        config.normalizeForCosine());
  }

  /**
   * System-clock overload with tombstone fields. Production callers use this when tombstones are
   * present.
   */
  public static Manifest buildWithTombstones(
      VectorCollectionConfig config,
      long generationNumber,
      long liveCount,
      long vectorsBinLength,
      long vectorsBinCrc32,
      long metadataBinLength,
      long metadataBinCrc32,
      long idmapBinLength,
      long idmapBinCrc32,
      long graphBinLength,
      long graphBinCrc32,
      long quantizedBinLength,
      long quantizedBinCrc32,
      long tombstoneCount,
      long tombstonesBinLength,
      long tombstonesBinCrc32) {
    return build(
        config,
        generationNumber,
        System.currentTimeMillis(),
        liveCount,
        vectorsBinLength,
        vectorsBinCrc32,
        metadataBinLength,
        metadataBinCrc32,
        idmapBinLength,
        idmapBinCrc32,
        graphBinLength,
        graphBinCrc32,
        quantizedBinLength,
        quantizedBinCrc32,
        tombstoneCount,
        tombstonesBinLength,
        tombstonesBinCrc32);
  }

  /**
   * Serializes this manifest to a fresh byte array of length exactly {@link #HEADER_SIZE},
   * including the trailing self-CRC word.
   */
  public byte[] toBytes() {
    byte[] out = new byte[HEADER_SIZE];
    ByteBuffer buf = ByteBuffer.wrap(out).order(ByteOrder.LITTLE_ENDIAN);
    buf.putInt(FileFormat.MAGIC_MANIFEST);
    buf.putInt(FileFormat.VERSION_MANIFEST);
    buf.putInt(HEADER_SIZE);
    buf.putInt(vectorsNormalized ? FLAG_VECTORS_NORMALIZED : 0); // flags: bit 0 = vectorsNormalized
    buf.putInt(dimension);
    buf.putInt(metric.ordinal());
    buf.putInt(indexType.ordinal());
    buf.putInt(quantizerKind.ordinal());
    buf.putLong(generationNumber);
    buf.putLong(createdEpochMillis);
    buf.putLong(liveCount);
    buf.putLong(vectorsBinLength);
    buf.putLong(vectorsBinCrc32);
    buf.putLong(metadataBinLength);
    buf.putLong(metadataBinCrc32);
    buf.putLong(idmapBinLength);
    buf.putLong(idmapBinCrc32);
    buf.putLong(graphBinLength);
    buf.putLong(graphBinCrc32);
    buf.putLong(quantizedBinLength);
    buf.putLong(quantizedBinCrc32);
    buf.putLong(tombstoneCount);
    buf.putLong(tombstonesBinLength);
    buf.putLong(tombstonesBinCrc32);
    // Self-CRC over bytes [0, SELF_CRC_OFFSET).
    long selfCrc = Checksums.ofBytes(out, 0, SELF_CRC_OFFSET);
    buf.putInt((int) selfCrc);
    return out;
  }

  /**
   * Parses a manifest from exactly {@link #HEADER_SIZE} bytes. Validates magic, version, header
   * length, and self-CRC. Throws {@link IOException} with a descriptive message on any mismatch.
   */
  public static Manifest fromBytes(byte[] bytes) throws IOException {
    Objects.requireNonNull(bytes, "bytes must not be null");
    if (bytes.length < HEADER_SIZE) {
      throw new IOException(
          "Manifest truncated: expected at least " + HEADER_SIZE + " bytes, got " + bytes.length);
    }
    ByteBuffer buf = ByteBuffer.wrap(bytes, 0, HEADER_SIZE).order(ByteOrder.LITTLE_ENDIAN);

    int magic = buf.getInt();
    if (magic != FileFormat.MAGIC_MANIFEST) {
      throw new IOException(
          String.format(
              "Manifest magic mismatch: expected 0x%08x, got 0x%08x",
              FileFormat.MAGIC_MANIFEST, magic));
    }
    int version = buf.getInt();
    if (version != FileFormat.VERSION_MANIFEST) {
      throw new IOException(
          "Manifest version mismatch: expected "
              + FileFormat.VERSION_MANIFEST
              + ", got "
              + version);
    }
    int headerLength = buf.getInt();
    if (headerLength != HEADER_SIZE) {
      throw new IOException(
          "Manifest header length mismatch: expected " + HEADER_SIZE + ", got " + headerLength);
    }
    int flags = buf.getInt();
    if ((flags & ~FLAG_VECTORS_NORMALIZED) != 0) {
      throw new IOException(
          "Manifest flags has unknown bits set (only bit 0 = vectorsNormalized is defined): "
              + flags);
    }
    boolean vectorsNormalized = (flags & FLAG_VECTORS_NORMALIZED) != 0;

    int dimension = buf.getInt();
    int metricOrdinal = buf.getInt();
    int indexTypeOrdinal = buf.getInt();
    int quantizerKindOrdinal = buf.getInt();
    long generationNumber = buf.getLong();
    long createdEpochMillis = buf.getLong();
    long liveCount = buf.getLong();
    long vectorsBinLength = buf.getLong();
    long vectorsBinCrc32 = buf.getLong();
    long metadataBinLength = buf.getLong();
    long metadataBinCrc32 = buf.getLong();
    long idmapBinLength = buf.getLong();
    long idmapBinCrc32 = buf.getLong();
    long graphBinLength = buf.getLong();
    long graphBinCrc32 = buf.getLong();
    long quantizedBinLength = buf.getLong();
    long quantizedBinCrc32 = buf.getLong();
    long tombstoneCount = buf.getLong();
    long tombstonesBinLength = buf.getLong();
    long tombstonesBinCrc32 = buf.getLong();
    int selfCrc = buf.getInt();

    long expectedSelfCrc = Checksums.ofBytes(bytes, 0, SELF_CRC_OFFSET);
    if ((selfCrc & 0xFFFFFFFFL) != expectedSelfCrc) {
      throw new IOException(
          String.format(
              "Manifest self-CRC mismatch: stored=0x%08x, computed=0x%08x",
              selfCrc, expectedSelfCrc));
    }

    SimilarityFunction metric =
        ordinalOrThrow(SimilarityFunction.values(), metricOrdinal, "metric");
    IndexType indexType = ordinalOrThrow(IndexType.values(), indexTypeOrdinal, "indexType");
    QuantizerKind quantizerKind =
        ordinalOrThrow(QuantizerKind.values(), quantizerKindOrdinal, "quantizerKind");

    return new Manifest(
        dimension,
        metric,
        indexType,
        quantizerKind,
        generationNumber,
        createdEpochMillis,
        liveCount,
        vectorsBinLength,
        vectorsBinCrc32,
        metadataBinLength,
        metadataBinCrc32,
        idmapBinLength,
        idmapBinCrc32,
        graphBinLength,
        graphBinCrc32,
        quantizedBinLength,
        quantizedBinCrc32,
        tombstoneCount,
        tombstonesBinLength,
        tombstonesBinCrc32,
        vectorsNormalized);
  }

  /**
   * Writes this manifest to {@code file} and fsyncs before returning. The file is opened with
   * {@link StandardOpenOption#CREATE_NEW}, so this is a one-shot write: reinvoking it on an
   * existing path is a programmer error, not an overwrite.
   */
  public void writeTo(Path file) throws IOException {
    byte[] encoded = toBytes();
    try (FileChannel ch =
        FileChannel.open(file, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE)) {
      ByteBuffer buf = ByteBuffer.wrap(encoded);
      while (buf.hasRemaining()) {
        ch.write(buf);
      }
      ch.force(true);
    }
  }

  /** Reads and validates a manifest from {@code file}. */
  public static Manifest readFrom(Path file) throws IOException {
    byte[] bytes = Files.readAllBytes(file);
    return fromBytes(bytes);
  }

  private static <E extends Enum<E>> E ordinalOrThrow(E[] values, int ordinal, String name)
      throws IOException {
    if (ordinal < 0 || ordinal >= values.length) {
      throw new IOException(
          "Manifest "
              + name
              + " ordinal out of range: "
              + ordinal
              + " (valid range 0-"
              + (values.length - 1)
              + ")");
    }
    return values[ordinal];
  }
}
