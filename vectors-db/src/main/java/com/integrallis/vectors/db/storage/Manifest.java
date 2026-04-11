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
 * <p>Layout (little-endian throughout):
 *
 * <pre>
 * Offset  Size  Field                       Notes
 * ------  ----  --------------------------  --------------------------------
 *   0      4    magic                       FileFormat.MAGIC_MANIFEST
 *   4      4    format version              FileFormat.VERSION_MANIFEST
 *   8      4    header length               bytes from offset 0 to end of header
 *  12      4    flags                        reserved, must be 0
 *  16      4    dimension                   vector dimension
 *  20      4    metric ordinal              SimilarityFunction.ordinal()
 *  24      4    index type ordinal          IndexType.ordinal()
 *  28      4    quantizer kind ordinal      QuantizerKind.ordinal()
 *  32      8    generation number           int64
 *  40      8    created epoch millis        int64
 *  48      8    live count                  int64
 *  56      8    vectors.bin length          int64
 *  64      8    vectors.bin CRC32           uint32 zero-extended (see note)
 *  72      8    metadata.bin length         int64
 *  80      8    metadata.bin CRC32          uint32 zero-extended (see note)
 *  88      8    idmap.bin length            int64
 *  96      8    idmap.bin CRC32             uint32 zero-extended (see note)
 * 104      4    self CRC32                  CRC32 over bytes [0, 104)
 * ------  ----  --------------------------
 * 108      -    (future extension area — version bump required to grow)
 * </pre>
 *
 * <p><b>CRC width asymmetry.</b> The per-file CRCs each occupy 8 bytes on disk even though the
 * current {@link java.util.zip.CRC32} is only 32 bits wide; the upper 32 bits are stored as zero.
 * This leaves room to upgrade the per-file hash to a 64-bit function (xxHash3, CRC64) without
 * changing the header layout or requiring a version bump — only the self-CRC would need to grow on
 * that transition. The self-CRC stays 4 bytes because it covers the fixed-size header and the cost
 * of strengthening it would have to be paid before the rest of the header could be validated.
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
    long idmapBinCrc32) {

  /** Total fixed header size on disk, including the self CRC. */
  public static final int HEADER_SIZE = 108;

  /** Offset in bytes at which the self-CRC32 word lives. */
  public static final int SELF_CRC_OFFSET = 104;

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
    if (vectorsBinLength < 0 || metadataBinLength < 0 || idmapBinLength < 0) {
      throw new IllegalArgumentException("file lengths must be >= 0");
    }
  }

  /**
   * Builds a manifest record from the current config plus per-file sizes and CRCs, stamping the
   * creation time from the system clock.
   *
   * <p>Prefer {@link #build(VectorCollectionConfig, long, long, long, long, long, long, long, long,
   * long)} from tests that need a deterministic {@code createdEpochMillis}.
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
      long idmapBinCrc32) {
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
        idmapBinCrc32);
  }

  /**
   * Clock-injectable variant of {@link #build(VectorCollectionConfig, long, long, long, long, long,
   * long, long, long)}. Callers that need reproducible manifests (tests, deterministic fixtures)
   * pass {@code createdEpochMillis} explicitly; production callers use the system-clock overload.
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
      long idmapBinCrc32) {
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
        idmapBinCrc32);
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
    buf.putInt(0); // flags reserved
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
    if (flags != 0) {
      throw new IOException("Manifest flags must be 0 in version 1, got " + flags);
    }

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
        idmapBinCrc32);
  }

  /**
   * Writes this manifest to {@code file} and fsyncs before returning. The file is opened with
   * {@link StandardOpenOption#CREATE_NEW}, so this is a one-shot write: reinvoking it on an
   * existing path is a programmer error, not an overwrite.
   *
   * @throws java.nio.file.FileAlreadyExistsException if {@code file} already exists — the
   *     generation-write pipeline MUST target a fresh path inside an in-flight {@code
   *     .gen-NNNNNNNNNNNNNNNN.tmp} directory that was created earlier in the same commit
   * @throws IOException if the target cannot be created or the fsync fails
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
