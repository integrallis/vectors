package com.integrallis.vectors.db.storage;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;

/**
 * Binary format constants, magic numbers, version numbers, and primitive codecs shared across every
 * persistent {@code vectors-db} file: {@code manifest.bin}, {@code idmap.bin}, {@code
 * metadata.bin}, and the {@code CURRENT} pointer file. The format is locked to {@linkplain
 * ByteOrder#LITTLE_ENDIAN little-endian} so that a collection written on one platform can be opened
 * on another (LE is native on x86 and ARM; the SIMD load path in {@code PanamaVectorUtilSupport}
 * also hardcodes LE via {@code FloatVector.fromMemorySegment(..., ByteOrder.LITTLE_ENDIAN)}).
 *
 * <p>This class is not instantiable — it is a pure namespace of constants and static helpers.
 */
public final class FileFormat {

  private FileFormat() {}

  // ---------------------------------------------------------------------------
  // Magic numbers — ASCII tokens embedded as little-endian int32. Chosen so the
  // raw bytes on disk spell a human-readable tag when viewed in a hex editor.
  // ---------------------------------------------------------------------------

  /** Magic for {@code manifest.bin}. Raw bytes spell {@code "VDBV"} (Vector DB Version). */
  public static final int MAGIC_MANIFEST = 0x56424456; // 'V' 'D' 'B' 'V' little-endian

  /** Magic for {@code idmap.bin}. Raw bytes spell {@code "VIDP"} (Vector ID Persistent). */
  public static final int MAGIC_IDMAP = 0x50444956; // 'V' 'I' 'D' 'P' little-endian

  /** Magic for {@code metadata.bin}. Raw bytes spell {@code "VMDB"} (Vector MetaData Bin). */
  public static final int MAGIC_METADATA = 0x42444D56; // 'V' 'M' 'D' 'B' little-endian

  /** Magic for {@code graph.bin}. Raw bytes spell {@code "VGPH"} (Vector GraPH). */
  public static final int MAGIC_GRAPH = 0x48504756; // 'V' 'G' 'P' 'H' little-endian

  // ---------------------------------------------------------------------------
  // Format versions — bumped whenever the on-disk layout changes in a
  // backward-incompatible way. Readers refuse files whose version they don't
  // recognize rather than silently interpreting garbage.
  // ---------------------------------------------------------------------------

  /**
   * Current manifest format version. Bumped to 2 in Step 4b: the header grew by 16 bytes to store
   * {@code graphBinLength} + {@code graphBinCrc32} for the new {@code graph.bin} file. Readers
   * refuse v1 files; there is no on-disk compat shim because {@code vectors-db} is unreleased.
   */
  public static final int VERSION_MANIFEST = 2;

  /** Current idmap format version. */
  public static final int VERSION_IDMAP = 1;

  /** Current metadata format version. */
  public static final int VERSION_METADATA = 1;

  /** Current graph.bin format version. */
  public static final int VERSION_GRAPH = 1;

  // ---------------------------------------------------------------------------
  // Directory protocol constants.
  // ---------------------------------------------------------------------------

  /** Name of the pointer file at the collection root. Holds an 8-byte int64 generation number. */
  public static final String CURRENT_FILE = "CURRENT";

  /** Name of the pointer file's tmp sibling during an in-flight atomic update. */
  public static final String CURRENT_TMP_FILE = "CURRENT.tmp";

  /** Basename used to format generation directories: {@code gen-NNNNNNNNNNNNNNNN} (16 digits). */
  public static final String GENERATION_DIR_PREFIX = "gen-";

  /**
   * Basename used to format in-flight tmp generation directories: {@code .gen-NNNNNNNNNNNNNNNN.tmp}
   * (16 digits).
   */
  public static final String GENERATION_TMP_DIR_PREFIX = ".gen-";

  /** Suffix for in-flight tmp generation directories. */
  public static final String GENERATION_TMP_DIR_SUFFIX = ".tmp";

  /**
   * Number of decimal digits used when zero-padding a generation number into a directory name.
   * Sixteen is chosen so {@code 10^16 - 1 ≈ 10 quadrillion} commits fit without widening, and so
   * lexicographic sort order on the file system matches numeric sort order. Any generation number ≥
   * {@code 10^16} would produce a wider name and break the sort invariant — hence the guard in
   * {@link #generationDirName(long)}.
   */
  public static final int GENERATION_DIGITS = 16;

  /** Inclusive upper bound on a generation number that still fits in {@link #GENERATION_DIGITS}. */
  public static final long MAX_GENERATION_NUMBER = 9_999_999_999_999_999L; // 10^16 - 1

  /** File name of the vectors file inside every generation directory. */
  public static final String VECTORS_FILE = "vectors.bin";

  /** File name of the idmap file inside every generation directory. */
  public static final String IDMAP_FILE = "idmap.bin";

  /** File name of the metadata file inside every generation directory. */
  public static final String METADATA_FILE = "metadata.bin";

  /** File name of the manifest file inside every generation directory. */
  public static final String MANIFEST_FILE = "manifest.bin";

  /**
   * File name of the HNSW graph topology file inside a generation directory. Only present when the
   * collection's {@code indexType} is {@link com.integrallis.vectors.db.IndexType#HNSW}; FLAT
   * generations omit the file entirely and the Manifest reports {@code graphBinLength == 0}.
   */
  public static final String GRAPH_FILE = "graph.bin";

  /**
   * Fixed header size in bytes used by both {@link
   * com.integrallis.vectors.db.storage.MappedIdMapper} and {@link
   * com.integrallis.vectors.db.storage.MappedMetadataStore}. The layout is identical in both
   * stores: {@code magic (4) + version (4) + count (4) + heap byte length (4) = 16}.
   */
  public static final int STORE_HEADER_SIZE = 16;

  /**
   * Formats a generation number as {@code gen-NNNNNNNNNNNNNNNN} (16-digit zero-padded). The padding
   * makes lexicographic sort order equal to numeric sort order, so {@link java.nio.file.Files#list}
   * + sort returns generations in commit order.
   *
   * @throws IllegalArgumentException if {@code generationNumber < 0} or {@code > } {@link
   *     #MAX_GENERATION_NUMBER}. A value above that bound would produce a 17-digit name that
   *     lexicographically precedes every 16-digit name, silently breaking the sort invariant.
   */
  public static String generationDirName(long generationNumber) {
    checkGenerationNumber(generationNumber);
    return String.format("%s%016d", GENERATION_DIR_PREFIX, generationNumber);
  }

  /** Formats the in-flight tmp directory name for a given generation number. */
  public static String generationTmpDirName(long generationNumber) {
    checkGenerationNumber(generationNumber);
    return String.format(
        "%s%016d%s", GENERATION_TMP_DIR_PREFIX, generationNumber, GENERATION_TMP_DIR_SUFFIX);
  }

  /**
   * Parses a {@code gen-NNNNNNNNNNNNNNNN} directory name back into its generation number. Returns
   * {@code -1} if {@code name} doesn't match the expected pattern (wrong prefix, wrong digit count,
   * non-numeric digits, or negative value).
   */
  public static long parseGenerationDirName(String name) {
    if (name == null || !name.startsWith(GENERATION_DIR_PREFIX)) {
      return -1L;
    }
    String digits = name.substring(GENERATION_DIR_PREFIX.length());
    if (digits.length() != GENERATION_DIGITS) {
      return -1L;
    }
    try {
      long n = Long.parseLong(digits);
      return n < 0 ? -1L : n;
    } catch (NumberFormatException e) {
      return -1L;
    }
  }

  /**
   * Parses an in-flight tmp directory name {@code .gen-NNNNNNNNNNNNNNNN.tmp} back into its
   * generation number. Returns {@code -1} if the name doesn't match the exact expected pattern,
   * including digit count — this is the "is this name something we own?" check used by the recovery
   * sweep so a coincidentally-named user file (e.g. {@code .gen-.tmp}) is not deleted.
   */
  public static long parseGenerationTmpDirName(String name) {
    if (name == null
        || !name.startsWith(GENERATION_TMP_DIR_PREFIX)
        || !name.endsWith(GENERATION_TMP_DIR_SUFFIX)) {
      return -1L;
    }
    int digitsStart = GENERATION_TMP_DIR_PREFIX.length();
    int digitsEnd = name.length() - GENERATION_TMP_DIR_SUFFIX.length();
    if (digitsEnd - digitsStart != GENERATION_DIGITS) {
      return -1L;
    }
    String digits = name.substring(digitsStart, digitsEnd);
    try {
      long n = Long.parseLong(digits);
      return n < 0 ? -1L : n;
    } catch (NumberFormatException e) {
      return -1L;
    }
  }

  private static void checkGenerationNumber(long generationNumber) {
    if (generationNumber < 0) {
      throw new IllegalArgumentException("generationNumber must be >= 0: " + generationNumber);
    }
    if (generationNumber > MAX_GENERATION_NUMBER) {
      throw new IllegalArgumentException(
          "generationNumber exceeds "
              + GENERATION_DIGITS
              + "-digit format ("
              + MAX_GENERATION_NUMBER
              + "): "
              + generationNumber);
    }
  }

  // ---------------------------------------------------------------------------
  // Varint codec — unsigned LEB128, matching Protobuf's base128. Single byte
  // for values in [0, 127]; two bytes in [128, 16383]; etc. Used for
  // length-prefixing UTF-8 strings and metadata values in idmap.bin /
  // metadata.bin. Never used on the SIMD scoring hot path.
  // ---------------------------------------------------------------------------

  /**
   * Writes an unsigned varint to {@code out} and returns the number of bytes written. Accepts any
   * non-negative {@code int}; throws for negatives because on-disk lengths are unsigned.
   */
  public static int writeVarInt(ByteBuffer out, int value) {
    if (value < 0) {
      throw new IllegalArgumentException("varint value must be non-negative: " + value);
    }
    int bytesWritten = 0;
    while ((value & ~0x7F) != 0) {
      out.put((byte) ((value & 0x7F) | 0x80));
      value >>>= 7;
      bytesWritten++;
    }
    out.put((byte) (value & 0x7F));
    return bytesWritten + 1;
  }

  /**
   * Writes an unsigned varint directly into a growing {@link ByteArrayOutputStream}. This overload
   * exists so that per-document encoders (e.g. {@code MappedMetadataStore.Writer.encodeDocument})
   * can share a single varint implementation with the {@link ByteBuffer}-based path instead of
   * duplicating it inside the store writer. Returns the number of bytes written.
   */
  public static int writeVarInt(ByteArrayOutputStream out, int value) {
    if (value < 0) {
      throw new IllegalArgumentException("varint value must be non-negative: " + value);
    }
    int bytesWritten = 0;
    while ((value & ~0x7F) != 0) {
      out.write((value & 0x7F) | 0x80);
      value >>>= 7;
      bytesWritten++;
    }
    out.write(value & 0x7F);
    return bytesWritten + 1;
  }

  /**
   * Reads an unsigned varint from {@code in} at its current position and advances the position by
   * the number of bytes consumed. Returns the decoded value. Throws {@link IOException} if the
   * varint is malformed (more than 5 continuation bytes or buffer underflow).
   */
  public static int readVarInt(ByteBuffer in) throws IOException {
    int result = 0;
    int shift = 0;
    for (int i = 0; i < 5; i++) {
      if (!in.hasRemaining()) {
        throw new IOException("truncated varint at buffer position " + in.position());
      }
      int b = in.get() & 0xFF;
      result |= (b & 0x7F) << shift;
      if ((b & 0x80) == 0) {
        return result;
      }
      shift += 7;
    }
    throw new IOException(
        "varint too long (more than 5 bytes) at buffer position " + in.position());
  }

  /**
   * Reads an unsigned varint directly from a {@link MemorySegment} at the given byte offset.
   * Returns the decoded value in the low 32 bits and the number of bytes consumed in the high 32
   * bits, packed into a single {@code long} to avoid allocating a pair object on the mmap read
   * path. Extract via {@code (int) result} for the value and {@code (int)(result >>> 32)} for the
   * byte count.
   */
  public static long readVarIntFromSegment(MemorySegment seg, long offset) throws IOException {
    int result = 0;
    int shift = 0;
    long limit = seg.byteSize();
    for (int i = 0; i < 5; i++) {
      if (offset + i >= limit) {
        throw new IOException("truncated varint at segment offset " + (offset + i));
      }
      int b = seg.get(ValueLayout.JAVA_BYTE, offset + i) & 0xFF;
      result |= (b & 0x7F) << shift;
      if ((b & 0x80) == 0) {
        return ((long) (i + 1) << 32) | (result & 0xFFFFFFFFL);
      }
      shift += 7;
    }
    throw new IOException("varint too long (more than 5 bytes) at segment offset " + offset);
  }

  /**
   * Returns the number of bytes a varint encoding of {@code value} would consume. Useful when
   * sizing a buffer before writing.
   */
  public static int varIntSize(int value) {
    if (value < 0) {
      throw new IllegalArgumentException("varint value must be non-negative: " + value);
    }
    int size = 1;
    while ((value & ~0x7F) != 0) {
      value >>>= 7;
      size++;
    }
    return size;
  }

  // ---------------------------------------------------------------------------
  // UTF-8 helpers — used to size and emit length-prefixed strings.
  // ---------------------------------------------------------------------------

  /**
   * Encodes {@code s} to a UTF-8 byte array. {@code null} is rejected — callers must validate
   * non-null ids and non-null metadata values upstream at the collection boundary so a null here is
   * a programmer error, not a valid on-disk state.
   */
  public static byte[] utf8(String s) {
    java.util.Objects.requireNonNull(s, "s must not be null");
    return s.getBytes(StandardCharsets.UTF_8);
  }

  /** Decodes a UTF-8 slice from {@code bytes} starting at {@code offset}. */
  public static String fromUtf8(byte[] bytes, int offset, int length) {
    return new String(bytes, offset, length, StandardCharsets.UTF_8);
  }
}
