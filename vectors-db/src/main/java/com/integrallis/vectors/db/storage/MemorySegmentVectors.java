package com.integrallis.vectors.db.storage;

import com.integrallis.vectors.storage.memory.AlignmentUtil;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Objects;

/**
 * Thread-safe zero-copy {@code float32} random-access vector adapter backed by a memory-mapped
 * file. This is the mmap-backed analogue of {@code MappedVectorStore} — it reads the same on-disk
 * layout (raw packed {@code float32} values with 64-byte alignment per vector) but differs in one
 * important way: lifetime is owned by a <b>caller-provided</b> {@link Arena}.
 *
 * <p>This distinction matters because a Step 4a {@code Generation} shares a single {@link
 * Arena#ofShared()} across every mmap'd file in one generation — vectors, idmap, metadata, manifest
 * — so that retiring a generation unmaps all four files atomically in one operation. If we instead
 * used {@code MappedVectorStore.open(...)} here, each file would own its own arena and lifecycle
 * management would be tangled across four close() calls.
 *
 * <p><b>File layout.</b> No header. Vectors are packed starting at byte offset 0, one per {@code
 * stride} bytes, where {@code stride = alignUp(dimension * 4, 64)}. For {@code dim=128} this is
 * exactly {@code 512} bytes per vector (no padding); for {@code dim=100} it is {@code alignUp(400,
 * 64) = 448} bytes (48 bytes of zero padding). This matches {@link
 * com.integrallis.vectors.storage.store.MappedVectorStore} byte-for-byte, so a {@code vectors.bin}
 * written by {@code VectorStoreWriter} opens cleanly here.
 *
 * <p><b>Zero-copy SIMD.</b> The headline feature: {@link #vectorSlice(int)} returns a {@code
 * MemorySegment} view into the mmap'd page, with no intermediate {@code float[]} copy. That slice
 * can be fed directly to {@code VectorUtil.dotProduct(MemorySegment, MemorySegment, int)} (and
 * friends), which in turn call {@code FloatVector.fromMemorySegment(SPECIES, slice, 0, LE)} — a
 * HotSpot intrinsic that becomes a single {@code vmovups} / {@code vmovaps} load on x86
 * AVX2/AVX-512 or an {@code ld1} on ARM NEON/SVE. Disk page cache → L1 → SIMD register with zero
 * float copies.
 *
 * <p>The 64-byte alignment guarantees that every slice begins on a cache-line and SIMD-register
 * boundary, so no peeled scalar head loop is ever required on the score path.
 *
 * <p><b>Byte order.</b> Floats on disk are little-endian, matching {@code VectorStoreWriter}'s
 * output and the hardcoded {@code ByteOrder.LITTLE_ENDIAN} in {@code PanamaVectorUtilSupport}. This
 * is a deliberate cross-platform file compatibility choice (free on both x86 and ARM, which are
 * native LE).
 *
 * <p><b>Thread safety.</b> All instance state is {@code final} and the underlying {@code
 * MemorySegment} is safe for concurrent reads from any number of threads — that's the whole point
 * of {@link Arena#ofShared()}. Two threads may call {@link #vectorSlice(int)} on different (or the
 * same) ordinals concurrently without any synchronization.
 *
 * <p><b>Close semantics.</b> {@link #close()} is a no-op — the caller-provided {@link Arena} owns
 * the mmap lifetime, and the owning {@code Generation} closes the arena exactly once when its
 * refcount drops to zero. Double-closing the arena (e.g., if each mapped store closed it
 * independently) would throw {@code IllegalStateException}.
 */
public final class MemorySegmentVectors implements AutoCloseable {

  private final MemorySegment segment;
  private final int size;
  private final int dimension;
  private final long stride;
  private final long dataOffset;
  private final int rawVectorByteSize;

  private MemorySegmentVectors(
      MemorySegment segment, int size, int dimension, long stride, long dataOffset) {
    this.segment = segment;
    this.size = size;
    this.dimension = dimension;
    this.stride = stride;
    this.dataOffset = dataOffset;
    this.rawVectorByteSize = dimension * Float.BYTES;
  }

  /**
   * Opens a {@code vectors.bin} file and maps it into the caller-provided {@link Arena}. The {@code
   * arena} is not closed by this class — it is owned by the caller (typically a {@code Generation}
   * that shares the arena across vectors/idmap/metadata/manifest).
   *
   * <p>The file is expected to be exactly the layout written by {@code VectorStoreWriter}: packed
   * float32 values, each padded up to {@code 64}-byte alignment. The file size must be at least
   * {@code size * alignUp(dimension * 4, 64)} bytes — trailing bytes (if any) are permitted but
   * ignored.
   *
   * @param file path to {@code vectors.bin}
   * @param size number of vectors (must match the vector count recorded in the manifest)
   * @param dimension number of float32 components per vector
   * @param arena caller-owned arena; its lifetime must outlive this instance
   * @return a new {@code MemorySegmentVectors} bound to the given arena
   * @throws IOException if the file cannot be opened or is shorter than expected
   * @throws NullPointerException if {@code file} or {@code arena} is null
   * @throws IllegalArgumentException if {@code size < 0} or {@code dimension < 1}
   */
  public static MemorySegmentVectors open(Path file, int size, int dimension, Arena arena)
      throws IOException {
    Objects.requireNonNull(file, "file must not be null");
    Objects.requireNonNull(arena, "arena must not be null");
    if (size < 0) {
      throw new IllegalArgumentException("size must be >= 0: " + size);
    }
    if (dimension < 1) {
      throw new IllegalArgumentException("dimension must be >= 1: " + dimension);
    }

    long stride =
        AlignmentUtil.alignUp((long) dimension * Float.BYTES, AlignmentUtil.VECTOR_ALIGNMENT);
    long expectedSize = stride * size;

    try (FileChannel ch = FileChannel.open(file, StandardOpenOption.READ)) {
      long fileSize = ch.size();
      if (fileSize < expectedSize) {
        throw new IOException(
            "vectors.bin truncated: expected at least "
                + expectedSize
                + " bytes ("
                + size
                + " vectors × "
                + stride
                + " stride), got "
                + fileSize);
      }
      MemorySegment seg;
      if (fileSize == 0) {
        // Empty file — map() on a zero-length file behaves inconsistently across JDKs. Allocate a
        // zero-length segment inside the caller-provided arena so every instance field points at a
        // segment whose lifetime is tied to the arena (not to the shared sentinel
        // MemorySegment.NULL,
        // whose scope is the global scope and would never be unmapped).
        seg = arena.allocate(0L);
      } else {
        seg = ch.map(FileChannel.MapMode.READ_ONLY, 0, fileSize, arena);
      }
      return new MemorySegmentVectors(seg, size, dimension, stride, 0L);
    }
  }

  /** Returns the number of vectors stored. */
  public int size() {
    return size;
  }

  /** Returns the number of float32 components per vector. */
  public int dimension() {
    return dimension;
  }

  /**
   * Returns the byte stride between consecutive vectors (includes 64-byte alignment padding). For
   * {@code dim=128} this is {@code 512}; for {@code dim=100} it is {@code 448}.
   */
  public long stride() {
    return stride;
  }

  /**
   * Returns a zero-copy view of the vector at the given ordinal. The returned slice covers exactly
   * {@code dimension * 4} bytes — the raw vector data, without trailing alignment padding.
   *
   * <p>This slice is safe to pass to {@code VectorUtil.dotProduct(MemorySegment, MemorySegment,
   * int)}, {@code VectorUtil.squareDistance(MemorySegment, MemorySegment, int)}, or {@code
   * VectorUtil.cosine(MemorySegment, MemorySegment, int)} — all of which load from the slice via
   * {@code FloatVector.fromMemorySegment} with {@code ByteOrder.LITTLE_ENDIAN}.
   *
   * @param ordinal the vector ordinal in {@code [0, size())}
   * @return zero-copy {@code MemorySegment} view
   * @throws IndexOutOfBoundsException if {@code ordinal} is out of range
   */
  public MemorySegment vectorSlice(int ordinal) {
    checkOrdinal(ordinal);
    return segment.asSlice(dataOffset + (long) ordinal * stride, rawVectorByteSize);
  }

  /**
   * Returns the underlying whole-file {@link MemorySegment}. Intended for tight inner-loop scoring
   * paths (e.g. {@code MappedFlatScanAdapter}) that want to avoid the per-ordinal bounds check +
   * {@code asSlice} virtual call in {@link #vectorSlice(int)} and instead do {@code
   * segment.asSlice(vectorOffsetFor(i), rawBytes)} directly inside a hot loop.
   *
   * <p>Callers MUST NOT attempt to close this segment — its lifetime is owned by the
   * caller-provided {@link Arena} passed to {@link #open(Path, int, int, Arena)}.
   */
  public MemorySegment segment() {
    return segment;
  }

  /**
   * Returns the byte offset within {@link #segment()} where the given ordinal's vector begins.
   * Equivalent to {@code ordinal * stride()} for the default layout. Combine with {@link
   * #segment()} and {@link #rawVectorByteSize()} in a tight scoring loop.
   *
   * @param ordinal the vector ordinal in {@code [0, size())}
   * @throws IndexOutOfBoundsException if {@code ordinal} is out of range
   */
  public long vectorOffsetFor(int ordinal) {
    checkOrdinal(ordinal);
    return dataOffset + (long) ordinal * stride;
  }

  /**
   * Returns the raw (un-padded) byte size of a single vector — i.e. {@code dimension * 4} for
   * float32. This is the byte length of every slice returned by {@link #vectorSlice(int)} and the
   * correct value to pass as the second argument to {@code MemorySegment.asSlice(offset, byteSize)}
   * in a manual scoring loop.
   */
  public int rawVectorByteSize() {
    return rawVectorByteSize;
  }

  private void checkOrdinal(int ordinal) {
    if (ordinal < 0 || ordinal >= size) {
      throw new IndexOutOfBoundsException("ordinal " + ordinal + " out of range [0, " + size + ")");
    }
  }

  /**
   * No-op — the caller-provided {@link Arena} owns the mmap lifetime. The owning {@code Generation}
   * closes the arena exactly once when its refcount drops to zero.
   */
  @Override
  public void close() {
    // no-op
  }
}
