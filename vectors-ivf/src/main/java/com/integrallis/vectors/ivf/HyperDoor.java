package com.integrallis.vectors.ivf;

/**
 * Cross-tier link record for a single cluster ordinal. All four tiers share the same ordinal space;
 * physical tier offsets are computed by O(1) stride arithmetic — no hash lookup, no pointer
 * chasing.
 *
 * <pre>
 *   T0: bitSet.get(t0BitOffset)
 *   T1: sq8Segment.asSlice(t1ByteOffset, D)
 *   T2: vectorsMmap.asSlice(t2FileOffset, D * Float.BYTES)
 *   T3: s3Backend.get(objectKey, t3ObjectOffset, D * Float.BYTES)
 * </pre>
 *
 * <p>In practice, for contiguous ordinals within a tier, the stride arithmetic degenerates to
 * {@code offset = ordinal * stride}, so HyperDoor records are not stored per-vector — they are
 * computed on the fly from the ordinal and the cluster's per-tier strides. The record exists as an
 * explicit documentation artifact and for non-contiguous layouts after compaction gaps.
 *
 * <p>Fields with value {@code -1} indicate the tier is not locally materialised. T0 ({@code
 * t0BitOffset}) and T3 ({@code t3ObjectOffset}) are always &ge; 0.
 *
 * @param clusterOrdinal position within this cluster's ordinal space (0-based)
 * @param t0BitOffset bit offset in the 1-bit RaBitQ bit-set (always present; &ge; 0)
 * @param t1ByteOffset byte offset in the SQ8 {@code MemorySegment} (-1 = not materialised)
 * @param t2FileOffset byte offset in the mmap'd {@code vectors.bin} (-1 = not loaded)
 * @param t3ObjectOffset byte offset in the S3 object key (always present; &ge; 0)
 */
public record HyperDoor(
    int clusterOrdinal,
    long t0BitOffset,
    int t1ByteOffset,
    long t2FileOffset,
    long t3ObjectOffset) {

  /** Constructs a fully-materialised HyperDoor where all tiers are available. */
  public static HyperDoor full(int ordinal, int dim) {
    return new HyperDoor(
        ordinal,
        (long) ordinal * dim, // T0: bit offset  (1 bit per dim — simplification: 1 bit per vec)
        ordinal * dim, // T1: byte offset into SQ8 segment (1 byte per dim)
        (long) ordinal * dim * Float.BYTES, // T2: float32 mmap offset
        (long) ordinal * dim * Float.BYTES // T3: S3 object byte offset
        );
  }

  /** Constructs a HyperDoor with only T0 and T3 available (T1 and T2 not materialised). */
  public static HyperDoor t0AndT3Only(int ordinal, int dim) {
    return new HyperDoor(
        ordinal, (long) ordinal * dim, -1, -1L, (long) ordinal * dim * Float.BYTES);
  }

  /** Returns {@code true} if the SQ8 (T1) tier is locally materialised. */
  public boolean hasT1() {
    return t1ByteOffset >= 0;
  }

  /** Returns {@code true} if the float32 mmap (T2) tier is locally materialised. */
  public boolean hasT2() {
    return t2FileOffset >= 0;
  }
}
