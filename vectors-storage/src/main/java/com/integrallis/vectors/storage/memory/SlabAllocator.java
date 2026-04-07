package com.integrallis.vectors.storage.memory;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SegmentAllocator;

/**
 * Bump allocator over a pre-allocated slab of off-heap memory. Allocations are O(1) (no syscall per
 * allocation); all memory is freed when the backing arena is closed.
 *
 * <p>Uses {@link SegmentAllocator#slicingAllocator(MemorySegment)} internally for efficient
 * sub-segment allocation.
 *
 * <p>Usage:
 *
 * <pre>{@code
 * try (Arena arena = Arena.ofConfined()) {
 *     var slab = SlabAllocator.create(arena, 1024 * 1024, AlignmentUtil.VECTOR_ALIGNMENT);
 *     MemorySegment vec1 = slab.allocate(128 * 4);
 *     MemorySegment vec2 = slab.allocate(128 * 4);
 *     // ... all freed when arena.close()
 * }
 * }</pre>
 */
public final class SlabAllocator {

  private final MemorySegment slab;
  private final SegmentAllocator allocator;
  private final long capacity;
  private long allocated;

  private SlabAllocator(MemorySegment slab, SegmentAllocator allocator, long capacity) {
    this.slab = slab;
    this.allocator = allocator;
    this.capacity = capacity;
    this.allocated = 0;
  }

  /**
   * Creates a new slab allocator with the given capacity and alignment.
   *
   * @param arena the arena that owns the slab's lifetime
   * @param capacityBytes total slab size in bytes
   * @param alignment alignment for the slab itself (e.g., 64 for SIMD)
   * @return a new slab allocator
   */
  public static SlabAllocator create(Arena arena, long capacityBytes, int alignment) {
    MemorySegment slab = arena.allocate(capacityBytes, alignment);
    SegmentAllocator slicing = SegmentAllocator.slicingAllocator(slab);
    return new SlabAllocator(slab, slicing, capacityBytes);
  }

  /**
   * Allocates a sub-segment of the given size from the slab.
   *
   * @param byteSize the number of bytes to allocate
   * @return a segment within the slab
   * @throws IndexOutOfBoundsException if the slab is exhausted
   */
  public MemorySegment allocate(long byteSize) {
    MemorySegment result = allocator.allocate(byteSize);
    allocated += byteSize;
    return result;
  }

  /**
   * Allocates a sub-segment with the given size and alignment.
   *
   * @param byteSize the number of bytes to allocate
   * @param alignment the alignment for the allocation
   * @return a segment within the slab
   * @throws IndexOutOfBoundsException if the slab is exhausted
   */
  public MemorySegment allocate(long byteSize, long alignment) {
    MemorySegment result = allocator.allocate(byteSize, alignment);
    allocated += byteSize;
    return result;
  }

  /** Returns the total capacity of the slab in bytes. */
  public long capacity() {
    return capacity;
  }

  /** Returns an approximation of the number of bytes allocated so far. */
  public long allocated() {
    return allocated;
  }

  /** Returns the backing slab segment. */
  public MemorySegment slab() {
    return slab;
  }
}
