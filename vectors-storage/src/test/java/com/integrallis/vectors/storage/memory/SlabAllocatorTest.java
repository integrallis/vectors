package com.integrallis.vectors.storage.memory;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Tests for {@link SlabAllocator}. */
@Tag("unit")
class SlabAllocatorTest {

  @Test
  void allocate_returnsDistinctSegments() {
    try (Arena arena = Arena.ofConfined()) {
      SlabAllocator slab = SlabAllocator.create(arena, 1024, AlignmentUtil.VECTOR_ALIGNMENT);

      MemorySegment a = slab.allocate(64);
      MemorySegment b = slab.allocate(64);

      // Segments should not overlap
      assertThat(a.address()).isNotEqualTo(b.address());
      assertThat(a.byteSize()).isEqualTo(64);
      assertThat(b.byteSize()).isEqualTo(64);
    }
  }

  @Test
  void allocate_segmentsAreWritable() {
    try (Arena arena = Arena.ofConfined()) {
      SlabAllocator slab = SlabAllocator.create(arena, 256, AlignmentUtil.VECTOR_ALIGNMENT);

      MemorySegment seg = slab.allocate(16);
      seg.set(ValueLayout.JAVA_INT, 0, 42);
      assertThat(seg.get(ValueLayout.JAVA_INT, 0)).isEqualTo(42);
    }
  }

  @Test
  void capacity_matchesCreation() {
    try (Arena arena = Arena.ofConfined()) {
      SlabAllocator slab = SlabAllocator.create(arena, 4096, AlignmentUtil.VECTOR_ALIGNMENT);
      assertThat(slab.capacity()).isEqualTo(4096);
    }
  }

  @Test
  void allocated_tracksUsage() {
    try (Arena arena = Arena.ofConfined()) {
      SlabAllocator slab = SlabAllocator.create(arena, 4096, AlignmentUtil.VECTOR_ALIGNMENT);
      assertThat(slab.allocated()).isZero();

      slab.allocate(100);
      assertThat(slab.allocated()).isEqualTo(100);

      slab.allocate(200);
      assertThat(slab.allocated()).isEqualTo(300);
    }
  }

  @Test
  void overflow_throwsWhenExhausted() {
    try (Arena arena = Arena.ofConfined()) {
      SlabAllocator slab = SlabAllocator.create(arena, 128, AlignmentUtil.VECTOR_ALIGNMENT);
      slab.allocate(100);
      // The slicing allocator should throw when the slab is exhausted
      assertThatThrownBy(() -> slab.allocate(100)).isInstanceOf(IndexOutOfBoundsException.class);
    }
  }

  @Test
  void manySmallAllocations() {
    try (Arena arena = Arena.ofConfined()) {
      int count = 100;
      int size = 16;
      SlabAllocator slab =
          SlabAllocator.create(arena, (long) count * size * 2, AlignmentUtil.VECTOR_ALIGNMENT);

      MemorySegment[] segments = new MemorySegment[count];
      for (int i = 0; i < count; i++) {
        segments[i] = slab.allocate(size);
        // Write a unique value
        segments[i].set(ValueLayout.JAVA_INT, 0, i);
      }

      // Verify all values are preserved
      for (int i = 0; i < count; i++) {
        assertThat(segments[i].get(ValueLayout.JAVA_INT, 0)).isEqualTo(i);
      }
    }
  }

  @Test
  void slab_returnsBackingSegment() {
    try (Arena arena = Arena.ofConfined()) {
      SlabAllocator slab = SlabAllocator.create(arena, 4096, AlignmentUtil.VECTOR_ALIGNMENT);
      MemorySegment backing = slab.slab();
      assertThat(backing.byteSize()).isEqualTo(4096);
    }
  }
}
