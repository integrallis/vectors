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

import com.integrallis.vectors.hnsw.RandomAccessVectors;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Objects;

/**
 * Build-safe, mmap-backed {@link RandomAccessVectors} for constructing an index whose vectors are
 * too large to hold on the heap (e.g. 100M × 768d ≈ 307 GB). Unlike {@link
 * MemorySegmentRandomAccessVectors} — which returns a single shared per-thread scratch buffer and
 * is therefore unusable during graph construction (the builder holds one vector while fetching
 * others, violating the scratch invariant) — this returns a <b>fresh array per call</b> ({@link
 * #sharesReturnBuffer()} is {@code false}). That makes it safe to pass to {@code
 * ConcurrentHnswGraphBuilder} and, via {@code RandomAccessVectorDataset}, to the Ext-RaBitQ
 * quantizer.
 *
 * <p><b>Bounded-RAM build.</b> The full float32 vectors stay memory-mapped from {@code vectors.bin}
 * (the OS pages them in/out; not counted against heap), so a build holds only the graph adjacency
 * (~26 GB @ M=32, 100M) plus the accumulating codes (~50 GB @ 4-bit, 768d) in RAM — a ~128 GB box
 * builds 100M, versus the ~384 GB an in-heap {@code float[][]} build would need. Fresh-array
 * allocation per {@code getVector} is a non-issue: build is one-time and dominated by graph
 * construction, not allocation.
 *
 * <p>Segment scoring is supported: {@link #vectorSegment(int)} returns a stable zero-copy slice
 * into the immutable mmap (distinct ordinals yield distinct, independently valid views for the
 * arena's lifetime), so the builder's fused SIMD path can score straight from the mapped page.
 */
public final class MappedBuildVectors implements RandomAccessVectors {

  private final MemorySegmentVectors mapped;
  private final int dimension;

  /**
   * @param mapped the memory-mapped {@code vectors.bin} (see {@link MemorySegmentVectors#open})
   */
  public MappedBuildVectors(MemorySegmentVectors mapped) {
    this.mapped = Objects.requireNonNull(mapped, "mapped");
    this.dimension = mapped.dimension();
  }

  @Override
  public int size() {
    return mapped.size();
  }

  @Override
  public int dimension() {
    return dimension;
  }

  /**
   * Returns a FRESH {@code float[dimension]} copied from the mmap — safe to retain across calls.
   */
  @Override
  public float[] getVector(int ordinal) {
    float[] dst = new float[dimension];
    MemorySegment.copy(mapped.vectorSlice(ordinal), ValueLayout.JAVA_FLOAT, 0L, dst, 0, dimension);
    return dst;
  }

  /** Fresh array per call — never a shared buffer. This is what makes the mapper build-safe. */
  @Override
  public boolean sharesReturnBuffer() {
    return false;
  }

  /** mmap slices are stable and zero-copy — enable the builder's segment-scoring fast path. */
  @Override
  public boolean supportsSegments() {
    return true;
  }

  @Override
  public MemorySegment vectorSegment(int ordinal) {
    return mapped.vectorSlice(ordinal);
  }
}
