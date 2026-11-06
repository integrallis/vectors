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
package com.integrallis.vectors.core;

import java.lang.foreign.MemorySegment;

/**
 * Off-heap float-vector distance and similarity kernels. Operates on vectors stored in {@link
 * MemorySegment}s, enabling zero-copy access to native or memory-mapped data.
 */
public interface SegmentDistanceKernel {

  /** Computes the dot product from off-heap float vectors stored in MemorySegments. */
  float dotProduct(MemorySegment a, MemorySegment b, int dimensions);

  /** Computes the squared L2 distance from off-heap float vectors stored in MemorySegments. */
  float squareDistance(MemorySegment a, MemorySegment b, int dimensions);

  /**
   * Computes the cosine similarity from off-heap float vectors stored in MemorySegments. Returns
   * value in [-1, 1].
   */
  float cosine(MemorySegment a, MemorySegment b, int dimensions);

  /** Segment cosine with the norm of the first vector prepared by this provider. */
  default float cosineWithQueryNorm(
      MemorySegment a, MemorySegment b, int dimensions, float queryNorm) {
    return cosine(a, b, dimensions);
  }
}
