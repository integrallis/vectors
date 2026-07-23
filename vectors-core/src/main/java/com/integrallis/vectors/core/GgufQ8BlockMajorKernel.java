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

/** Output accumulation strategy for block-major GGUF Q8_0 batch kernels. */
public enum GgufQ8BlockMajorKernel {
  /** Updates the caller's batch-major output after every weight block. */
  SCATTERED,

  /** Accumulates one output row contiguously and scatters it once after all weight blocks. */
  ROW_ACCUMULATED,

  /**
   * Retains eight strided floating lanes across weight blocks and reduces once per batch output.
   * This non-associative order is deterministic but need not be bit-identical to scalar block
   * accumulation.
   */
  FLOAT_LANE_ACCUMULATED
}
