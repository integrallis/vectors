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

/** Register-tile strategy used by GGUF Q6_K batched matrix multiplication. */
public enum GgufQ6BatchedKernel {
  /** Computes one query at a time to minimize live vector state and allocation pressure. */
  ONE_QUERY_BLOCK,

  /** Reuses each unpacked weight vector across two queries to maximize prefill throughput. */
  TWO_QUERY_BLOCK
}
