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

/** Arithmetic graph used by GGUF Q4_0 by Q8_0 kernels. */
public enum GgufQ4Kernel {
  /** Widens signed operands and multiplies them as integer lanes. */
  WIDENED,

  /** Uses signed-short pairwise multiply-add when the active vector shape supports it. */
  SHORT_PAIRWISE,

  /**
   * Uses unsigned-Q4/signed-Q8 byte pairwise multiply-add with Q8 zero-point corrections computed
   * once during activation quantization.
   */
  UNSIGNED_PAIRWISE
}
