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
package com.integrallis.vectors.quantization;

/**
 * Binary quantization modes.
 *
 * <ul>
 *   <li>{@link #SIGN_BIT} — Simple sign-bit encoding with Hamming distance scoring. 32x
 *       compression. Fast but lower recall (~0.70). Best for pre-normalized embeddings.
 *   <li>{@link #BBQ} — Better Binary Quantization: centroid-relative encoding with per-vector
 *       correction factors and asymmetric int4-query x 1-bit-stored scoring. Raw bit codes are 32x
 *       compressed; 12 bytes of correction floats reduce the effective ratio (e.g. ~18x for
 *       dim=128, ~28x for dim=768). Higher recall (~0.90+). Derived from RaBitQ / Lucene BBQ.
 * </ul>
 */
public enum BinaryMode {

  /** Sign-bit encoding: {@code bit[d] = (v[d] >= 0) ? 1 : 0}. Zero maps to 1. Hamming scoring. */
  SIGN_BIT,

  /** Centroid-relative encoding with asymmetric scoring and per-vector corrections. */
  BBQ
}
