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
package com.integrallis.vectors.db;

/** Quantizer backend selector. */
public enum QuantizerKind {
  /** No quantization — full-precision float vectors. */
  NONE,

  /** Scalar quantization, 8-bit. */
  SQ8,

  /** Scalar quantization, 4-bit. */
  SQ4,

  /** Product quantization. */
  PQ,

  /** Binary quantization (sign-bit / BBQ). */
  BQ,

  /** RaBitQ rotation + sign-bit quantization. */
  RABITQ,

  /** Nonlinear per-vector quantization. */
  NVQ,

  /** TurboQuant: rotation + data-independent Lloyd-Max scalar quantization. */
  TURBOQUANT,

  /**
   * Half-precision (IEEE 754 binary16 / fp16) scalar storage — two bytes per coordinate, upcast to
   * {@code float32} for scoring. Near-lossless (recall ≈ full precision) at 2x compression.
   *
   * <p>Appended last so the existing ordinals stay stable on disk ({@code quantized.bin} stores the
   * ordinal).
   */
  FP16
}
