package com.integrallis.vectors.db;

/**
 * Quantizer backend selector. Step 2 only wires {@link #NONE}; other values are reserved for
 * subsequent steps.
 */
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
  NVQ
}
