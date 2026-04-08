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
