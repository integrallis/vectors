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
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Objects;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.LongVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorShuffle;
import jdk.incubator.vector.VectorSpecies;

/**
 * Panama Vector API SIMD implementation of {@link VectorUtilSupport}. Uses FMA-based accumulation
 * with multiple independent accumulators to saturate CPU execution ports.
 *
 * <p>Key patterns (synthesized from Lucene 10.x and JVector):
 *
 * <ul>
 *   <li>4x unrolling for dot product and L2 (1 FMA per iteration)
 *   <li>4x unrolling for array cosine and MemorySegment cosine
 *   <li>Conditional FMA dispatch via {@link PanamaConstants#HAS_FAST_VECTOR_FMA}
 *   <li>{@code SPECIES_PREFERRED} for main loop with scalar tail
 *   <li>Byte operations use widening conversions (B2S, S2I) and fixed-width tiers to avoid overflow
 * </ul>
 */
final class PanamaVectorUtilSupport implements VectorUtilSupport {

  private static final int Q4_SHORT_PAIRWISE_MIN_BLOCKS = 32;

  // Species are capped to PanamaConstants.MAX_BITS (default 256) to avoid AVX-512 frequency
  // downclock; opt into wider with -Dvectors.maxBits=512. The cap only ever narrows below the
  // hardware-preferred width, so every loop below stays correct (they are written generically
  // against FLOAT_SPECIES.length()/loopBound). Note the byte kernels use their own fixed 256-bit
  // tiers (ByteVector.SPECIES_64 -> IntVector.SPECIES_256) independent of these constants.
  static final VectorSpecies<Float> FLOAT_SPECIES =
      PanamaConstants.preferredSpecies(FloatVector.SPECIES_PREFERRED);
  static final VectorSpecies<Byte> BYTE_SPECIES =
      PanamaConstants.preferredSpecies(ByteVector.SPECIES_PREFERRED);
  static final VectorSpecies<Short> SHORT_SPECIES =
      PanamaConstants.preferredSpecies(ShortVector.SPECIES_PREFERRED);
  static final VectorSpecies<Integer> INT_SPECIES =
      PanamaConstants.preferredSpecies(IntVector.SPECIES_PREFERRED);
  static final VectorSpecies<Long> LONG_SPECIES =
      PanamaConstants.preferredSpecies(LongVector.SPECIES_PREFERRED);
  static final int VECTOR_BITSIZE = FLOAT_SPECIES.vectorBitSize();
  private static final VectorShuffle<Short> SWAP_ADJACENT_SHORTS =
      VectorShuffle.fromValues(
          ShortVector.SPECIES_256, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);
  private static final VectorShuffle<Short> SWAP_SHORT_PAIRS =
      VectorShuffle.fromValues(
          ShortVector.SPECIES_256, 2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13);
  private static final VectorShuffle<Short> SELECT_LOW_GROUPS =
      VectorShuffle.fromValues(
          ShortVector.SPECIES_256, 0, 4, 8, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
  private static final VectorShuffle<Short> SELECT_HIGH_GROUPS =
      VectorShuffle.fromValues(
          ShortVector.SPECIES_256, 0, 0, 0, 0, 0, 4, 8, 12, 0, 0, 0, 0, 0, 0, 0, 0);
  private static final VectorMask<Short> HIGH_GROUP_LANES =
      VectorMask.fromLong(ShortVector.SPECIES_256, 0xF0L);
  private static final VectorShuffle<Integer> SWAP_ADJACENT_INTS =
      VectorShuffle.fromValues(IntVector.SPECIES_256, 1, 0, 3, 2, 5, 4, 7, 6);
  private static final VectorShuffle<Integer> SELECT_LOW_INT_GROUPS =
      VectorShuffle.fromValues(IntVector.SPECIES_256, 0, 2, 4, 6, 0, 0, 0, 0);
  private static final VectorShuffle<Integer> SELECT_HIGH_INT_GROUPS =
      VectorShuffle.fromValues(IntVector.SPECIES_256, 0, 0, 0, 0, 0, 2, 4, 6);
  private static final VectorMask<Integer> HIGH_INT_GROUP_LANES =
      VectorMask.fromLong(IntVector.SPECIES_256, 0xF0L);
  private static final VectorShuffle<Short> SHORT_256_EVEN_LANES =
      VectorShuffle.makeUnzip(ShortVector.SPECIES_256, 0);
  private static final VectorShuffle<Short> SHORT_256_ODD_LANES =
      VectorShuffle.makeUnzip(ShortVector.SPECIES_256, 1);
  private static final VectorShuffle<Byte> BYTE_128_EVEN_LANES =
      VectorShuffle.makeUnzip(ByteVector.SPECIES_128, 0);
  private static final VectorShuffle<Byte> BYTE_128_ODD_LANES =
      VectorShuffle.makeUnzip(ByteVector.SPECIES_128, 1);
  private static final VectorShuffle<Short> SHORT_128_EVEN_LANES =
      VectorShuffle.makeUnzip(ShortVector.SPECIES_128, 0);
  private static final VectorShuffle<Short> SHORT_128_ODD_LANES =
      VectorShuffle.makeUnzip(ShortVector.SPECIES_128, 1);
  private static final short[] Q4_PAIR_FACTORS = {1, 1, 1, 1, 1, 1, 1, 1};
  private static final ByteVector Q5_HIGH_BIT_MASKS =
      ByteVector.fromArray(
          ByteVector.SPECIES_64, new byte[] {1, 2, 4, 8, 16, 32, 64, (byte) 0x80}, 0);

  // --- Conditional FMA helpers ---

  static FloatVector fma(FloatVector a, FloatVector b, FloatVector c) {
    if (PanamaConstants.HAS_FAST_VECTOR_FMA) {
      return a.fma(b, c);
    } else {
      return a.mul(b).add(c);
    }
  }

  static boolean useQ4ShortPairwise(GgufQ4Kernel kernel, int vectorBits, int blocks) {
    boolean requested =
        switch (Objects.requireNonNull(kernel, "kernel")) {
          case WIDENED, UNSIGNED_PAIRWISE -> false;
          case SHORT_PAIRWISE -> true;
        };
    return requested && vectorBits >= 256 && blocks >= Q4_SHORT_PAIRWISE_MIN_BLOCKS;
  }

  static ByteVector q4HighNibbles(ByteVector packed) {
    // Logical shift zero-fills each byte lane, so the result is already in [0, 15].
    return packed.lanewise(VectorOperators.LSHR, 4);
  }

  private static boolean useQ4ShortPairwise(GgufQ4Kernel kernel, int blocks) {
    return useQ4ShortPairwise(kernel, VECTOR_BITSIZE, blocks);
  }

  static boolean useQ4UnsignedPairwise(GgufQ4Kernel kernel, int vectorBits, int blocks) {
    return Objects.requireNonNull(kernel, "kernel") == GgufQ4Kernel.UNSIGNED_PAIRWISE
        && vectorBits >= 256
        && blocks >= Q4_SHORT_PAIRWISE_MIN_BLOCKS;
  }

  private static boolean useQ4UnsignedPairwise(GgufQ4Kernel kernel, int blocks) {
    return useQ4UnsignedPairwise(kernel, VECTOR_BITSIZE, blocks);
  }

  private static void quantizeQ8_0ForQ4(
      float[] query,
      int queryOffset,
      int dimensions,
      byte[] q8Quants,
      int quantOffset,
      float[] q8Scales,
      int scaleOffset,
      int[] q8ZeroPointCorrections,
      int correctionOffset,
      boolean computeZeroPointCorrections) {
    if (computeZeroPointCorrections) {
      GgufQuantizationSupport.quantizeQ8_0WithQ4Corrections(
          query,
          queryOffset,
          dimensions,
          q8Quants,
          quantOffset,
          q8Scales,
          scaleOffset,
          q8ZeroPointCorrections,
          correctionOffset);
      return;
    }
    GgufQuantizationSupport.quantizeQ8_0(
        query, queryOffset, dimensions, q8Quants, quantOffset, q8Scales, scaleOffset);
  }

  private static void quantizeQ8_0ForBatchedQ4(
      float[] query,
      int queryOffset,
      int dimensions,
      byte[] q8Quants,
      int quantOffset,
      float[] q8Scales,
      int scaleOffset,
      int[] q8ZeroPointCorrections,
      int correctionOffset,
      boolean computeZeroPointCorrections) {
    if (computeZeroPointCorrections) {
      GgufQuantizationSupport.quantizeQ8_0WithCombinedQ4Corrections(
          query,
          queryOffset,
          dimensions,
          q8Quants,
          quantOffset,
          q8Scales,
          scaleOffset,
          q8ZeroPointCorrections,
          correctionOffset);
      return;
    }
    GgufQuantizationSupport.quantizeQ8_0(
        query, queryOffset, dimensions, q8Quants, quantOffset, q8Scales, scaleOffset);
  }

  // --- Float dot product: 4x unrolled FMA (Lucene pattern) ---

  @Override
  public float dotProduct(float[] a, float[] b) {
    return dotProduct(a, 0, b, 0, a.length);
  }

  @Override
  public float dotProduct(float[] a, int aOffset, float[] b, int bOffset, int length) {
    int i = 0;
    float res = 0f;

    // Only vectorize if worth the overhead
    if (length > 2 * FLOAT_SPECIES.length()) {
      int limit = FLOAT_SPECIES.loopBound(length);
      res += dotProductBody(a, aOffset, b, bOffset, limit);
      i += limit;
    }

    // Scalar tail
    for (; i < length; i++) {
      res = MathUtil.fma(a[aOffset + i], b[bOffset + i], res);
    }
    return res;
  }

  private float dotProductBody(float[] a, int aOffset, float[] b, int bOffset, int limit) {
    FloatVector acc1 = FloatVector.zero(FLOAT_SPECIES);
    FloatVector acc2 = FloatVector.zero(FLOAT_SPECIES);
    FloatVector acc3 = FloatVector.zero(FLOAT_SPECIES);
    FloatVector acc4 = FloatVector.zero(FLOAT_SPECIES);
    int unrolledLimit = limit - 3 * FLOAT_SPECIES.length();
    int i = 0;

    // Main loop: 4x unrolled for ILP
    for (; i < unrolledLimit; i += 4 * FLOAT_SPECIES.length()) {
      FloatVector va1 = FloatVector.fromArray(FLOAT_SPECIES, a, aOffset + i);
      FloatVector vb1 = FloatVector.fromArray(FLOAT_SPECIES, b, bOffset + i);
      acc1 = fma(va1, vb1, acc1);

      FloatVector va2 =
          FloatVector.fromArray(FLOAT_SPECIES, a, aOffset + i + FLOAT_SPECIES.length());
      FloatVector vb2 =
          FloatVector.fromArray(FLOAT_SPECIES, b, bOffset + i + FLOAT_SPECIES.length());
      acc2 = fma(va2, vb2, acc2);

      FloatVector va3 =
          FloatVector.fromArray(FLOAT_SPECIES, a, aOffset + i + 2 * FLOAT_SPECIES.length());
      FloatVector vb3 =
          FloatVector.fromArray(FLOAT_SPECIES, b, bOffset + i + 2 * FLOAT_SPECIES.length());
      acc3 = fma(va3, vb3, acc3);

      FloatVector va4 =
          FloatVector.fromArray(FLOAT_SPECIES, a, aOffset + i + 3 * FLOAT_SPECIES.length());
      FloatVector vb4 =
          FloatVector.fromArray(FLOAT_SPECIES, b, bOffset + i + 3 * FLOAT_SPECIES.length());
      acc4 = fma(va4, vb4, acc4);
    }

    // Vector tail: remaining full vectors
    for (; i < limit; i += FLOAT_SPECIES.length()) {
      FloatVector va = FloatVector.fromArray(FLOAT_SPECIES, a, aOffset + i);
      FloatVector vb = FloatVector.fromArray(FLOAT_SPECIES, b, bOffset + i);
      acc1 = fma(va, vb, acc1);
    }

    // Reduce 4 accumulators to scalar
    FloatVector res1 = acc1.add(acc2);
    FloatVector res2 = acc3.add(acc4);
    return reduceAdd(res1.add(res2));
  }

  // --- Float square distance (L2): 4x unrolled sub+FMA ---

  @Override
  public float squareDistance(float[] a, float[] b) {
    return squareDistance(a, 0, b, 0, a.length);
  }

  @Override
  public float squareDistance(float[] a, int aOffset, float[] b, int bOffset, int length) {
    int i = 0;
    float res = 0f;

    if (length > 2 * FLOAT_SPECIES.length()) {
      int limit = FLOAT_SPECIES.loopBound(length);
      res += squareDistanceBody(a, aOffset, b, bOffset, limit);
      i += limit;
    }

    // Scalar tail
    for (; i < length; i++) {
      float diff = a[aOffset + i] - b[bOffset + i];
      res += diff * diff;
    }
    return res;
  }

  private float squareDistanceBody(float[] a, int aOffset, float[] b, int bOffset, int limit) {
    FloatVector acc1 = FloatVector.zero(FLOAT_SPECIES);
    FloatVector acc2 = FloatVector.zero(FLOAT_SPECIES);
    FloatVector acc3 = FloatVector.zero(FLOAT_SPECIES);
    FloatVector acc4 = FloatVector.zero(FLOAT_SPECIES);
    int unrolledLimit = limit - 3 * FLOAT_SPECIES.length();
    int i = 0;

    for (; i < unrolledLimit; i += 4 * FLOAT_SPECIES.length()) {
      FloatVector diff1 =
          FloatVector.fromArray(FLOAT_SPECIES, a, aOffset + i)
              .sub(FloatVector.fromArray(FLOAT_SPECIES, b, bOffset + i));
      acc1 = fma(diff1, diff1, acc1);

      FloatVector diff2 =
          FloatVector.fromArray(FLOAT_SPECIES, a, aOffset + i + FLOAT_SPECIES.length())
              .sub(FloatVector.fromArray(FLOAT_SPECIES, b, bOffset + i + FLOAT_SPECIES.length()));
      acc2 = fma(diff2, diff2, acc2);

      FloatVector diff3 =
          FloatVector.fromArray(FLOAT_SPECIES, a, aOffset + i + 2 * FLOAT_SPECIES.length())
              .sub(
                  FloatVector.fromArray(
                      FLOAT_SPECIES, b, bOffset + i + 2 * FLOAT_SPECIES.length()));
      acc3 = fma(diff3, diff3, acc3);

      FloatVector diff4 =
          FloatVector.fromArray(FLOAT_SPECIES, a, aOffset + i + 3 * FLOAT_SPECIES.length())
              .sub(
                  FloatVector.fromArray(
                      FLOAT_SPECIES, b, bOffset + i + 3 * FLOAT_SPECIES.length()));
      acc4 = fma(diff4, diff4, acc4);
    }

    for (; i < limit; i += FLOAT_SPECIES.length()) {
      FloatVector diff =
          FloatVector.fromArray(FLOAT_SPECIES, a, aOffset + i)
              .sub(FloatVector.fromArray(FLOAT_SPECIES, b, bOffset + i));
      acc1 = fma(diff, diff, acc1);
    }

    FloatVector res1 = acc1.add(acc2);
    FloatVector res2 = acc3.add(acc4);
    return res1.add(res2).reduceLanes(VectorOperators.ADD);
  }

  // --- Float cosine similarity: 4x unrolled, 3 FMAs per iteration ---

  @Override
  public float cosine(float[] a, float[] b) {
    int i = 0;
    float sum = 0f;
    float norm1 = 0f;
    float norm2 = 0f;

    if (a.length >= 4 * FLOAT_SPECIES.length()) {
      // 4x unrolled main body: 12 independent FMA accumulators hide FMA latency on NEON/AVX.
      int limit = FLOAT_SPECIES.loopBound(a.length);
      float[] result = cosineBody4x(a, b, limit);
      sum = result[0];
      norm1 = result[1];
      norm2 = result[2];
      i = limit;
    } else if (a.length >= FLOAT_SPECIES.length()) {
      // Short vectors: single-accumulator vector body (no unroll), avoids unroll-prologue cost.
      int limit = FLOAT_SPECIES.loopBound(a.length);
      float[] result = cosineBody1x(a, b, limit);
      sum = result[0];
      norm1 = result[1];
      norm2 = result[2];
      i = limit;
    }

    // Scalar tail
    for (; i < a.length; i++) {
      sum = MathUtil.fma(a[i], b[i], sum);
      norm1 = MathUtil.fma(a[i], a[i], norm1);
      norm2 = MathUtil.fma(b[i], b[i], norm2);
    }

    return (float) (sum / Math.sqrt((double) norm1 * (double) norm2));
  }

  private float[] cosineBody4x(float[] a, float[] b, int limit) {
    FloatVector s0 = FloatVector.zero(FLOAT_SPECIES);
    FloatVector s1 = FloatVector.zero(FLOAT_SPECIES);
    FloatVector s2 = FloatVector.zero(FLOAT_SPECIES);
    FloatVector s3 = FloatVector.zero(FLOAT_SPECIES);
    FloatVector n1_0 = FloatVector.zero(FLOAT_SPECIES);
    FloatVector n1_1 = FloatVector.zero(FLOAT_SPECIES);
    FloatVector n1_2 = FloatVector.zero(FLOAT_SPECIES);
    FloatVector n1_3 = FloatVector.zero(FLOAT_SPECIES);
    FloatVector n2_0 = FloatVector.zero(FLOAT_SPECIES);
    FloatVector n2_1 = FloatVector.zero(FLOAT_SPECIES);
    FloatVector n2_2 = FloatVector.zero(FLOAT_SPECIES);
    FloatVector n2_3 = FloatVector.zero(FLOAT_SPECIES);
    int i = 0;
    int lanes = FLOAT_SPECIES.length();
    int unrolledLimit = limit - 3 * lanes;

    for (; i < unrolledLimit; i += 4 * lanes) {
      FloatVector va0 = FloatVector.fromArray(FLOAT_SPECIES, a, i);
      FloatVector vb0 = FloatVector.fromArray(FLOAT_SPECIES, b, i);
      FloatVector va1 = FloatVector.fromArray(FLOAT_SPECIES, a, i + lanes);
      FloatVector vb1 = FloatVector.fromArray(FLOAT_SPECIES, b, i + lanes);
      FloatVector va2 = FloatVector.fromArray(FLOAT_SPECIES, a, i + 2 * lanes);
      FloatVector vb2 = FloatVector.fromArray(FLOAT_SPECIES, b, i + 2 * lanes);
      FloatVector va3 = FloatVector.fromArray(FLOAT_SPECIES, a, i + 3 * lanes);
      FloatVector vb3 = FloatVector.fromArray(FLOAT_SPECIES, b, i + 3 * lanes);

      s0 = fma(va0, vb0, s0);
      s1 = fma(va1, vb1, s1);
      s2 = fma(va2, vb2, s2);
      s3 = fma(va3, vb3, s3);
      n1_0 = fma(va0, va0, n1_0);
      n1_1 = fma(va1, va1, n1_1);
      n1_2 = fma(va2, va2, n1_2);
      n1_3 = fma(va3, va3, n1_3);
      n2_0 = fma(vb0, vb0, n2_0);
      n2_1 = fma(vb1, vb1, n2_1);
      n2_2 = fma(vb2, vb2, n2_2);
      n2_3 = fma(vb3, vb3, n2_3);
    }

    // Vector tail (1 lane-width at a time)
    for (; i < limit; i += lanes) {
      FloatVector va = FloatVector.fromArray(FLOAT_SPECIES, a, i);
      FloatVector vb = FloatVector.fromArray(FLOAT_SPECIES, b, i);
      s0 = fma(va, vb, s0);
      n1_0 = fma(va, va, n1_0);
      n2_0 = fma(vb, vb, n2_0);
    }

    return new float[] {
      s0.add(s1).add(s2.add(s3)).reduceLanes(VectorOperators.ADD),
      n1_0.add(n1_1).add(n1_2.add(n1_3)).reduceLanes(VectorOperators.ADD),
      n2_0.add(n2_1).add(n2_2.add(n2_3)).reduceLanes(VectorOperators.ADD)
    };
  }

  private float[] cosineBody1x(float[] a, float[] b, int limit) {
    FloatVector s = FloatVector.zero(FLOAT_SPECIES);
    FloatVector n1 = FloatVector.zero(FLOAT_SPECIES);
    FloatVector n2 = FloatVector.zero(FLOAT_SPECIES);
    int lanes = FLOAT_SPECIES.length();
    for (int i = 0; i < limit; i += lanes) {
      FloatVector va = FloatVector.fromArray(FLOAT_SPECIES, a, i);
      FloatVector vb = FloatVector.fromArray(FLOAT_SPECIES, b, i);
      s = fma(va, vb, s);
      n1 = fma(va, va, n1);
      n2 = fma(vb, vb, n2);
    }
    return new float[] {
      s.reduceLanes(VectorOperators.ADD),
      n1.reduceLanes(VectorOperators.ADD),
      n2.reduceLanes(VectorOperators.ADD)
    };
  }

  /** SIMD Q4_0 by Q8_0 GEMV with one activation quantization shared by all rows. */
  @Override
  public void ggufQ4_0Q8_0MatVecDot(
      float[] query,
      MemorySegment qWeight,
      int rows,
      int cols,
      float[] out,
      byte[] q8Quants,
      float[] q8Scales,
      int[] q8ZeroPointCorrections,
      GgufQ4Kernel kernel) {
    long rowBytes = (long) (cols / GGUF_Q_BLOCK_SIZE) * GGUF_Q4_0_BLOCK_BYTES;
    int blocks = cols / GGUF_Q_BLOCK_SIZE;
    boolean useShortPairwise = useQ4ShortPairwise(kernel, blocks);
    boolean useUnsignedPairwise = useQ4UnsignedPairwise(kernel, blocks);
    quantizeQ8_0ForQ4(
        query, 0, cols, q8Quants, 0, q8Scales, 0, q8ZeroPointCorrections, 0, useUnsignedPairwise);
    GgufParallelSupport.forEachRow(
        qWeight,
        rows,
        cols,
        row ->
            out[row] =
                useUnsignedPairwise
                    ? q4_0Q8_0UnsignedPairwiseRowDot(
                        qWeight, row * rowBytes, blocks, q8Quants, q8Scales, q8ZeroPointCorrections)
                    : q4_0Q8_0RowDot(
                        qWeight, row * rowBytes, blocks, q8Quants, q8Scales, useShortPairwise));
  }

  @Override
  public void ggufQ4_0Q8_0DualMatVecDot(
      float[] query,
      MemorySegment firstWeight,
      int firstRows,
      float[] firstOut,
      MemorySegment secondWeight,
      int secondRows,
      float[] secondOut,
      int cols,
      byte[] q8Quants,
      float[] q8Scales,
      int[] q8ZeroPointCorrections,
      GgufQ4Kernel kernel) {
    long rowBytes = (long) (cols / GGUF_Q_BLOCK_SIZE) * GGUF_Q4_0_BLOCK_BYTES;
    int blocks = cols / GGUF_Q_BLOCK_SIZE;
    boolean useShortPairwise = useQ4ShortPairwise(kernel, blocks);
    boolean useUnsignedPairwise = useQ4UnsignedPairwise(kernel, blocks);
    quantizeQ8_0ForQ4(
        query, 0, cols, q8Quants, 0, q8Scales, 0, q8ZeroPointCorrections, 0, useUnsignedPairwise);
    int totalRows = Math.addExact(firstRows, secondRows);
    GgufParallelSupport.forEachRow(
        firstWeight,
        secondWeight,
        totalRows,
        cols,
        // Keep the SIMD body in this lambda. Extracting it allocated about 38 MB per grouped call.
        row -> {
          MemorySegment qWeight;
          float[] out;
          int matrixRow;
          if (row < firstRows) {
            qWeight = firstWeight;
            out = firstOut;
            matrixRow = row;
          } else {
            qWeight = secondWeight;
            out = secondOut;
            matrixRow = row - firstRows;
          }

          long rowOffset = matrixRow * rowBytes;
          if (useUnsignedPairwise) {
            out[matrixRow] =
                q4_0Q8_0UnsignedPairwiseRowDot(
                    qWeight, rowOffset, blocks, q8Quants, q8Scales, q8ZeroPointCorrections);
            return;
          }
          if (VECTOR_BITSIZE >= 256) {
            FloatVector accumulator = FloatVector.zero(FloatVector.SPECIES_256);
            for (int block = 0; block < blocks; block++) {
              long blockOffset = rowOffset + (long) block * GGUF_Q4_0_BLOCK_BYTES;
              float scale =
                  Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset)) * q8Scales[block];
              IntVector integerLanes =
                  useShortPairwise
                      ? q4_0Q8_0ShortPairwiseIntegerLanes(
                          qWeight, blockOffset + Short.BYTES, q8Quants, block * GGUF_Q_BLOCK_SIZE)
                      : q4_0Q8_0IntegerLanes(
                          qWeight, blockOffset + Short.BYTES, q8Quants, block * GGUF_Q_BLOCK_SIZE);
              FloatVector products =
                  (FloatVector)
                      integerLanes.convertShape(VectorOperators.I2F, FloatVector.SPECIES_256, 0);
              accumulator =
                  fma(products, FloatVector.broadcast(FloatVector.SPECIES_256, scale), accumulator);
            }
            out[matrixRow] = reduceAdd(accumulator);
            return;
          }

          float sum = 0.0f;
          for (int block = 0; block < blocks; block++) {
            long blockOffset = rowOffset + (long) block * GGUF_Q4_0_BLOCK_BYTES;
            float scale =
                Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset)) * q8Scales[block];
            int integerSum =
                q4_0Q8_0IntegerDot(
                    qWeight, blockOffset + Short.BYTES, q8Quants, block * GGUF_Q_BLOCK_SIZE);
            sum = MathUtil.fma(scale, integerSum, sum);
          }
          out[matrixRow] = sum;
        });
  }

  @Override
  public void ggufQ4_0Q8_0TripleMatVecDot(
      float[] query,
      MemorySegment firstWeight,
      int firstRows,
      float[] firstOut,
      MemorySegment secondWeight,
      int secondRows,
      float[] secondOut,
      MemorySegment thirdWeight,
      int thirdRows,
      float[] thirdOut,
      int cols,
      byte[] q8Quants,
      float[] q8Scales,
      int[] q8ZeroPointCorrections,
      GgufQ4Kernel kernel) {
    long rowBytes = (long) (cols / GGUF_Q_BLOCK_SIZE) * GGUF_Q4_0_BLOCK_BYTES;
    int blocks = cols / GGUF_Q_BLOCK_SIZE;
    boolean useShortPairwise = useQ4ShortPairwise(kernel, blocks);
    boolean useUnsignedPairwise = useQ4UnsignedPairwise(kernel, blocks);
    quantizeQ8_0ForQ4(
        query, 0, cols, q8Quants, 0, q8Scales, 0, q8ZeroPointCorrections, 0, useUnsignedPairwise);
    int secondStart = firstRows;
    int thirdStart = Math.addExact(firstRows, secondRows);
    int totalRows = Math.addExact(thirdStart, thirdRows);
    GgufParallelSupport.forEachRow(
        firstWeight,
        secondWeight,
        thirdWeight,
        totalRows,
        cols,
        // Keep the SIMD body in this lambda. Extracting it allocated about 38 MB per grouped call.
        row -> {
          MemorySegment qWeight;
          float[] out;
          int matrixRow;
          if (row < secondStart) {
            qWeight = firstWeight;
            out = firstOut;
            matrixRow = row;
          } else if (row < thirdStart) {
            qWeight = secondWeight;
            out = secondOut;
            matrixRow = row - secondStart;
          } else {
            qWeight = thirdWeight;
            out = thirdOut;
            matrixRow = row - thirdStart;
          }

          long rowOffset = matrixRow * rowBytes;
          if (useUnsignedPairwise) {
            out[matrixRow] =
                q4_0Q8_0UnsignedPairwiseRowDot(
                    qWeight, rowOffset, blocks, q8Quants, q8Scales, q8ZeroPointCorrections);
            return;
          }
          if (VECTOR_BITSIZE >= 256) {
            FloatVector accumulator = FloatVector.zero(FloatVector.SPECIES_256);
            for (int block = 0; block < blocks; block++) {
              long blockOffset = rowOffset + (long) block * GGUF_Q4_0_BLOCK_BYTES;
              float scale =
                  Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset)) * q8Scales[block];
              IntVector integerLanes =
                  useShortPairwise
                      ? q4_0Q8_0ShortPairwiseIntegerLanes(
                          qWeight, blockOffset + Short.BYTES, q8Quants, block * GGUF_Q_BLOCK_SIZE)
                      : q4_0Q8_0IntegerLanes(
                          qWeight, blockOffset + Short.BYTES, q8Quants, block * GGUF_Q_BLOCK_SIZE);
              FloatVector products =
                  (FloatVector)
                      integerLanes.convertShape(VectorOperators.I2F, FloatVector.SPECIES_256, 0);
              accumulator =
                  fma(products, FloatVector.broadcast(FloatVector.SPECIES_256, scale), accumulator);
            }
            out[matrixRow] = reduceAdd(accumulator);
            return;
          }

          float sum = 0.0f;
          for (int block = 0; block < blocks; block++) {
            long blockOffset = rowOffset + (long) block * GGUF_Q4_0_BLOCK_BYTES;
            float scale =
                Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset)) * q8Scales[block];
            int integerSum =
                q4_0Q8_0IntegerDot(
                    qWeight, blockOffset + Short.BYTES, q8Quants, block * GGUF_Q_BLOCK_SIZE);
            sum = MathUtil.fma(scale, integerSum, sum);
          }
          out[matrixRow] = sum;
        });
  }

  @Override
  public void ggufQ4_0Q8_0BatchedMatmul(
      float[] queries,
      MemorySegment qWeight,
      int batchSize,
      int rows,
      int cols,
      float[] out,
      byte[] q8Quants,
      float[] q8Scales,
      int[] q8ZeroPointCorrections,
      float[] laneScratch,
      GgufQ4Kernel kernel) {
    if (batchSize == 1) {
      ggufQ4_0Q8_0MatVecDot(
          queries, qWeight, rows, cols, out, q8Quants, q8Scales, q8ZeroPointCorrections, kernel);
      return;
    }
    if (VECTOR_BITSIZE < 256) {
      VectorUtilSupport.super.ggufQ4_0Q8_0BatchedMatmul(
          queries,
          qWeight,
          batchSize,
          rows,
          cols,
          out,
          q8Quants,
          q8Scales,
          q8ZeroPointCorrections,
          laneScratch,
          kernel);
      return;
    }

    int blocks = cols / GGUF_Q_BLOCK_SIZE;
    boolean useShortPairwise = useQ4ShortPairwise(kernel, blocks);
    boolean useUnsignedPairwise = useQ4UnsignedPairwise(kernel, blocks);
    for (int batch = 0; batch < batchSize; batch++) {
      quantizeQ8_0ForBatchedQ4(
          queries,
          batch * cols,
          cols,
          q8Quants,
          batch * cols,
          q8Scales,
          batch * blocks,
          q8ZeroPointCorrections,
          batch * blocks * 4,
          useUnsignedPairwise);
    }

    long rowBytes = (long) blocks * GGUF_Q4_0_BLOCK_BYTES;
    GgufParallelSupport.forEachRow(
        qWeight,
        rows,
        cols,
        row ->
            ggufQ4_0Q8_0BatchedMatmulRow(
                qWeight,
                batchSize,
                rows,
                cols,
                row,
                out,
                q8Quants,
                q8Scales,
                q8ZeroPointCorrections,
                laneScratch,
                blocks,
                rowBytes,
                useShortPairwise,
                useUnsignedPairwise));
  }

  @Override
  public void ggufQ4_0Q8_0BatchedMatmulRows(
      MemorySegment qWeight,
      int batchSize,
      int rows,
      int cols,
      int fromRow,
      int toRow,
      float[] out,
      GgufQ8_0Batch activation,
      float[] laneScratch,
      GgufQ4Kernel kernel) {
    if (VECTOR_BITSIZE < 256) {
      VectorUtilSupport.super.ggufQ4_0Q8_0BatchedMatmulRows(
          qWeight, batchSize, rows, cols, fromRow, toRow, out, activation, laneScratch, kernel);
      return;
    }

    int blocks = cols / GGUF_Q_BLOCK_SIZE;
    long rowBytes = (long) blocks * GGUF_Q4_0_BLOCK_BYTES;
    boolean useShortPairwise = useQ4ShortPairwise(kernel, blocks);
    boolean useUnsignedPairwise = useQ4UnsignedPairwise(kernel, blocks);
    for (int row = fromRow; row < toRow; row++) {
      ggufQ4_0Q8_0BatchedMatmulRow(
          qWeight,
          batchSize,
          rows,
          cols,
          row,
          out,
          activation.quants(),
          activation.scales(),
          activation.zeroPointCorrections(),
          laneScratch,
          blocks,
          rowBytes,
          useShortPairwise,
          useUnsignedPairwise);
    }
  }

  private void ggufQ4_0Q8_0BatchedMatmulRow(
      MemorySegment qWeight,
      int batchSize,
      int rows,
      int cols,
      int row,
      float[] out,
      byte[] q8Quants,
      float[] q8Scales,
      int[] q8ZeroPointCorrections,
      float[] laneScratch,
      int blocks,
      long rowBytes,
      boolean useShortPairwise,
      boolean useUnsignedPairwise) {
    if (useUnsignedPairwise) {
      ggufQ4_0Q8_0UnsignedPairwiseBatchedMatmulBlockPairRow(
          qWeight,
          batchSize,
          rows,
          cols,
          row,
          row,
          out,
          q8Quants,
          q8Scales,
          q8ZeroPointCorrections,
          laneScratch,
          blocks,
          rowBytes);
      return;
    }

    long rowOffset = row * rowBytes;
    int rowLaneOffset = row * batchSize * FloatVector.SPECIES_256.length();
    int rowLaneEnd = rowLaneOffset + batchSize * FloatVector.SPECIES_256.length();
    for (int lane = rowLaneOffset; lane < rowLaneEnd; lane++) {
      laneScratch[lane] = 0.0f;
    }

    for (int block = 0; block < blocks; block++) {
      long blockOffset = rowOffset + (long) block * GGUF_Q4_0_BLOCK_BYTES;
      float weightScale = Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset));
      ByteVector packed =
          ByteVector.fromMemorySegment(
              ByteVector.SPECIES_128, qWeight, blockOffset + Short.BYTES, ByteOrder.LITTLE_ENDIAN);
      ByteVector lowNibbles = packed.and((byte) 0x0F);
      ByteVector highNibbles = q4HighNibbles(packed);
      ShortVector low =
          (ShortVector)
              lowNibbles
                  .sub((byte) 8)
                  .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0);
      ShortVector high =
          (ShortVector)
              highNibbles
                  .sub((byte) 8)
                  .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0);

      for (int batch = 0; batch < batchSize; batch++) {
        int laneOffset = rowLaneOffset + batch * FloatVector.SPECIES_256.length();
        accumulateQ4_0BatchQuery(
            laneScratch,
            laneOffset,
            low,
            high,
            q8Quants,
            (batch * cols) + block * GGUF_Q_BLOCK_SIZE,
            weightScale * q8Scales[batch * blocks + block],
            useShortPairwise);
      }
    }

    for (int batch = 0; batch < batchSize; batch++) {
      int laneOffset = rowLaneOffset + batch * FloatVector.SPECIES_256.length();
      out[batch * rows + row] =
          reduceAdd(FloatVector.fromArray(FloatVector.SPECIES_256, laneScratch, laneOffset));
    }
  }

  private static void ggufQ4_0Q8_0UnsignedPairwiseBatchedMatmulBlockPairRow(
      MemorySegment qWeight,
      int batchSize,
      int rows,
      int cols,
      int row,
      int scratchRow,
      float[] out,
      byte[] q8Quants,
      float[] q8Scales,
      int[] q8ZeroPointCorrections,
      float[] laneScratch,
      int blocks,
      long rowBytes) {
    long rowOffset = row * rowBytes;
    int rowLaneOffset = scratchRow * batchSize * FloatVector.SPECIES_256.length();
    int rowLaneEnd = rowLaneOffset + batchSize * FloatVector.SPECIES_256.length();
    for (int lane = rowLaneOffset; lane < rowLaneEnd; lane++) {
      laneScratch[lane] = 0.0f;
    }

    ShortVector pairFactors = ShortVector.fromArray(ShortVector.SPECIES_128, Q4_PAIR_FACTORS, 0);
    int block = 0;
    for (; block + 1 < blocks; block += 2) {
      long firstBlockOffset = rowOffset + (long) block * GGUF_Q4_0_BLOCK_BYTES;
      ByteVector firstPacked =
          ByteVector.fromMemorySegment(
              ByteVector.SPECIES_128,
              qWeight,
              firstBlockOffset + Short.BYTES,
              ByteOrder.LITTLE_ENDIAN);
      ByteVector firstLowNibbles = firstPacked.and((byte) 0x0F);
      ByteVector firstHighNibbles = q4HighNibbles(firstPacked);
      float firstWeightScale = Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, firstBlockOffset));

      long secondBlockOffset = firstBlockOffset + GGUF_Q4_0_BLOCK_BYTES;
      ByteVector secondPacked =
          ByteVector.fromMemorySegment(
              ByteVector.SPECIES_128,
              qWeight,
              secondBlockOffset + Short.BYTES,
              ByteOrder.LITTLE_ENDIAN);
      ByteVector secondLowNibbles = secondPacked.and((byte) 0x0F);
      ByteVector secondHighNibbles = q4HighNibbles(secondPacked);
      float secondWeightScale = Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, secondBlockOffset));

      for (int batch = 0; batch < batchSize; batch++) {
        int laneOffset = rowLaneOffset + batch * FloatVector.SPECIES_256.length();
        int blockIndex = batch * blocks + block;
        accumulateQ4_0UnsignedBatchQueryBlockPair(
            laneScratch,
            laneOffset,
            firstLowNibbles,
            firstHighNibbles,
            secondLowNibbles,
            secondHighNibbles,
            q8Quants,
            batch * cols + block * GGUF_Q_BLOCK_SIZE,
            q8ZeroPointCorrections,
            blockIndex * 4,
            firstWeightScale * q8Scales[blockIndex],
            secondWeightScale * q8Scales[blockIndex + 1],
            pairFactors);
      }
    }

    if (block < blocks) {
      long blockOffset = rowOffset + (long) block * GGUF_Q4_0_BLOCK_BYTES;
      float weightScale = Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset));
      ByteVector packed =
          ByteVector.fromMemorySegment(
              ByteVector.SPECIES_128, qWeight, blockOffset + Short.BYTES, ByteOrder.LITTLE_ENDIAN);
      ByteVector lowNibbles = packed.and((byte) 0x0F);
      ByteVector highNibbles = q4HighNibbles(packed);
      for (int batch = 0; batch < batchSize; batch++) {
        int laneOffset = rowLaneOffset + batch * FloatVector.SPECIES_256.length();
        int blockIndex = batch * blocks + block;
        accumulateQ4_0UnsignedBatchQuery(
            laneScratch,
            laneOffset,
            lowNibbles,
            highNibbles,
            q8Quants,
            batch * cols + block * GGUF_Q_BLOCK_SIZE,
            q8ZeroPointCorrections,
            blockIndex * 4,
            weightScale * q8Scales[blockIndex]);
      }
    }

    for (int batch = 0; batch < batchSize; batch++) {
      int laneOffset = rowLaneOffset + batch * FloatVector.SPECIES_256.length();
      out[batch * rows + row] =
          reduceAdd(FloatVector.fromArray(FloatVector.SPECIES_128, laneScratch, laneOffset));
    }
  }

  @Override
  public void ggufQ4_0Q8_0DualBatchedMatmul(
      float[] queries,
      MemorySegment firstWeight,
      int firstRows,
      float[] firstOut,
      MemorySegment secondWeight,
      int secondRows,
      float[] secondOut,
      int batchSize,
      int cols,
      byte[] q8Quants,
      float[] q8Scales,
      int[] q8ZeroPointCorrections,
      float[] laneScratch,
      GgufQ4Kernel kernel) {
    if (batchSize == 1) {
      ggufQ4_0Q8_0DualMatVecDot(
          queries,
          firstWeight,
          firstRows,
          firstOut,
          secondWeight,
          secondRows,
          secondOut,
          cols,
          q8Quants,
          q8Scales,
          q8ZeroPointCorrections,
          kernel);
      return;
    }
    ggufQ4_0Q8_0GroupedBatchedMatmul(
        queries,
        firstWeight,
        firstRows,
        firstOut,
        secondWeight,
        secondRows,
        secondOut,
        secondWeight,
        0,
        secondOut,
        batchSize,
        cols,
        q8Quants,
        q8Scales,
        q8ZeroPointCorrections,
        laneScratch,
        kernel);
  }

  @Override
  public void ggufQ4_0Q8_0TripleBatchedMatmul(
      float[] queries,
      MemorySegment firstWeight,
      int firstRows,
      float[] firstOut,
      MemorySegment secondWeight,
      int secondRows,
      float[] secondOut,
      MemorySegment thirdWeight,
      int thirdRows,
      float[] thirdOut,
      int batchSize,
      int cols,
      byte[] q8Quants,
      float[] q8Scales,
      int[] q8ZeroPointCorrections,
      float[] laneScratch,
      GgufQ4Kernel kernel) {
    if (batchSize == 1) {
      ggufQ4_0Q8_0TripleMatVecDot(
          queries,
          firstWeight,
          firstRows,
          firstOut,
          secondWeight,
          secondRows,
          secondOut,
          thirdWeight,
          thirdRows,
          thirdOut,
          cols,
          q8Quants,
          q8Scales,
          q8ZeroPointCorrections,
          kernel);
      return;
    }
    ggufQ4_0Q8_0GroupedBatchedMatmul(
        queries,
        firstWeight,
        firstRows,
        firstOut,
        secondWeight,
        secondRows,
        secondOut,
        thirdWeight,
        thirdRows,
        thirdOut,
        batchSize,
        cols,
        q8Quants,
        q8Scales,
        q8ZeroPointCorrections,
        laneScratch,
        kernel);
  }

  private void ggufQ4_0Q8_0GroupedBatchedMatmul(
      float[] queries,
      MemorySegment firstWeight,
      int firstRows,
      float[] firstOut,
      MemorySegment secondWeight,
      int secondRows,
      float[] secondOut,
      MemorySegment thirdWeight,
      int thirdRows,
      float[] thirdOut,
      int batchSize,
      int cols,
      byte[] q8Quants,
      float[] q8Scales,
      int[] q8ZeroPointCorrections,
      float[] laneScratch,
      GgufQ4Kernel kernel) {
    if (VECTOR_BITSIZE < 256) {
      if (thirdRows == 0) {
        VectorUtilSupport.super.ggufQ4_0Q8_0DualBatchedMatmul(
            queries,
            firstWeight,
            firstRows,
            firstOut,
            secondWeight,
            secondRows,
            secondOut,
            batchSize,
            cols,
            q8Quants,
            q8Scales,
            q8ZeroPointCorrections,
            laneScratch,
            kernel);
      } else {
        VectorUtilSupport.super.ggufQ4_0Q8_0TripleBatchedMatmul(
            queries,
            firstWeight,
            firstRows,
            firstOut,
            secondWeight,
            secondRows,
            secondOut,
            thirdWeight,
            thirdRows,
            thirdOut,
            batchSize,
            cols,
            q8Quants,
            q8Scales,
            q8ZeroPointCorrections,
            laneScratch,
            kernel);
      }
      return;
    }

    int blocks = cols / GGUF_Q_BLOCK_SIZE;
    boolean useShortPairwise = useQ4ShortPairwise(kernel, blocks);
    boolean useUnsignedPairwise = useQ4UnsignedPairwise(kernel, blocks);
    for (int batch = 0; batch < batchSize; batch++) {
      quantizeQ8_0ForBatchedQ4(
          queries,
          batch * cols,
          cols,
          q8Quants,
          batch * cols,
          q8Scales,
          batch * blocks,
          q8ZeroPointCorrections,
          batch * blocks * 4,
          useUnsignedPairwise);
    }

    long rowBytes = (long) blocks * GGUF_Q4_0_BLOCK_BYTES;
    int secondStart = firstRows;
    int thirdStart = Math.addExact(firstRows, secondRows);
    int totalRows = Math.addExact(thirdStart, thirdRows);
    GgufParallelSupport.forEachRow(
        firstWeight,
        secondWeight,
        thirdWeight,
        totalRows,
        cols,
        // The combined row index gives every concurrent matrix row disjoint lane scratch.
        row -> {
          MemorySegment qWeight;
          float[] out;
          int matrixRow;
          int matrixRows;
          if (row < secondStart) {
            qWeight = firstWeight;
            out = firstOut;
            matrixRow = row;
            matrixRows = firstRows;
          } else if (row < thirdStart) {
            qWeight = secondWeight;
            out = secondOut;
            matrixRow = row - secondStart;
            matrixRows = secondRows;
          } else {
            qWeight = thirdWeight;
            out = thirdOut;
            matrixRow = row - thirdStart;
            matrixRows = thirdRows;
          }

          if (useUnsignedPairwise) {
            ggufQ4_0Q8_0UnsignedPairwiseBatchedMatmulBlockPairRow(
                qWeight,
                batchSize,
                matrixRows,
                cols,
                matrixRow,
                row,
                out,
                q8Quants,
                q8Scales,
                q8ZeroPointCorrections,
                laneScratch,
                blocks,
                rowBytes);
            return;
          }

          long rowOffset = matrixRow * rowBytes;
          int rowLaneOffset = row * batchSize * FloatVector.SPECIES_256.length();
          int rowLaneEnd = rowLaneOffset + batchSize * FloatVector.SPECIES_256.length();
          for (int lane = rowLaneOffset; lane < rowLaneEnd; lane++) {
            laneScratch[lane] = 0.0f;
          }

          for (int block = 0; block < blocks; block++) {
            long blockOffset = rowOffset + (long) block * GGUF_Q4_0_BLOCK_BYTES;
            float weightScale = Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset));
            ByteVector packed =
                ByteVector.fromMemorySegment(
                    ByteVector.SPECIES_128,
                    qWeight,
                    blockOffset + Short.BYTES,
                    ByteOrder.LITTLE_ENDIAN);
            ByteVector lowNibbles = packed.and((byte) 0x0F);
            ByteVector highNibbles = q4HighNibbles(packed);
            ShortVector low =
                (ShortVector)
                    lowNibbles
                        .sub((byte) 8)
                        .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0);
            ShortVector high =
                (ShortVector)
                    highNibbles
                        .sub((byte) 8)
                        .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0);

            for (int batch = 0; batch < batchSize; batch++) {
              int laneOffset = rowLaneOffset + batch * FloatVector.SPECIES_256.length();
              accumulateQ4_0BatchQuery(
                  laneScratch,
                  laneOffset,
                  low,
                  high,
                  q8Quants,
                  (batch * cols) + block * GGUF_Q_BLOCK_SIZE,
                  weightScale * q8Scales[batch * blocks + block],
                  useShortPairwise);
            }
          }

          for (int batch = 0; batch < batchSize; batch++) {
            int laneOffset = rowLaneOffset + batch * FloatVector.SPECIES_256.length();
            out[batch * matrixRows + matrixRow] =
                reduceAdd(FloatVector.fromArray(FloatVector.SPECIES_256, laneScratch, laneOffset));
          }
        });
  }

  private static void accumulateQ4_0BatchQuery(
      float[] laneScratch,
      int laneOffset,
      ShortVector low,
      ShortVector high,
      byte[] q8Quants,
      int quantOffset,
      float scale,
      boolean useShortPairwise) {
    ShortVector lowQuants =
        (ShortVector)
            ByteVector.fromArray(ByteVector.SPECIES_128, q8Quants, quantOffset)
                .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0);
    ShortVector highQuants =
        (ShortVector)
            ByteVector.fromArray(ByteVector.SPECIES_128, q8Quants, quantOffset + 16)
                .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0);
    IntVector integerLanes =
        useShortPairwise
            ? fourProductLanesFromShortPairs(low, high, lowQuants, highQuants)
            : fourProductLanes(low.mul(lowQuants), high.mul(highQuants));
    FloatVector products =
        (FloatVector) integerLanes.convertShape(VectorOperators.I2F, FloatVector.SPECIES_256, 0);
    FloatVector accumulator =
        FloatVector.fromArray(FloatVector.SPECIES_256, laneScratch, laneOffset);
    fma(products, FloatVector.broadcast(FloatVector.SPECIES_256, scale), accumulator)
        .intoArray(laneScratch, laneOffset);
  }

  private static void accumulateQ4_0UnsignedBatchQuery(
      float[] laneScratch,
      int laneOffset,
      ByteVector lowNibbles,
      ByteVector highNibbles,
      byte[] q8Quants,
      int quantOffset,
      int[] zeroPointCorrections,
      int correctionOffset,
      float scale) {
    ShortVector pairFactors = ShortVector.fromArray(ShortVector.SPECIES_128, Q4_PAIR_FACTORS, 0);
    IntVector lowProducts =
        q4_0Q8_0UnsignedPairwiseProducts(
            lowNibbles,
            ByteVector.fromArray(ByteVector.SPECIES_128, q8Quants, quantOffset),
            pairFactors);
    IntVector highProducts =
        q4_0Q8_0UnsignedPairwiseProducts(
            highNibbles,
            ByteVector.fromArray(ByteVector.SPECIES_128, q8Quants, quantOffset + 16),
            pairFactors);
    IntVector combinedGroups =
        lowProducts
            .add(highProducts)
            .sub(
                IntVector.fromArray(IntVector.SPECIES_128, zeroPointCorrections, correctionOffset));
    FloatVector accumulator =
        FloatVector.fromArray(FloatVector.SPECIES_128, laneScratch, laneOffset);
    fma(
            (FloatVector)
                combinedGroups.convertShape(VectorOperators.I2F, FloatVector.SPECIES_128, 0),
            FloatVector.broadcast(FloatVector.SPECIES_128, scale),
            accumulator)
        .intoArray(laneScratch, laneOffset);
  }

  private static void accumulateQ4_0UnsignedBatchQueryBlockPair(
      float[] laneScratch,
      int laneOffset,
      ByteVector firstLowNibbles,
      ByteVector firstHighNibbles,
      ByteVector secondLowNibbles,
      ByteVector secondHighNibbles,
      byte[] q8Quants,
      int quantOffset,
      int[] zeroPointCorrections,
      int correctionOffset,
      float firstScale,
      float secondScale,
      ShortVector pairFactors) {
    FloatVector accumulator =
        FloatVector.fromArray(FloatVector.SPECIES_128, laneScratch, laneOffset);

    IntVector firstLowProducts =
        q4_0Q8_0UnsignedPairwiseProducts(
            firstLowNibbles,
            ByteVector.fromArray(ByteVector.SPECIES_128, q8Quants, quantOffset),
            pairFactors);
    IntVector firstHighProducts =
        q4_0Q8_0UnsignedPairwiseProducts(
            firstHighNibbles,
            ByteVector.fromArray(ByteVector.SPECIES_128, q8Quants, quantOffset + 16),
            pairFactors);
    IntVector firstCombinedGroups =
        firstLowProducts
            .add(firstHighProducts)
            .sub(
                IntVector.fromArray(IntVector.SPECIES_128, zeroPointCorrections, correctionOffset));
    accumulator =
        fma(
            (FloatVector)
                firstCombinedGroups.convertShape(VectorOperators.I2F, FloatVector.SPECIES_128, 0),
            FloatVector.broadcast(FloatVector.SPECIES_128, firstScale),
            accumulator);

    int secondQuantOffset = quantOffset + GGUF_Q_BLOCK_SIZE;
    int secondCorrectionOffset = correctionOffset + 4;
    IntVector secondLowProducts =
        q4_0Q8_0UnsignedPairwiseProducts(
            secondLowNibbles,
            ByteVector.fromArray(ByteVector.SPECIES_128, q8Quants, secondQuantOffset),
            pairFactors);
    IntVector secondHighProducts =
        q4_0Q8_0UnsignedPairwiseProducts(
            secondHighNibbles,
            ByteVector.fromArray(ByteVector.SPECIES_128, q8Quants, secondQuantOffset + 16),
            pairFactors);
    IntVector secondCombinedGroups =
        secondLowProducts
            .add(secondHighProducts)
            .sub(
                IntVector.fromArray(
                    IntVector.SPECIES_128, zeroPointCorrections, secondCorrectionOffset));
    fma(
            (FloatVector)
                secondCombinedGroups.convertShape(VectorOperators.I2F, FloatVector.SPECIES_128, 0),
            FloatVector.broadcast(FloatVector.SPECIES_128, secondScale),
            accumulator)
        .intoArray(laneScratch, laneOffset);
  }

  /** Fixed split-half hsum tree; Vector.reduceLanes does not guarantee an evaluation order. */
  private static float reduceAdd(FloatVector vector) {
    return switch (vector.length()) {
      case 1 -> vector.lane(0);
      case 2 -> vector.lane(1) + vector.lane(0);
      case 4 -> (vector.lane(2) + vector.lane(0)) + (vector.lane(3) + vector.lane(1));
      case 8 -> {
        float even = (vector.lane(4) + vector.lane(0)) + (vector.lane(6) + vector.lane(2));
        float odd = (vector.lane(5) + vector.lane(1)) + (vector.lane(7) + vector.lane(3));
        yield even + odd;
      }
      case 16 -> {
        float lane0 = vector.lane(8) + vector.lane(0);
        float lane1 = vector.lane(9) + vector.lane(1);
        float lane2 = vector.lane(10) + vector.lane(2);
        float lane3 = vector.lane(11) + vector.lane(3);
        float lane4 = vector.lane(12) + vector.lane(4);
        float lane5 = vector.lane(13) + vector.lane(5);
        float lane6 = vector.lane(14) + vector.lane(6);
        float lane7 = vector.lane(15) + vector.lane(7);
        float even = (lane4 + lane0) + (lane6 + lane2);
        float odd = (lane5 + lane1) + (lane7 + lane3);
        yield even + odd;
      }
      default -> throw new AssertionError("unsupported float vector length: " + vector.length());
    };
  }

  private static float q4_0Q8_0RowDot(
      MemorySegment qWeight,
      long rowOffset,
      int blocks,
      byte[] q8Quants,
      float[] q8Scales,
      boolean useShortPairwise) {
    if (VECTOR_BITSIZE >= 256) {
      FloatVector accumulator = FloatVector.zero(FloatVector.SPECIES_256);
      for (int block = 0; block < blocks; block++) {
        long blockOffset = rowOffset + (long) block * GGUF_Q4_0_BLOCK_BYTES;
        float scale =
            Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset)) * q8Scales[block];
        IntVector integerLanes =
            useShortPairwise
                ? q4_0Q8_0ShortPairwiseIntegerLanes(
                    qWeight, blockOffset + Short.BYTES, q8Quants, block * GGUF_Q_BLOCK_SIZE)
                : q4_0Q8_0IntegerLanes(
                    qWeight, blockOffset + Short.BYTES, q8Quants, block * GGUF_Q_BLOCK_SIZE);
        FloatVector products =
            (FloatVector)
                integerLanes.convertShape(VectorOperators.I2F, FloatVector.SPECIES_256, 0);
        accumulator =
            fma(products, FloatVector.broadcast(FloatVector.SPECIES_256, scale), accumulator);
      }
      return reduceAdd(accumulator);
    }

    float sum = 0.0f;
    for (int block = 0; block < blocks; block++) {
      long blockOffset = rowOffset + (long) block * GGUF_Q4_0_BLOCK_BYTES;
      float scale = Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset)) * q8Scales[block];
      int integerSum =
          q4_0Q8_0IntegerDot(
              qWeight, blockOffset + Short.BYTES, q8Quants, block * GGUF_Q_BLOCK_SIZE);
      sum = MathUtil.fma(scale, integerSum, sum);
    }
    return sum;
  }

  private static int q4_0Q8_0IntegerDot(
      MemorySegment qWeight, long nibbleOffset, byte[] q8Quants, int quantOffset) {
    if (VECTOR_BITSIZE >= 256) {
      ByteVector packed =
          ByteVector.fromMemorySegment(
              ByteVector.SPECIES_128, qWeight, nibbleOffset, ByteOrder.LITTLE_ENDIAN);
      ByteVector low = packed.and((byte) 0x0F).sub((byte) 8);
      ByteVector high = q4HighNibbles(packed).sub((byte) 8);
      ShortVector low16 =
          (ShortVector) low.convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0);
      ShortVector high16 =
          (ShortVector) high.convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0);
      ShortVector qLow16 =
          (ShortVector)
              ByteVector.fromArray(ByteVector.SPECIES_128, q8Quants, quantOffset)
                  .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0);
      ShortVector qHigh16 =
          (ShortVector)
              ByteVector.fromArray(ByteVector.SPECIES_128, q8Quants, quantOffset + 16)
                  .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0);
      return low16.mul(qLow16).reduceLanes(VectorOperators.ADD)
          + high16.mul(qHigh16).reduceLanes(VectorOperators.ADD);
    }

    int sum = 0;
    for (int half = 0; half < 16; half += 8) {
      ByteVector packed =
          ByteVector.fromMemorySegment(
              ByteVector.SPECIES_64, qWeight, nibbleOffset + half, ByteOrder.LITTLE_ENDIAN);
      ByteVector low = packed.and((byte) 0x0F).sub((byte) 8);
      ByteVector high = q4HighNibbles(packed).sub((byte) 8);
      ShortVector low16 =
          (ShortVector) low.convertShape(VectorOperators.B2S, ShortVector.SPECIES_128, 0);
      ShortVector high16 =
          (ShortVector) high.convertShape(VectorOperators.B2S, ShortVector.SPECIES_128, 0);
      ShortVector qLow16 =
          (ShortVector)
              ByteVector.fromArray(ByteVector.SPECIES_64, q8Quants, quantOffset + half)
                  .convertShape(VectorOperators.B2S, ShortVector.SPECIES_128, 0);
      ShortVector qHigh16 =
          (ShortVector)
              ByteVector.fromArray(ByteVector.SPECIES_64, q8Quants, quantOffset + 16 + half)
                  .convertShape(VectorOperators.B2S, ShortVector.SPECIES_128, 0);
      sum += low16.mul(qLow16).reduceLanes(VectorOperators.ADD);
      sum += high16.mul(qHigh16).reduceLanes(VectorOperators.ADD);
    }
    return sum;
  }

  static IntVector q4_0Q8_0IntegerLanes(
      MemorySegment qWeight, long nibbleOffset, byte[] q8Quants, int quantOffset) {
    ByteVector packed =
        ByteVector.fromMemorySegment(
            ByteVector.SPECIES_128, qWeight, nibbleOffset, ByteOrder.LITTLE_ENDIAN);
    ByteVector low = packed.and((byte) 0x0F).sub((byte) 8);
    ByteVector high = q4HighNibbles(packed).sub((byte) 8);
    ShortVector lowProducts =
        ((ShortVector) low.convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0))
            .mul(
                (ShortVector)
                    ByteVector.fromArray(ByteVector.SPECIES_128, q8Quants, quantOffset)
                        .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0));
    ShortVector highProducts =
        ((ShortVector) high.convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0))
            .mul(
                (ShortVector)
                    ByteVector.fromArray(ByteVector.SPECIES_128, q8Quants, quantOffset + 16)
                        .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0));

    return fourProductLanes(lowProducts, highProducts);
  }

  static IntVector q4_0Q8_0ShortPairwiseIntegerLanes(
      MemorySegment qWeight, long nibbleOffset, byte[] q8Quants, int quantOffset) {
    ByteVector packed =
        ByteVector.fromMemorySegment(
            ByteVector.SPECIES_128, qWeight, nibbleOffset, ByteOrder.LITTLE_ENDIAN);
    ByteVector low = packed.and((byte) 0x0F).sub((byte) 8);
    ByteVector high = q4HighNibbles(packed).sub((byte) 8);
    ShortVector low16 =
        (ShortVector) low.convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0);
    ShortVector high16 =
        (ShortVector) high.convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0);
    ShortVector qLow16 =
        (ShortVector)
            ByteVector.fromArray(ByteVector.SPECIES_128, q8Quants, quantOffset)
                .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0);
    ShortVector qHigh16 =
        (ShortVector)
            ByteVector.fromArray(ByteVector.SPECIES_128, q8Quants, quantOffset + 16)
                .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0);
    return fourProductLanesFromShortPairs(low16, high16, qLow16, qHigh16);
  }

  static float q4_0Q8_0UnsignedPairwiseRowDot(
      MemorySegment qWeight,
      long rowOffset,
      int blocks,
      byte[] q8Quants,
      float[] q8Scales,
      int[] zeroPointCorrections) {
    ShortVector pairFactors = ShortVector.fromArray(ShortVector.SPECIES_128, Q4_PAIR_FACTORS, 0);
    FloatVector lowAccumulator = FloatVector.zero(FloatVector.SPECIES_128);
    FloatVector highAccumulator = FloatVector.zero(FloatVector.SPECIES_128);
    for (int block = 0; block < blocks; block++) {
      long blockOffset = rowOffset + (long) block * GGUF_Q4_0_BLOCK_BYTES;
      float scale = Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset)) * q8Scales[block];
      ByteVector packed =
          ByteVector.fromMemorySegment(
              ByteVector.SPECIES_128, qWeight, blockOffset + Short.BYTES, ByteOrder.LITTLE_ENDIAN);
      ByteVector low = packed.and((byte) 0x0F);
      ByteVector high = q4HighNibbles(packed);
      int quantOffset = block * GGUF_Q_BLOCK_SIZE;
      int correctionOffset = block * 8;
      IntVector lowGroups =
          q4_0Q8_0UnsignedPairwiseGroups(
              low,
              ByteVector.fromArray(ByteVector.SPECIES_128, q8Quants, quantOffset),
              IntVector.fromArray(IntVector.SPECIES_128, zeroPointCorrections, correctionOffset),
              pairFactors);
      IntVector highGroups =
          q4_0Q8_0UnsignedPairwiseGroups(
              high,
              ByteVector.fromArray(ByteVector.SPECIES_128, q8Quants, quantOffset + 16),
              IntVector.fromArray(
                  IntVector.SPECIES_128, zeroPointCorrections, correctionOffset + 4),
              pairFactors);
      FloatVector scaleVector = FloatVector.broadcast(FloatVector.SPECIES_128, scale);
      lowAccumulator =
          fma(
              (FloatVector) lowGroups.convertShape(VectorOperators.I2F, FloatVector.SPECIES_128, 0),
              scaleVector,
              lowAccumulator);
      highAccumulator =
          fma(
              (FloatVector)
                  highGroups.convertShape(VectorOperators.I2F, FloatVector.SPECIES_128, 0),
              scaleVector,
              highAccumulator);
    }
    float even =
        (highAccumulator.lane(0) + lowAccumulator.lane(0))
            + (highAccumulator.lane(2) + lowAccumulator.lane(2));
    float odd =
        (highAccumulator.lane(1) + lowAccumulator.lane(1))
            + (highAccumulator.lane(3) + lowAccumulator.lane(3));
    return even + odd;
  }

  private static IntVector q4_0Q8_0UnsignedPairwiseGroups(
      ByteVector unsignedNibbles,
      ByteVector signedQ8,
      IntVector zeroPointCorrection,
      ShortVector pairFactors) {
    return q4_0Q8_0UnsignedPairwiseProducts(unsignedNibbles, signedQ8, pairFactors)
        .sub(zeroPointCorrection);
  }

  private static IntVector q4_0Q8_0UnsignedPairwiseProducts(
      ByteVector unsignedNibbles, ByteVector signedQ8, ShortVector pairFactors) {
    ShortVector pairProducts = multiplyAddUnsignedSignedBytes128(unsignedNibbles, signedQ8);
    return multiplyAddSignedShorts128(pairProducts, pairFactors);
  }

  private static IntVector fourProductLanesFromShortPairs(
      ShortVector low, ShortVector high, ShortVector qLow, ShortVector qHigh) {
    IntVector lowPairs = multiplyAddSignedShorts256(low, qLow);
    IntVector highPairs = multiplyAddSignedShorts256(high, qHigh);
    IntVector lowGroups =
        lowPairs.add(lowPairs.rearrange(SWAP_ADJACENT_INTS)).rearrange(SELECT_LOW_INT_GROUPS);
    IntVector highGroups =
        highPairs.add(highPairs.rearrange(SWAP_ADJACENT_INTS)).rearrange(SELECT_HIGH_INT_GROUPS);
    return lowGroups.blend(highGroups, HIGH_INT_GROUP_LANES);
  }

  private static IntVector multiplyAddSignedShorts256(ShortVector left, ShortVector right) {
    ShortVector leftEven = left.rearrange(SHORT_256_EVEN_LANES);
    ShortVector leftOdd = left.rearrange(SHORT_256_ODD_LANES);
    ShortVector rightEven = right.rearrange(SHORT_256_EVEN_LANES);
    ShortVector rightOdd = right.rearrange(SHORT_256_ODD_LANES);
    IntVector leftEvenInts =
        (IntVector) leftEven.convertShape(VectorOperators.S2I, IntVector.SPECIES_256, 0);
    IntVector leftOddInts =
        (IntVector) leftOdd.convertShape(VectorOperators.S2I, IntVector.SPECIES_256, 0);
    IntVector rightEvenInts =
        (IntVector) rightEven.convertShape(VectorOperators.S2I, IntVector.SPECIES_256, 0);
    IntVector rightOddInts =
        (IntVector) rightOdd.convertShape(VectorOperators.S2I, IntVector.SPECIES_256, 0);
    return leftEvenInts.mul(rightEvenInts).add(leftOddInts.mul(rightOddInts));
  }

  private static ShortVector multiplyAddUnsignedSignedBytes128(
      ByteVector unsigned, ByteVector signed) {
    ByteVector unsignedEven = unsigned.rearrange(BYTE_128_EVEN_LANES);
    ByteVector unsignedOdd = unsigned.rearrange(BYTE_128_ODD_LANES);
    ByteVector signedEven = signed.rearrange(BYTE_128_EVEN_LANES);
    ByteVector signedOdd = signed.rearrange(BYTE_128_ODD_LANES);
    ShortVector unsignedEvenShorts =
        (ShortVector)
            unsignedEven.convertShape(VectorOperators.ZERO_EXTEND_B2S, ShortVector.SPECIES_128, 0);
    ShortVector unsignedOddShorts =
        (ShortVector)
            unsignedOdd.convertShape(VectorOperators.ZERO_EXTEND_B2S, ShortVector.SPECIES_128, 0);
    ShortVector signedEvenShorts =
        (ShortVector) signedEven.convertShape(VectorOperators.B2S, ShortVector.SPECIES_128, 0);
    ShortVector signedOddShorts =
        (ShortVector) signedOdd.convertShape(VectorOperators.B2S, ShortVector.SPECIES_128, 0);
    return unsignedEvenShorts
        .mul(signedEvenShorts)
        .lanewise(VectorOperators.SADD, unsignedOddShorts.mul(signedOddShorts));
  }

  private static IntVector multiplyAddSignedShorts128(ShortVector left, ShortVector right) {
    ShortVector leftEven = left.rearrange(SHORT_128_EVEN_LANES);
    ShortVector leftOdd = left.rearrange(SHORT_128_ODD_LANES);
    ShortVector rightEven = right.rearrange(SHORT_128_EVEN_LANES);
    ShortVector rightOdd = right.rearrange(SHORT_128_ODD_LANES);
    IntVector leftEvenInts =
        (IntVector) leftEven.convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 0);
    IntVector leftOddInts =
        (IntVector) leftOdd.convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 0);
    IntVector rightEvenInts =
        (IntVector) rightEven.convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 0);
    IntVector rightOddInts =
        (IntVector) rightOdd.convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 0);
    return leftEvenInts.mul(rightEvenInts).add(leftOddInts.mul(rightOddInts));
  }

  private static ShortVector sumGroupsOfFour(ShortVector products) {
    ShortVector pairs = products.add(products.rearrange(SWAP_ADJACENT_SHORTS));
    return pairs.add(pairs.rearrange(SWAP_SHORT_PAIRS));
  }

  private static IntVector fourProductLanes(ShortVector first, ShortVector second) {
    ShortVector lowGroups = sumGroupsOfFour(first).rearrange(SELECT_LOW_GROUPS);
    ShortVector highGroups = sumGroupsOfFour(second).rearrange(SELECT_HIGH_GROUPS);
    ShortVector groups = lowGroups.blend(highGroups, HIGH_GROUP_LANES);
    return (IntVector) groups.convertShape(VectorOperators.S2I, IntVector.SPECIES_256, 0);
  }

  /** SIMD Q4_K by Q8_K GEMV with one activation quantization shared by all rows. */
  @Override
  public void ggufQ4_KQ8_KMatVecDot(
      float[] query,
      MemorySegment qWeight,
      int rows,
      int cols,
      float[] out,
      byte[] q8Quants,
      float[] q8Scales,
      short[] q8Sums) {
    GgufQuantizationSupport.quantizeQ8_K(query, cols, q8Quants, q8Scales, q8Sums);

    long rowBytes = (long) (cols / GGUF_Q4_K_BLOCK_SIZE) * GGUF_Q4_K_BLOCK_BYTES;
    if (qWeight.isMapped() && PanamaConstants.USE_MAPPED_Q4_K_LONG_OFFSETS) {
      GgufParallelSupport.forEachRow(
          qWeight,
          rows,
          cols,
          row ->
              out[row] =
                  ggufQ4_KQ8_KLongOffsetRowDot(
                      qWeight, row * rowBytes, rowBytes, q8Quants, q8Scales, q8Sums));
      return;
    }

    int blocks = cols / GGUF_Q4_K_BLOCK_SIZE;
    GgufParallelSupport.forEachRow(
        qWeight,
        rows,
        cols,
        row -> {
          float sum = 0.0f;
          long rowOffset = row * rowBytes;
          for (int block = 0; block < blocks; block++) {
            long blockOffset = rowOffset + (long) block * GGUF_Q4_K_BLOCK_BYTES;
            float q8Scale = q8Scales[block];
            float d = Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset)) * q8Scale;
            float dMin =
                Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset + Short.BYTES))
                    * q8Scale;
            long scalesOffset = blockOffset + GGUF_Q4_K_SCALES_OFFSET;
            long quantsOffset = blockOffset + GGUF_Q4_K_QUANTS_OFFSET;
            int activationOffset = block * GGUF_Q4_K_BLOCK_SIZE;
            int quantizedSum = 0;
            int minimumSum = 0;

            for (int group = 0; group < 8; group++) {
              int scale = GgufQuantizationSupport.qKScale(qWeight, scalesOffset, group);
              int min = GgufQuantizationSupport.qKMin(qWeight, scalesOffset, group);
              long packedOffset = quantsOffset + (long) (group >>> 1) * 32;
              int shift = (group & 1) * 4;
              int groupActivationOffset = activationOffset + group * 32;
              int groupDot =
                  q4_KQ8_KIntegerDot(qWeight, packedOffset, shift, q8Quants, groupActivationOffset);
              quantizedSum += scale * groupDot;
              int sumOffset = groupActivationOffset / GGUF_Q8_K_SUM_BLOCK_SIZE;
              minimumSum += min * (q8Sums[sumOffset] + q8Sums[sumOffset + 1]);
            }

            sum = MathUtil.fma(d, quantizedSum, sum);
            sum = MathUtil.fma(-dMin, minimumSum, sum);
          }
          out[row] = sum;
        });
  }

  /** SIMD Q4_K by a batch of Q8_K activations with row-local weight reuse. */
  @Override
  public void ggufQ4_KQ8_KBatchedMatmul(
      float[] queries,
      MemorySegment qWeight,
      int batchSize,
      int rows,
      int cols,
      float[] out,
      byte[] q8Quants,
      float[] q8Scales,
      short[] q8Sums) {
    if (batchSize == 1) {
      ggufQ4_KQ8_KMatVecDot(queries, qWeight, rows, cols, out, q8Quants, q8Scales, q8Sums);
      return;
    }

    int blocks = cols / GGUF_Q4_K_BLOCK_SIZE;
    int sumsPerBatch = cols / GGUF_Q8_K_SUM_BLOCK_SIZE;
    for (int batch = 0; batch < batchSize; batch++) {
      GgufQuantizationSupport.quantizeQ8_K(
          queries,
          batch * cols,
          cols,
          q8Quants,
          batch * cols,
          q8Scales,
          batch * blocks,
          q8Sums,
          batch * sumsPerBatch);
    }

    long rowBytes = (long) blocks * GGUF_Q4_K_BLOCK_BYTES;
    if (qWeight.isMapped() && PanamaConstants.USE_MAPPED_Q4_K_LONG_OFFSETS) {
      ggufQ4_KQ8_KLongOffsetBatchedMatmul(
          qWeight, batchSize, rows, cols, out, q8Quants, q8Scales, q8Sums);
      return;
    }

    GgufParallelSupport.forEachRow(
        qWeight,
        rows,
        cols,
        row -> {
          for (int batch = 0; batch < batchSize; batch++) {
            out[batch * rows + row] = 0.0f;
          }

          long rowOffset = row * rowBytes;
          for (int block = 0; block < blocks; block++) {
            long blockOffset = rowOffset + (long) block * GGUF_Q4_K_BLOCK_BYTES;
            float weightScale = Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset));
            float weightMinScale =
                Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset + Short.BYTES));
            long scalesOffset = blockOffset + GGUF_Q4_K_SCALES_OFFSET;
            long quantsOffset = blockOffset + GGUF_Q4_K_QUANTS_OFFSET;
            int blockActivationOffset = block * GGUF_Q4_K_BLOCK_SIZE;

            for (int batch = 0; batch < batchSize; batch++) {
              int quantBatchOffset = batch * cols;
              int scaleBatchOffset = batch * blocks;
              int sumBatchOffset = batch * sumsPerBatch;
              float q8Scale = q8Scales[scaleBatchOffset + block];
              float d = weightScale * q8Scale;
              float dMin = weightMinScale * q8Scale;
              int quantizedSum = 0;
              int minimumSum = 0;

              for (int group = 0; group < 8; group++) {
                int scale = GgufQuantizationSupport.qKScale(qWeight, scalesOffset, group);
                int min = GgufQuantizationSupport.qKMin(qWeight, scalesOffset, group);
                long packedOffset = quantsOffset + (long) (group >>> 1) * 32;
                int shift = (group & 1) * 4;
                int groupActivationOffset = blockActivationOffset + group * 32;
                int groupDot =
                    q4_KQ8_KIntegerDot(
                        qWeight,
                        packedOffset,
                        shift,
                        q8Quants,
                        quantBatchOffset + groupActivationOffset);
                quantizedSum += scale * groupDot;
                int activationSumOffset =
                    sumBatchOffset + groupActivationOffset / GGUF_Q8_K_SUM_BLOCK_SIZE;
                minimumSum += min * (q8Sums[activationSumOffset] + q8Sums[activationSumOffset + 1]);
              }

              int outputOffset = batch * rows + row;
              float sum = out[outputOffset];
              sum = MathUtil.fma(d, quantizedSum, sum);
              sum = MathUtil.fma(-dMin, minimumSum, sum);
              out[outputOffset] = sum;
            }
          }
        });
  }

  static void ggufQ4_KQ8_KLongOffsetBatchedMatmul(
      MemorySegment qWeight,
      int batchSize,
      int rows,
      int cols,
      float[] out,
      byte[] q8Quants,
      float[] q8Scales,
      short[] q8Sums) {
    int blocks = cols / GGUF_Q4_K_BLOCK_SIZE;
    int sumsPerBatch = cols / GGUF_Q8_K_SUM_BLOCK_SIZE;
    long rowBytes = (long) blocks * GGUF_Q4_K_BLOCK_BYTES;
    GgufParallelSupport.forEachRow(
        qWeight,
        rows,
        cols,
        row -> {
          for (int batch = 0; batch < batchSize; batch++) {
            out[batch * rows + row] = 0.0f;
          }

          long rowEnd = (row + 1L) * rowBytes;
          int block = 0;
          for (long blockOffset = row * rowBytes;
              blockOffset < rowEnd;
              blockOffset += GGUF_Q4_K_BLOCK_BYTES, block++) {
            float weightScale = Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset));
            float weightMinScale =
                Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset + Short.BYTES));
            long scalesOffset = blockOffset + GGUF_Q4_K_SCALES_OFFSET;
            long quantsOffset = blockOffset + GGUF_Q4_K_QUANTS_OFFSET;
            int blockActivationOffset = block * GGUF_Q4_K_BLOCK_SIZE;

            for (int batch = 0; batch < batchSize; batch++) {
              int quantBatchOffset = batch * cols;
              int scaleBatchOffset = batch * blocks;
              int sumBatchOffset = batch * sumsPerBatch;
              float q8Scale = q8Scales[scaleBatchOffset + block];
              float d = weightScale * q8Scale;
              float dMin = weightMinScale * q8Scale;
              int quantizedSum = 0;
              int minimumSum = 0;

              for (int group = 0; group < 8; group++) {
                int scale = GgufQuantizationSupport.qKScale(qWeight, scalesOffset, group);
                int min = GgufQuantizationSupport.qKMin(qWeight, scalesOffset, group);
                long packedOffset = quantsOffset + (long) (group >>> 1) * 32;
                int shift = (group & 1) * 4;
                int groupActivationOffset = blockActivationOffset + group * 32;
                int groupDot =
                    q4_KQ8_KIntegerDot(
                        qWeight,
                        packedOffset,
                        shift,
                        q8Quants,
                        quantBatchOffset + groupActivationOffset);
                quantizedSum += scale * groupDot;
                int activationSumOffset =
                    sumBatchOffset + groupActivationOffset / GGUF_Q8_K_SUM_BLOCK_SIZE;
                minimumSum += min * (q8Sums[activationSumOffset] + q8Sums[activationSumOffset + 1]);
              }

              int outputOffset = batch * rows + row;
              float sum = out[outputOffset];
              sum = MathUtil.fma(d, quantizedSum, sum);
              sum = MathUtil.fma(-dMin, minimumSum, sum);
              out[outputOffset] = sum;
            }
          }
        });
  }

  @Override
  public void ggufQ4_KQ8_KDualBatchedMatmul(
      float[] queries,
      MemorySegment firstWeight,
      int firstRows,
      float[] firstOut,
      MemorySegment secondWeight,
      int secondRows,
      float[] secondOut,
      int batchSize,
      int cols,
      byte[] q8Quants,
      float[] q8Scales,
      short[] q8Sums) {
    if (batchSize == 1) {
      ggufQ4_KQ8_KDualMatVecDot(
          queries,
          firstWeight,
          firstRows,
          firstOut,
          secondWeight,
          secondRows,
          secondOut,
          cols,
          q8Quants,
          q8Scales,
          q8Sums);
      return;
    }

    int blocks = cols / GGUF_Q4_K_BLOCK_SIZE;
    int sumsPerBatch = cols / GGUF_Q8_K_SUM_BLOCK_SIZE;
    quantizeQ8_KBatch(queries, batchSize, cols, blocks, sumsPerBatch, q8Quants, q8Scales, q8Sums);
    long rowBytes = (long) blocks * GGUF_Q4_K_BLOCK_BYTES;
    int totalRows = Math.addExact(firstRows, secondRows);
    boolean useLongOffsets =
        firstWeight.isMapped()
            && secondWeight.isMapped()
            && PanamaConstants.USE_MAPPED_Q4_K_LONG_OFFSETS;
    GgufParallelSupport.forEachRow(
        firstWeight,
        secondWeight,
        totalRows,
        cols,
        row -> {
          boolean first = row < firstRows;
          MemorySegment weight = first ? firstWeight : secondWeight;
          int matrixRow = first ? row : row - firstRows;
          int matrixRows = first ? firstRows : secondRows;
          float[] out = first ? firstOut : secondOut;
          ggufQ4_KQ8_KBatchedRowDot(
              weight,
              matrixRow * rowBytes,
              useLongOffsets,
              batchSize,
              matrixRows,
              matrixRow,
              cols,
              blocks,
              sumsPerBatch,
              out,
              q8Quants,
              q8Scales,
              q8Sums);
        });
  }

  @Override
  public void ggufQ4_KQ8_KDualMatVecDot(
      float[] query,
      MemorySegment firstWeight,
      int firstRows,
      float[] firstOut,
      MemorySegment secondWeight,
      int secondRows,
      float[] secondOut,
      int cols,
      byte[] q8Quants,
      float[] q8Scales,
      short[] q8Sums) {
    ggufQ4_KQ8_KGroupedMatVecDot(
        query,
        firstWeight,
        firstRows,
        firstOut,
        secondWeight,
        secondRows,
        secondOut,
        secondWeight,
        0,
        secondOut,
        cols,
        q8Quants,
        q8Scales,
        q8Sums);
  }

  @Override
  public void ggufQ4_KQ8_KTripleMatVecDot(
      float[] query,
      MemorySegment firstWeight,
      int firstRows,
      float[] firstOut,
      MemorySegment secondWeight,
      int secondRows,
      float[] secondOut,
      MemorySegment thirdWeight,
      int thirdRows,
      float[] thirdOut,
      int cols,
      byte[] q8Quants,
      float[] q8Scales,
      short[] q8Sums) {
    ggufQ4_KQ8_KGroupedMatVecDot(
        query,
        firstWeight,
        firstRows,
        firstOut,
        secondWeight,
        secondRows,
        secondOut,
        thirdWeight,
        thirdRows,
        thirdOut,
        cols,
        q8Quants,
        q8Scales,
        q8Sums);
  }

  private static void ggufQ4_KQ8_KGroupedMatVecDot(
      float[] query,
      MemorySegment firstWeight,
      int firstRows,
      float[] firstOut,
      MemorySegment secondWeight,
      int secondRows,
      float[] secondOut,
      MemorySegment thirdWeight,
      int thirdRows,
      float[] thirdOut,
      int cols,
      byte[] q8Quants,
      float[] q8Scales,
      short[] q8Sums) {
    GgufQuantizationSupport.quantizeQ8_K(query, cols, q8Quants, q8Scales, q8Sums);

    long rowBytes = (long) (cols / GGUF_Q4_K_BLOCK_SIZE) * GGUF_Q4_K_BLOCK_BYTES;
    int blocks = cols / GGUF_Q4_K_BLOCK_SIZE;
    int secondStart = firstRows;
    int thirdStart = Math.addExact(firstRows, secondRows);
    int totalRows = Math.addExact(thirdStart, thirdRows);
    if (firstWeight.isMapped()
        && secondWeight.isMapped()
        && thirdWeight.isMapped()
        && PanamaConstants.USE_MAPPED_Q4_K_LONG_OFFSETS) {
      GgufParallelSupport.forEachRow(
          firstWeight,
          secondWeight,
          thirdWeight,
          totalRows,
          cols,
          row -> {
            if (row < secondStart) {
              firstOut[row] =
                  ggufQ4_KQ8_KLongOffsetRowDot(
                      firstWeight, row * rowBytes, rowBytes, q8Quants, q8Scales, q8Sums);
            } else if (row < thirdStart) {
              int matrixRow = row - secondStart;
              secondOut[matrixRow] =
                  ggufQ4_KQ8_KLongOffsetRowDot(
                      secondWeight, matrixRow * rowBytes, rowBytes, q8Quants, q8Scales, q8Sums);
            } else {
              int matrixRow = row - thirdStart;
              thirdOut[matrixRow] =
                  ggufQ4_KQ8_KLongOffsetRowDot(
                      thirdWeight, matrixRow * rowBytes, rowBytes, q8Quants, q8Scales, q8Sums);
            }
          });
      return;
    }

    GgufParallelSupport.forEachRow(
        firstWeight,
        secondWeight,
        thirdWeight,
        totalRows,
        cols,
        row -> {
          MemorySegment qWeight;
          float[] out;
          int matrixRow;
          if (row < secondStart) {
            qWeight = firstWeight;
            out = firstOut;
            matrixRow = row;
          } else if (row < thirdStart) {
            qWeight = secondWeight;
            out = secondOut;
            matrixRow = row - secondStart;
          } else {
            qWeight = thirdWeight;
            out = thirdOut;
            matrixRow = row - thirdStart;
          }

          float sum = 0.0f;
          long rowOffset = matrixRow * rowBytes;
          for (int block = 0; block < blocks; block++) {
            long blockOffset = rowOffset + (long) block * GGUF_Q4_K_BLOCK_BYTES;
            float q8Scale = q8Scales[block];
            float d = Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset)) * q8Scale;
            float dMin =
                Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset + Short.BYTES))
                    * q8Scale;
            long scalesOffset = blockOffset + GGUF_Q4_K_SCALES_OFFSET;
            long quantsOffset = blockOffset + GGUF_Q4_K_QUANTS_OFFSET;
            int activationOffset = block * GGUF_Q4_K_BLOCK_SIZE;
            int quantizedSum = 0;
            int minimumSum = 0;

            for (int group = 0; group < 8; group++) {
              int scale = GgufQuantizationSupport.qKScale(qWeight, scalesOffset, group);
              int min = GgufQuantizationSupport.qKMin(qWeight, scalesOffset, group);
              long packedOffset = quantsOffset + (long) (group >>> 1) * 32;
              int shift = (group & 1) * 4;
              int groupActivationOffset = activationOffset + group * 32;
              int groupDot =
                  q4_KQ8_KIntegerDot(qWeight, packedOffset, shift, q8Quants, groupActivationOffset);
              quantizedSum += scale * groupDot;
              int sumOffset = groupActivationOffset / GGUF_Q8_K_SUM_BLOCK_SIZE;
              minimumSum += min * (q8Sums[sumOffset] + q8Sums[sumOffset + 1]);
            }

            sum = MathUtil.fma(d, quantizedSum, sum);
            sum = MathUtil.fma(-dMin, minimumSum, sum);
          }
          out[matrixRow] = sum;
        });
  }

  private static float ggufQ4_KQ8_KVectorRowDot(
      MemorySegment qWeight,
      long rowOffset,
      int blocks,
      byte[] q8Quants,
      float[] q8Scales,
      short[] q8Sums) {
    float sum = 0.0f;
    for (int block = 0; block < blocks; block++) {
      long blockOffset = rowOffset + (long) block * GGUF_Q4_K_BLOCK_BYTES;
      float q8Scale = q8Scales[block];
      float d = Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset)) * q8Scale;
      float dMin =
          Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset + Short.BYTES)) * q8Scale;
      long scalesOffset = blockOffset + GGUF_Q4_K_SCALES_OFFSET;
      long quantsOffset = blockOffset + GGUF_Q4_K_QUANTS_OFFSET;
      int activationOffset = block * GGUF_Q4_K_BLOCK_SIZE;
      int quantizedSum = 0;
      int minimumSum = 0;

      for (int group = 0; group < 8; group++) {
        int scale = GgufQuantizationSupport.qKScale(qWeight, scalesOffset, group);
        int min = GgufQuantizationSupport.qKMin(qWeight, scalesOffset, group);
        long packedOffset = quantsOffset + (long) (group >>> 1) * 32;
        int shift = (group & 1) * 4;
        int groupActivationOffset = activationOffset + group * 32;
        int groupDot =
            q4_KQ8_KIntegerDot(qWeight, packedOffset, shift, q8Quants, groupActivationOffset);
        quantizedSum += scale * groupDot;
        int sumOffset = groupActivationOffset / GGUF_Q8_K_SUM_BLOCK_SIZE;
        minimumSum += min * (q8Sums[sumOffset] + q8Sums[sumOffset + 1]);
      }

      sum = MathUtil.fma(d, quantizedSum, sum);
      sum = MathUtil.fma(-dMin, minimumSum, sum);
    }
    return sum;
  }

  static float ggufQ4_KQ8_KLongOffsetRowDot(
      MemorySegment qWeight,
      long rowOffset,
      long rowBytes,
      byte[] q8Quants,
      float[] q8Scales,
      short[] q8Sums) {
    float sum = 0.0f;
    int block = 0;
    long rowLimit = rowOffset + rowBytes;
    for (long blockOffset = rowOffset;
        blockOffset < rowLimit;
        blockOffset += GGUF_Q4_K_BLOCK_BYTES, block++) {
      float q8Scale = q8Scales[block];
      float d = Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset)) * q8Scale;
      float dMin =
          Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset + Short.BYTES)) * q8Scale;
      long scalesOffset = blockOffset + GGUF_Q4_K_SCALES_OFFSET;
      long quantsOffset = blockOffset + GGUF_Q4_K_QUANTS_OFFSET;
      int activationOffset = block * GGUF_Q4_K_BLOCK_SIZE;
      int quantizedSum = 0;
      int minimumSum = 0;

      for (int group = 0; group < 8; group++) {
        int scale = GgufQuantizationSupport.qKScale(qWeight, scalesOffset, group);
        int min = GgufQuantizationSupport.qKMin(qWeight, scalesOffset, group);
        long packedOffset = quantsOffset + (long) (group >>> 1) * 32;
        int shift = (group & 1) * 4;
        int groupActivationOffset = activationOffset + group * 32;
        int groupDot =
            q4_KQ8_KIntegerDot(qWeight, packedOffset, shift, q8Quants, groupActivationOffset);
        quantizedSum += scale * groupDot;
        int sumOffset = groupActivationOffset / GGUF_Q8_K_SUM_BLOCK_SIZE;
        minimumSum += min * (q8Sums[sumOffset] + q8Sums[sumOffset + 1]);
      }

      sum = MathUtil.fma(d, quantizedSum, sum);
      sum = MathUtil.fma(-dMin, minimumSum, sum);
    }
    return sum;
  }

  @Override
  public void ggufQ4_KQ4_KQ6_KQ8_KTripleMatVecDot(
      float[] query,
      MemorySegment firstWeight,
      int firstRows,
      float[] firstOut,
      MemorySegment secondWeight,
      int secondRows,
      float[] secondOut,
      MemorySegment thirdWeight,
      int thirdRows,
      float[] thirdOut,
      int cols,
      byte[] q8Quants,
      float[] q8Scales,
      short[] q8Sums) {
    if (VECTOR_BITSIZE < 256) {
      VectorUtilSupport.super.ggufQ4_KQ4_KQ6_KQ8_KTripleMatVecDot(
          query,
          firstWeight,
          firstRows,
          firstOut,
          secondWeight,
          secondRows,
          secondOut,
          thirdWeight,
          thirdRows,
          thirdOut,
          cols,
          q8Quants,
          q8Scales,
          q8Sums);
      return;
    }

    GgufQuantizationSupport.quantizeQ8_K(query, cols, q8Quants, q8Scales, q8Sums);
    int blocks = cols / GGUF_Q4_K_BLOCK_SIZE;
    long q4RowBytes = (long) blocks * GGUF_Q4_K_BLOCK_BYTES;
    long q6RowBytes = (long) blocks * GGUF_Q6_K_BLOCK_BYTES;
    int secondStart = firstRows;
    int thirdStart = Math.addExact(firstRows, secondRows);
    int totalRows = Math.addExact(thirdStart, thirdRows);
    if (firstWeight.isMapped()
        && secondWeight.isMapped()
        && thirdWeight.isMapped()
        && PanamaConstants.USE_MAPPED_Q4_K_LONG_OFFSETS) {
      GgufParallelSupport.forEachRow(
          firstWeight,
          secondWeight,
          thirdWeight,
          totalRows,
          cols,
          row -> {
            if (row < secondStart) {
              firstOut[row] =
                  ggufQ4_KQ8_KLongOffsetRowDot(
                      firstWeight, row * q4RowBytes, q4RowBytes, q8Quants, q8Scales, q8Sums);
            } else if (row < thirdStart) {
              int matrixRow = row - secondStart;
              secondOut[matrixRow] =
                  ggufQ4_KQ8_KLongOffsetRowDot(
                      secondWeight, matrixRow * q4RowBytes, q4RowBytes, q8Quants, q8Scales, q8Sums);
            } else {
              int matrixRow = row - thirdStart;
              thirdOut[matrixRow] =
                  ggufQ6_KQ8_KLongOffsetRowDot(
                      thirdWeight, matrixRow * q6RowBytes, q6RowBytes, q8Quants, q8Scales);
            }
          });
      return;
    }

    GgufParallelSupport.forEachRow(
        firstWeight,
        secondWeight,
        thirdWeight,
        totalRows,
        cols,
        row -> {
          if (row < secondStart) {
            firstOut[row] =
                ggufQ4_KQ8_KVectorRowDot(
                    firstWeight, row * q4RowBytes, blocks, q8Quants, q8Scales, q8Sums);
          } else if (row < thirdStart) {
            int matrixRow = row - secondStart;
            secondOut[matrixRow] =
                ggufQ4_KQ8_KVectorRowDot(
                    secondWeight, matrixRow * q4RowBytes, blocks, q8Quants, q8Scales, q8Sums);
          } else {
            int matrixRow = row - thirdStart;
            thirdOut[matrixRow] =
                ggufQ6_KQ8_KVectorRowDot(
                    thirdWeight, matrixRow * q6RowBytes, blocks, q8Quants, q8Scales);
          }
        });
  }

  @Override
  public void ggufQ4_KQ4_KQ6_KQ8_KTripleBatchedMatmul(
      float[] queries,
      MemorySegment firstWeight,
      int firstRows,
      float[] firstOut,
      MemorySegment secondWeight,
      int secondRows,
      float[] secondOut,
      MemorySegment thirdWeight,
      int thirdRows,
      float[] thirdOut,
      int batchSize,
      int cols,
      byte[] q8Quants,
      float[] q8Scales,
      short[] q8Sums) {
    if (batchSize == 1) {
      ggufQ4_KQ4_KQ6_KQ8_KTripleMatVecDot(
          queries,
          firstWeight,
          firstRows,
          firstOut,
          secondWeight,
          secondRows,
          secondOut,
          thirdWeight,
          thirdRows,
          thirdOut,
          cols,
          q8Quants,
          q8Scales,
          q8Sums);
      return;
    }
    if (VECTOR_BITSIZE < 256) {
      VectorUtilSupport.super.ggufQ4_KQ4_KQ6_KQ8_KTripleBatchedMatmul(
          queries,
          firstWeight,
          firstRows,
          firstOut,
          secondWeight,
          secondRows,
          secondOut,
          thirdWeight,
          thirdRows,
          thirdOut,
          batchSize,
          cols,
          q8Quants,
          q8Scales,
          q8Sums);
      return;
    }

    int blocks = cols / GGUF_Q4_K_BLOCK_SIZE;
    int sumsPerBatch = cols / GGUF_Q8_K_SUM_BLOCK_SIZE;
    quantizeQ8_KBatch(queries, batchSize, cols, blocks, sumsPerBatch, q8Quants, q8Scales, q8Sums);
    long q4RowBytes = (long) blocks * GGUF_Q4_K_BLOCK_BYTES;
    long q6RowBytes = (long) blocks * GGUF_Q6_K_BLOCK_BYTES;
    int secondStart = firstRows;
    int thirdStart = Math.addExact(firstRows, secondRows);
    int totalRows = Math.addExact(thirdStart, thirdRows);
    boolean useLongQ4Offsets =
        firstWeight.isMapped()
            && secondWeight.isMapped()
            && PanamaConstants.USE_MAPPED_Q4_K_LONG_OFFSETS;
    GgufParallelSupport.forEachRow(
        firstWeight,
        secondWeight,
        thirdWeight,
        totalRows,
        cols,
        row -> {
          if (row < secondStart) {
            ggufQ4_KQ8_KBatchedRowDot(
                firstWeight,
                row * q4RowBytes,
                useLongQ4Offsets,
                batchSize,
                firstRows,
                row,
                cols,
                blocks,
                sumsPerBatch,
                firstOut,
                q8Quants,
                q8Scales,
                q8Sums);
          } else if (row < thirdStart) {
            int matrixRow = row - secondStart;
            ggufQ4_KQ8_KBatchedRowDot(
                secondWeight,
                matrixRow * q4RowBytes,
                useLongQ4Offsets,
                batchSize,
                secondRows,
                matrixRow,
                cols,
                blocks,
                sumsPerBatch,
                secondOut,
                q8Quants,
                q8Scales,
                q8Sums);
          } else {
            int matrixRow = row - thirdStart;
            ggufQ6_KQ8_KBatchedRowDot(
                thirdWeight,
                matrixRow * q6RowBytes,
                batchSize,
                thirdRows,
                matrixRow,
                cols,
                blocks,
                thirdOut,
                q8Quants,
                q8Scales);
          }
        });
  }

  private static void quantizeQ8_KBatch(
      float[] queries,
      int batchSize,
      int cols,
      int blocks,
      int sumsPerBatch,
      byte[] q8Quants,
      float[] q8Scales,
      short[] q8Sums) {
    for (int batch = 0; batch < batchSize; batch++) {
      GgufQuantizationSupport.quantizeQ8_K(
          queries,
          batch * cols,
          cols,
          q8Quants,
          batch * cols,
          q8Scales,
          batch * blocks,
          q8Sums,
          batch * sumsPerBatch);
    }
  }

  private static void ggufQ4_KQ8_KBatchedRowDot(
      MemorySegment qWeight,
      long rowOffset,
      boolean useLongOffsets,
      int batchSize,
      int rows,
      int row,
      int cols,
      int blocks,
      int sumsPerBatch,
      float[] out,
      byte[] q8Quants,
      float[] q8Scales,
      short[] q8Sums) {
    for (int batch = 0; batch < batchSize; batch++) {
      out[batch * rows + row] = 0.0f;
    }

    long blockOffset = rowOffset;
    for (int block = 0; block < blocks; block++) {
      if (!useLongOffsets) {
        blockOffset = rowOffset + (long) block * GGUF_Q4_K_BLOCK_BYTES;
      }
      float weightScale = Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset));
      float weightMinScale =
          Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset + Short.BYTES));
      long scalesOffset = blockOffset + GGUF_Q4_K_SCALES_OFFSET;
      long quantsOffset = blockOffset + GGUF_Q4_K_QUANTS_OFFSET;
      int blockActivationOffset = block * GGUF_Q4_K_BLOCK_SIZE;

      for (int batch = 0; batch < batchSize; batch++) {
        int quantBatchOffset = batch * cols;
        int scaleBatchOffset = batch * blocks;
        int sumBatchOffset = batch * sumsPerBatch;
        float q8Scale = q8Scales[scaleBatchOffset + block];
        float d = weightScale * q8Scale;
        float dMin = weightMinScale * q8Scale;
        int quantizedSum = 0;
        int minimumSum = 0;

        for (int group = 0; group < 8; group++) {
          int scale = GgufQuantizationSupport.qKScale(qWeight, scalesOffset, group);
          int min = GgufQuantizationSupport.qKMin(qWeight, scalesOffset, group);
          long packedOffset = quantsOffset + (long) (group >>> 1) * 32;
          int shift = (group & 1) * 4;
          int groupActivationOffset = blockActivationOffset + group * 32;
          int groupDot =
              q4_KQ8_KIntegerDot(
                  qWeight, packedOffset, shift, q8Quants, quantBatchOffset + groupActivationOffset);
          quantizedSum += scale * groupDot;
          int activationSumOffset =
              sumBatchOffset + groupActivationOffset / GGUF_Q8_K_SUM_BLOCK_SIZE;
          minimumSum += min * (q8Sums[activationSumOffset] + q8Sums[activationSumOffset + 1]);
        }

        int outputOffset = batch * rows + row;
        float sum = out[outputOffset];
        sum = MathUtil.fma(d, quantizedSum, sum);
        sum = MathUtil.fma(-dMin, minimumSum, sum);
        out[outputOffset] = sum;
      }
      if (useLongOffsets) {
        blockOffset += GGUF_Q4_K_BLOCK_BYTES;
      }
    }
  }

  private static void ggufQ6_KQ8_KBatchedRowDot(
      MemorySegment qWeight,
      long rowOffset,
      int batchSize,
      int rows,
      int row,
      int cols,
      int blocks,
      float[] out,
      byte[] q8Quants,
      float[] q8Scales) {
    for (int batch = 0; batch < batchSize; batch++) {
      out[batch * rows + row] = 0.0f;
    }

    for (int block = 0; block < blocks; block++) {
      long blockOffset = rowOffset + (long) block * GGUF_Q6_K_BLOCK_BYTES;
      float weightScale =
          Float.float16ToFloat(
              qWeight.get(
                  GGUF_LE_SHORT,
                  blockOffset + GGUF_Q6_K_QL_BYTES + GGUF_Q6_K_QH_BYTES + GGUF_Q6_K_SCALES));
      long qlOffset = blockOffset;
      long qhOffset = blockOffset + GGUF_Q6_K_QL_BYTES;
      long scaleOffset = qhOffset + GGUF_Q6_K_QH_BYTES;
      int blockActivationOffset = block * GGUF_Q6_K_BLOCK_SIZE;

      for (int batch = 0; batch < batchSize; batch++) {
        int quantBatchOffset = batch * cols;
        int scaleBatchOffset = batch * blocks;
        int blockSum = 0;

        for (int superBlock = 0; superBlock < 2; superBlock++) {
          long qlBase = qlOffset + (long) superBlock * 64;
          long qhBase = qhOffset + (long) superBlock * 32;
          long scaleBase = scaleOffset + (long) superBlock * 8;
          int quantBase = quantBatchOffset + blockActivationOffset + superBlock * 128;
          for (int chunk = 0; chunk < 32; chunk += 16) {
            int scaleIndex = chunk / 16;
            int s1 = qWeight.get(ValueLayout.JAVA_BYTE, scaleBase + scaleIndex);
            int s2 = qWeight.get(ValueLayout.JAVA_BYTE, scaleBase + scaleIndex + 2L);
            int s3 = qWeight.get(ValueLayout.JAVA_BYTE, scaleBase + scaleIndex + 4L);
            int s4 = qWeight.get(ValueLayout.JAVA_BYTE, scaleBase + scaleIndex + 6L);
            blockSum +=
                q6_KQ8_KIntegerDot(
                    qWeight,
                    qlBase + chunk,
                    qlBase + 32L + chunk,
                    qhBase + chunk,
                    q8Quants,
                    quantBase + chunk,
                    s1,
                    s2,
                    s3,
                    s4);
          }
        }

        int outputOffset = batch * rows + row;
        float d = weightScale * q8Scales[scaleBatchOffset + block];
        out[outputOffset] = MathUtil.fma(d, blockSum, out[outputOffset]);
      }
    }
  }

  static int q4_KQ8_KIntegerDot(
      MemorySegment qWeight, long packedOffset, int shift, byte[] q8Quants, int quantOffset) {
    return switch (shift) {
      case 0 -> q4_KQ8_KLowIntegerDot(qWeight, packedOffset, q8Quants, quantOffset);
      case 4 -> q4_KQ8_KHighIntegerDot(qWeight, packedOffset, q8Quants, quantOffset);
      default -> throw new IllegalArgumentException("Q4_K shift must be 0 or 4: " + shift);
    };
  }

  private static int q4_KQ8_KLowIntegerDot(
      MemorySegment qWeight, long packedOffset, byte[] q8Quants, int quantOffset) {
    if (VECTOR_BITSIZE >= 256) {
      int sum = 0;
      for (int index = 0; index < 32; index += 16) {
        ByteVector packed =
            ByteVector.fromMemorySegment(
                ByteVector.SPECIES_128, qWeight, packedOffset + index, ByteOrder.LITTLE_ENDIAN);
        ByteVector q4 = packed.and((byte) 0x0F);
        ShortVector q4Shorts =
            (ShortVector) q4.convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0);
        ShortVector q8Shorts =
            (ShortVector)
                ByteVector.fromArray(ByteVector.SPECIES_128, q8Quants, quantOffset + index)
                    .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0);
        sum += q4Shorts.mul(q8Shorts).reduceLanes(VectorOperators.ADD);
      }
      return sum;
    }

    if (VECTOR_BITSIZE >= 128) {
      int sum = 0;
      for (int index = 0; index < 32; index += 8) {
        ByteVector packed =
            ByteVector.fromMemorySegment(
                ByteVector.SPECIES_64, qWeight, packedOffset + index, ByteOrder.LITTLE_ENDIAN);
        ByteVector q4 = packed.and((byte) 0x0F);
        ShortVector q4Shorts =
            (ShortVector) q4.convertShape(VectorOperators.B2S, ShortVector.SPECIES_128, 0);
        ShortVector q8Shorts =
            (ShortVector)
                ByteVector.fromArray(ByteVector.SPECIES_64, q8Quants, quantOffset + index)
                    .convertShape(VectorOperators.B2S, ShortVector.SPECIES_128, 0);
        sum += q4Shorts.mul(q8Shorts).reduceLanes(VectorOperators.ADD);
      }
      return sum;
    }

    int sum = 0;
    for (int index = 0; index < 32; index++) {
      int packed = qWeight.get(ValueLayout.JAVA_BYTE, packedOffset + index) & 0xFF;
      sum += (packed & 0x0F) * q8Quants[quantOffset + index];
    }
    return sum;
  }

  private static int q4_KQ8_KHighIntegerDot(
      MemorySegment qWeight, long packedOffset, byte[] q8Quants, int quantOffset) {
    if (VECTOR_BITSIZE >= 256) {
      int sum = 0;
      for (int index = 0; index < 32; index += 16) {
        ByteVector packed =
            ByteVector.fromMemorySegment(
                ByteVector.SPECIES_128, qWeight, packedOffset + index, ByteOrder.LITTLE_ENDIAN);
        ByteVector q4 = packed.lanewise(VectorOperators.LSHR, 4).and((byte) 0x0F);
        ShortVector q4Shorts =
            (ShortVector) q4.convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0);
        ShortVector q8Shorts =
            (ShortVector)
                ByteVector.fromArray(ByteVector.SPECIES_128, q8Quants, quantOffset + index)
                    .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0);
        sum += q4Shorts.mul(q8Shorts).reduceLanes(VectorOperators.ADD);
      }
      return sum;
    }

    if (VECTOR_BITSIZE >= 128) {
      int sum = 0;
      for (int index = 0; index < 32; index += 8) {
        ByteVector packed =
            ByteVector.fromMemorySegment(
                ByteVector.SPECIES_64, qWeight, packedOffset + index, ByteOrder.LITTLE_ENDIAN);
        ByteVector q4 = packed.lanewise(VectorOperators.LSHR, 4).and((byte) 0x0F);
        ShortVector q4Shorts =
            (ShortVector) q4.convertShape(VectorOperators.B2S, ShortVector.SPECIES_128, 0);
        ShortVector q8Shorts =
            (ShortVector)
                ByteVector.fromArray(ByteVector.SPECIES_64, q8Quants, quantOffset + index)
                    .convertShape(VectorOperators.B2S, ShortVector.SPECIES_128, 0);
        sum += q4Shorts.mul(q8Shorts).reduceLanes(VectorOperators.ADD);
      }
      return sum;
    }

    int sum = 0;
    for (int index = 0; index < 32; index++) {
      int packed = qWeight.get(ValueLayout.JAVA_BYTE, packedOffset + index) & 0xFF;
      sum += ((packed >>> 4) & 0x0F) * q8Quants[quantOffset + index];
    }
    return sum;
  }

  /** SIMD Q5_K by Q8_K GEMV with one activation quantization shared by all rows. */
  @Override
  public void ggufQ5_KQ8_KMatVecDot(
      float[] query,
      MemorySegment qWeight,
      int rows,
      int cols,
      float[] out,
      byte[] q8Quants,
      float[] q8Scales,
      short[] q8Sums) {
    if (VECTOR_BITSIZE < 256) {
      VectorUtilSupport.super.ggufQ5_KQ8_KMatVecDot(
          query, qWeight, rows, cols, out, q8Quants, q8Scales, q8Sums);
      return;
    }

    GgufQuantizationSupport.quantizeQ8_K(query, cols, q8Quants, q8Scales, q8Sums);
    long rowBytes = (long) (cols / GGUF_Q5_K_BLOCK_SIZE) * GGUF_Q5_K_BLOCK_BYTES;
    if (qWeight.isMapped() && PanamaConstants.USE_MAPPED_Q5_K_LONG_OFFSETS) {
      GgufParallelSupport.forEachRow(
          qWeight,
          rows,
          cols,
          row ->
              out[row] =
                  ggufQ5_KQ8_KLongOffsetRowDot(
                      qWeight, row * rowBytes, rowBytes, q8Quants, q8Scales, q8Sums));
      return;
    }

    int blocks = cols / GGUF_Q5_K_BLOCK_SIZE;
    GgufParallelSupport.forEachRow(
        qWeight,
        rows,
        cols,
        row ->
            out[row] = q5_KQ8_KRowDot(qWeight, row * rowBytes, blocks, q8Quants, q8Scales, q8Sums));
  }

  /** SIMD Q5_K by a batch of Q8_K activations with row-local weight reuse. */
  @Override
  public void ggufQ5_KQ8_KBatchedMatmul(
      float[] queries,
      MemorySegment qWeight,
      int batchSize,
      int rows,
      int cols,
      float[] out,
      byte[] q8Quants,
      float[] q8Scales,
      short[] q8Sums) {
    if (batchSize == 1) {
      ggufQ5_KQ8_KMatVecDot(queries, qWeight, rows, cols, out, q8Quants, q8Scales, q8Sums);
      return;
    }
    if (VECTOR_BITSIZE < 256) {
      VectorUtilSupport.super.ggufQ5_KQ8_KBatchedMatmul(
          queries, qWeight, batchSize, rows, cols, out, q8Quants, q8Scales, q8Sums);
      return;
    }

    int blocks = cols / GGUF_Q5_K_BLOCK_SIZE;
    int sumsPerBatch = cols / GGUF_Q8_K_SUM_BLOCK_SIZE;
    for (int batch = 0; batch < batchSize; batch++) {
      GgufQuantizationSupport.quantizeQ8_K(
          queries,
          batch * cols,
          cols,
          q8Quants,
          batch * cols,
          q8Scales,
          batch * blocks,
          q8Sums,
          batch * sumsPerBatch);
    }

    long rowBytes = (long) blocks * GGUF_Q5_K_BLOCK_BYTES;
    GgufParallelSupport.forEachRow(
        qWeight,
        rows,
        cols,
        row -> {
          for (int batch = 0; batch < batchSize; batch++) {
            out[batch * rows + row] = 0.0f;
          }

          long rowOffset = row * rowBytes;
          for (int block = 0; block < blocks; block++) {
            long blockOffset = rowOffset + (long) block * GGUF_Q5_K_BLOCK_BYTES;
            float weightScale = Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset));
            float weightMinScale =
                Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset + Short.BYTES));
            long scalesOffset = blockOffset + GGUF_Q5_K_SCALES_OFFSET;
            long highBitsOffset = blockOffset + GGUF_Q5_K_HIGH_BITS_OFFSET;
            long quantsOffset = blockOffset + GGUF_Q5_K_QUANTS_OFFSET;
            int blockActivationOffset = block * GGUF_Q5_K_BLOCK_SIZE;

            for (int batch = 0; batch < batchSize; batch++) {
              int quantBatchOffset = batch * cols;
              int scaleBatchOffset = batch * blocks;
              int sumBatchOffset = batch * sumsPerBatch;
              float q8Scale = q8Scales[scaleBatchOffset + block];
              float d = weightScale * q8Scale;
              float dMin = weightMinScale * q8Scale;
              int quantizedSum = 0;
              int minimumSum = 0;

              for (int group = 0; group < 8; group++) {
                int scale = GgufQuantizationSupport.qKScale(qWeight, scalesOffset, group);
                int min = GgufQuantizationSupport.qKMin(qWeight, scalesOffset, group);
                long packedOffset = quantsOffset + (long) (group >>> 1) * 32;
                int shift = (group & 1) * 4;
                int highBit = 1 << group;
                int groupActivationOffset = blockActivationOffset + group * 32;
                int groupDot =
                    q5_KQ8_KIntegerDot(
                        qWeight,
                        packedOffset,
                        shift,
                        highBitsOffset,
                        highBit,
                        q8Quants,
                        quantBatchOffset + groupActivationOffset);
                quantizedSum += scale * groupDot;
                int activationSumOffset =
                    sumBatchOffset + groupActivationOffset / GGUF_Q8_K_SUM_BLOCK_SIZE;
                minimumSum += min * (q8Sums[activationSumOffset] + q8Sums[activationSumOffset + 1]);
              }

              int outputOffset = batch * rows + row;
              float sum = out[outputOffset];
              sum = MathUtil.fma(d, quantizedSum, sum);
              sum = MathUtil.fma(-dMin, minimumSum, sum);
              out[outputOffset] = sum;
            }
          }
        });
  }

  @Override
  public void ggufQ5_KQ8_KDualMatVecDot(
      float[] query,
      MemorySegment firstWeight,
      int firstRows,
      float[] firstOut,
      MemorySegment secondWeight,
      int secondRows,
      float[] secondOut,
      int cols,
      byte[] q8Quants,
      float[] q8Scales,
      short[] q8Sums) {
    if (VECTOR_BITSIZE < 256) {
      VectorUtilSupport.super.ggufQ5_KQ8_KDualMatVecDot(
          query,
          firstWeight,
          firstRows,
          firstOut,
          secondWeight,
          secondRows,
          secondOut,
          cols,
          q8Quants,
          q8Scales,
          q8Sums);
      return;
    }
    ggufQ5_KQ8_KGroupedMatVecDot(
        query,
        firstWeight,
        firstRows,
        firstOut,
        secondWeight,
        secondRows,
        secondOut,
        secondWeight,
        0,
        secondOut,
        cols,
        q8Quants,
        q8Scales,
        q8Sums);
  }

  @Override
  public void ggufQ5_KQ8_KTripleMatVecDot(
      float[] query,
      MemorySegment firstWeight,
      int firstRows,
      float[] firstOut,
      MemorySegment secondWeight,
      int secondRows,
      float[] secondOut,
      MemorySegment thirdWeight,
      int thirdRows,
      float[] thirdOut,
      int cols,
      byte[] q8Quants,
      float[] q8Scales,
      short[] q8Sums) {
    if (VECTOR_BITSIZE < 256) {
      VectorUtilSupport.super.ggufQ5_KQ8_KTripleMatVecDot(
          query,
          firstWeight,
          firstRows,
          firstOut,
          secondWeight,
          secondRows,
          secondOut,
          thirdWeight,
          thirdRows,
          thirdOut,
          cols,
          q8Quants,
          q8Scales,
          q8Sums);
      return;
    }
    ggufQ5_KQ8_KGroupedMatVecDot(
        query,
        firstWeight,
        firstRows,
        firstOut,
        secondWeight,
        secondRows,
        secondOut,
        thirdWeight,
        thirdRows,
        thirdOut,
        cols,
        q8Quants,
        q8Scales,
        q8Sums);
  }

  private static void ggufQ5_KQ8_KGroupedMatVecDot(
      float[] query,
      MemorySegment firstWeight,
      int firstRows,
      float[] firstOut,
      MemorySegment secondWeight,
      int secondRows,
      float[] secondOut,
      MemorySegment thirdWeight,
      int thirdRows,
      float[] thirdOut,
      int cols,
      byte[] q8Quants,
      float[] q8Scales,
      short[] q8Sums) {
    GgufQuantizationSupport.quantizeQ8_K(query, cols, q8Quants, q8Scales, q8Sums);

    long rowBytes = (long) (cols / GGUF_Q5_K_BLOCK_SIZE) * GGUF_Q5_K_BLOCK_BYTES;
    int blocks = cols / GGUF_Q5_K_BLOCK_SIZE;
    int secondStart = firstRows;
    int thirdStart = Math.addExact(firstRows, secondRows);
    int totalRows = Math.addExact(thirdStart, thirdRows);
    if (firstWeight.isMapped()
        && secondWeight.isMapped()
        && thirdWeight.isMapped()
        && PanamaConstants.USE_MAPPED_Q5_K_LONG_OFFSETS) {
      GgufParallelSupport.forEachRow(
          firstWeight,
          secondWeight,
          thirdWeight,
          totalRows,
          cols,
          row -> {
            if (row < secondStart) {
              firstOut[row] =
                  ggufQ5_KQ8_KLongOffsetRowDot(
                      firstWeight, row * rowBytes, rowBytes, q8Quants, q8Scales, q8Sums);
            } else if (row < thirdStart) {
              int matrixRow = row - secondStart;
              secondOut[matrixRow] =
                  ggufQ5_KQ8_KLongOffsetRowDot(
                      secondWeight, matrixRow * rowBytes, rowBytes, q8Quants, q8Scales, q8Sums);
            } else {
              int matrixRow = row - thirdStart;
              thirdOut[matrixRow] =
                  ggufQ5_KQ8_KLongOffsetRowDot(
                      thirdWeight, matrixRow * rowBytes, rowBytes, q8Quants, q8Scales, q8Sums);
            }
          });
      return;
    }

    GgufParallelSupport.forEachRow(
        firstWeight,
        secondWeight,
        thirdWeight,
        totalRows,
        cols,
        row -> {
          MemorySegment qWeight;
          float[] out;
          int matrixRow;
          if (row < secondStart) {
            qWeight = firstWeight;
            out = firstOut;
            matrixRow = row;
          } else if (row < thirdStart) {
            qWeight = secondWeight;
            out = secondOut;
            matrixRow = row - secondStart;
          } else {
            qWeight = thirdWeight;
            out = thirdOut;
            matrixRow = row - thirdStart;
          }

          out[matrixRow] =
              q5_KQ8_KRowDot(qWeight, matrixRow * rowBytes, blocks, q8Quants, q8Scales, q8Sums);
        });
  }

  private static float q5_KQ8_KRowDot(
      MemorySegment qWeight,
      long rowOffset,
      int blocks,
      byte[] q8Quants,
      float[] q8Scales,
      short[] q8Sums) {
    float sum = 0.0f;
    for (int block = 0; block < blocks; block++) {
      long blockOffset = rowOffset + (long) block * GGUF_Q5_K_BLOCK_BYTES;
      float q8Scale = q8Scales[block];
      float d = Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset)) * q8Scale;
      float dMin =
          Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset + Short.BYTES)) * q8Scale;
      long scalesOffset = blockOffset + GGUF_Q5_K_SCALES_OFFSET;
      long highBitsOffset = blockOffset + GGUF_Q5_K_HIGH_BITS_OFFSET;
      long quantsOffset = blockOffset + GGUF_Q5_K_QUANTS_OFFSET;
      int activationOffset = block * GGUF_Q5_K_BLOCK_SIZE;
      int quantizedSum = 0;
      int minimumSum = 0;

      for (int group = 0; group < 8; group++) {
        int scale = GgufQuantizationSupport.qKScale(qWeight, scalesOffset, group);
        int min = GgufQuantizationSupport.qKMin(qWeight, scalesOffset, group);
        long packedOffset = quantsOffset + (long) (group >>> 1) * 32;
        int shift = (group & 1) * 4;
        int highBit = 1 << group;
        int groupActivationOffset = activationOffset + group * 32;
        int groupDot =
            q5_KQ8_KIntegerDot(
                qWeight,
                packedOffset,
                shift,
                highBitsOffset,
                highBit,
                q8Quants,
                groupActivationOffset);
        quantizedSum += scale * groupDot;
        int sumOffset = groupActivationOffset / GGUF_Q8_K_SUM_BLOCK_SIZE;
        minimumSum += min * (q8Sums[sumOffset] + q8Sums[sumOffset + 1]);
      }

      sum = MathUtil.fma(d, quantizedSum, sum);
      sum = MathUtil.fma(-dMin, minimumSum, sum);
    }
    return sum;
  }

  static float ggufQ5_KQ8_KLongOffsetRowDot(
      MemorySegment qWeight,
      long rowOffset,
      long rowBytes,
      byte[] q8Quants,
      float[] q8Scales,
      short[] q8Sums) {
    float sum = 0.0f;
    int block = 0;
    long rowLimit = rowOffset + rowBytes;
    for (long blockOffset = rowOffset;
        blockOffset < rowLimit;
        blockOffset += GGUF_Q5_K_BLOCK_BYTES, block++) {
      float q8Scale = q8Scales[block];
      float d = Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset)) * q8Scale;
      float dMin =
          Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset + Short.BYTES)) * q8Scale;
      long scalesOffset = blockOffset + GGUF_Q5_K_SCALES_OFFSET;
      long highBitsOffset = blockOffset + GGUF_Q5_K_HIGH_BITS_OFFSET;
      long quantsOffset = blockOffset + GGUF_Q5_K_QUANTS_OFFSET;
      int activationOffset = block * GGUF_Q5_K_BLOCK_SIZE;
      int quantizedSum = 0;
      int minimumSum = 0;

      for (int group = 0; group < 8; group++) {
        int scale = GgufQuantizationSupport.qKScale(qWeight, scalesOffset, group);
        int min = GgufQuantizationSupport.qKMin(qWeight, scalesOffset, group);
        long packedOffset = quantsOffset + (long) (group >>> 1) * 32;
        int shift = (group & 1) * 4;
        int highBit = 1 << group;
        int groupActivationOffset = activationOffset + group * 32;
        int groupDot =
            q5_KQ8_KIntegerDot(
                qWeight,
                packedOffset,
                shift,
                highBitsOffset,
                highBit,
                q8Quants,
                groupActivationOffset);
        quantizedSum += scale * groupDot;
        int sumOffset = groupActivationOffset / GGUF_Q8_K_SUM_BLOCK_SIZE;
        minimumSum += min * (q8Sums[sumOffset] + q8Sums[sumOffset + 1]);
      }

      sum = MathUtil.fma(d, quantizedSum, sum);
      sum = MathUtil.fma(-dMin, minimumSum, sum);
    }
    return sum;
  }

  static int q5_KQ8_KIntegerDot(
      MemorySegment qWeight,
      long packedOffset,
      int shift,
      long highBitsOffset,
      int highBit,
      byte[] q8Quants,
      int quantOffset) {
    int sum = q4_KQ8_KIntegerDot(qWeight, packedOffset, shift, q8Quants, quantOffset);
    int highShift = Integer.numberOfTrailingZeros(highBit);
    if (VECTOR_BITSIZE >= 256) {
      for (int index = 0; index < 32; index += 16) {
        ByteVector highBits =
            ByteVector.fromMemorySegment(
                ByteVector.SPECIES_128, qWeight, highBitsOffset + index, ByteOrder.LITTLE_ENDIAN);
        ShortVector bits =
            (ShortVector)
                highBits
                    .lanewise(VectorOperators.LSHR, highShift)
                    .and((byte) 1)
                    .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0);
        ShortVector quants =
            (ShortVector)
                ByteVector.fromArray(ByteVector.SPECIES_128, q8Quants, quantOffset + index)
                    .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0);
        sum += 16 * bits.mul(quants).reduceLanes(VectorOperators.ADD);
      }
      return sum;
    }

    for (int index = 0; index < 32; index += 8) {
      ByteVector highBits =
          ByteVector.fromMemorySegment(
              ByteVector.SPECIES_64, qWeight, highBitsOffset + index, ByteOrder.LITTLE_ENDIAN);
      ShortVector bits =
          (ShortVector)
              highBits
                  .lanewise(VectorOperators.LSHR, highShift)
                  .and((byte) 1)
                  .convertShape(VectorOperators.B2S, ShortVector.SPECIES_128, 0);
      ShortVector quants =
          (ShortVector)
              ByteVector.fromArray(ByteVector.SPECIES_64, q8Quants, quantOffset + index)
                  .convertShape(VectorOperators.B2S, ShortVector.SPECIES_128, 0);
      sum += 16 * bits.mul(quants).reduceLanes(VectorOperators.ADD);
    }
    return sum;
  }

  /** SIMD Q5_0 by Q8_0 GEMV with one activation quantization shared by all rows. */
  @Override
  public void ggufQ5_0Q8_0MatVecDot(
      float[] query,
      MemorySegment qWeight,
      int rows,
      int cols,
      float[] out,
      byte[] q8Quants,
      float[] q8Scales) {
    GgufQuantizationSupport.quantizeQ8_0(query, cols, q8Quants, q8Scales);
    long rowBytes = (long) (cols / GGUF_Q_BLOCK_SIZE) * GGUF_Q5_0_BLOCK_BYTES;
    int blocks = cols / GGUF_Q_BLOCK_SIZE;
    GgufParallelSupport.forEachRow(
        qWeight,
        rows,
        cols,
        row -> {
          float sum = 0.0f;
          long rowOffset = row * rowBytes;
          for (int block = 0; block < blocks; block++) {
            long blockOffset = rowOffset + (long) block * GGUF_Q5_0_BLOCK_BYTES;
            float scale =
                Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset)) * q8Scales[block];
            int highBits = qWeight.get(GGUF_LE_INT, blockOffset + Short.BYTES);
            IntVector integerLanes =
                q5_0Q8_0IntegerLanes(
                    qWeight,
                    blockOffset + Short.BYTES + Integer.BYTES,
                    highBits,
                    q8Quants,
                    block * GGUF_Q_BLOCK_SIZE);
            sum = MathUtil.fma(scale, integerLanes.reduceLanes(VectorOperators.ADD), sum);
          }
          out[row] = sum;
        });
  }

  /** SIMD Q5_0 by a batch of Q8_0 activations with row-local weight reuse. */
  @Override
  public void ggufQ5_0Q8_0BatchedMatmul(
      float[] queries,
      MemorySegment qWeight,
      int batchSize,
      int rows,
      int cols,
      float[] out,
      byte[] q8Quants,
      float[] q8Scales) {
    if (batchSize == 1) {
      ggufQ5_0Q8_0MatVecDot(queries, qWeight, rows, cols, out, q8Quants, q8Scales);
      return;
    }

    int blocks = cols / GGUF_Q_BLOCK_SIZE;
    for (int batch = 0; batch < batchSize; batch++) {
      GgufQuantizationSupport.quantizeQ8_0(
          queries, batch * cols, cols, q8Quants, batch * cols, q8Scales, batch * blocks);
    }

    long rowBytes = (long) blocks * GGUF_Q5_0_BLOCK_BYTES;
    int effectiveCols = Math.multiplyExact(cols, batchSize);
    GgufParallelSupport.forEachRow(
        qWeight,
        rows,
        effectiveCols,
        row -> {
          for (int batch = 0; batch < batchSize; batch++) {
            out[batch * rows + row] = 0.0f;
          }

          long rowOffset = row * rowBytes;
          for (int block = 0; block < blocks; block++) {
            long blockOffset = rowOffset + (long) block * GGUF_Q5_0_BLOCK_BYTES;
            float weightScale = Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset));
            int highBits = qWeight.get(GGUF_LE_INT, blockOffset + Short.BYTES);
            long packedOffset = blockOffset + Short.BYTES + Integer.BYTES;
            int blockActivationOffset = block * GGUF_Q_BLOCK_SIZE;
            for (int batch = 0; batch < batchSize; batch++) {
              int integerSum =
                  q5_0Q8_0IntegerLanes(
                          qWeight,
                          packedOffset,
                          highBits,
                          q8Quants,
                          batch * cols + blockActivationOffset)
                      .reduceLanes(VectorOperators.ADD);
              int outputIndex = batch * rows + row;
              float scale = weightScale * q8Scales[batch * blocks + block];
              out[outputIndex] = MathUtil.fma(scale, integerSum, out[outputIndex]);
            }
          }
        });
  }

  static IntVector q5_0Q8_0IntegerLanes(
      MemorySegment qWeight, long packedOffset, int highBits, byte[] q8Quants, int quantOffset) {
    IntVector accumulator = IntVector.zero(IntVector.SPECIES_256);
    for (int index = 0; index < 16; index += 8) {
      ByteVector packed =
          ByteVector.fromMemorySegment(
              ByteVector.SPECIES_64, qWeight, packedOffset + index, ByteOrder.LITTLE_ENDIAN);
      ByteVector low = q5Values(packed.and((byte) 0x0F), highBits >>> index);
      ByteVector high =
          q5Values(
              packed.lanewise(VectorOperators.LSHR, 4).and((byte) 0x0F), highBits >>> (index + 16));
      IntVector lowWeights =
          (IntVector) low.convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0);
      IntVector highWeights =
          (IntVector) high.convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0);
      IntVector lowQuants =
          (IntVector)
              ByteVector.fromArray(ByteVector.SPECIES_64, q8Quants, quantOffset + index)
                  .convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0);
      IntVector highQuants =
          (IntVector)
              ByteVector.fromArray(ByteVector.SPECIES_64, q8Quants, quantOffset + index + 16)
                  .convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0);
      accumulator = accumulator.add(lowWeights.mul(lowQuants)).add(highWeights.mul(highQuants));
    }
    return accumulator;
  }

  private static ByteVector q5Values(ByteVector lowBits, int highBits) {
    VectorMask<Byte> highMask =
        Q5_HIGH_BIT_MASKS.and((byte) (highBits & 0xFF)).compare(VectorOperators.NE, (byte) 0);
    return lowBits.add((byte) 16, highMask).sub((byte) 16);
  }

  /** SIMD Q6_K by Q8_K GEMV with one activation quantization shared by all rows. */
  @Override
  public void ggufQ6_KQ8_KMatVecDot(
      float[] query,
      MemorySegment qWeight,
      int rows,
      int cols,
      float[] out,
      byte[] q8Quants,
      float[] q8Scales) {
    if (VECTOR_BITSIZE < 256) {
      VectorUtilSupport.super.ggufQ6_KQ8_KMatVecDot(
          query, qWeight, rows, cols, out, q8Quants, q8Scales);
      return;
    }

    GgufQuantizationSupport.quantizeQ8_K(query, cols, q8Quants, q8Scales, null);
    long rowBytes = (long) (cols / GGUF_Q6_K_BLOCK_SIZE) * GGUF_Q6_K_BLOCK_BYTES;
    if (qWeight.isMapped() && PanamaConstants.USE_MAPPED_Q6_K_LONG_OFFSETS) {
      GgufParallelSupport.forEachRow(
          qWeight,
          rows,
          cols,
          row ->
              out[row] =
                  ggufQ6_KQ8_KLongOffsetRowDot(
                      qWeight, row * rowBytes, rowBytes, q8Quants, q8Scales));
      return;
    }

    int blocks = cols / GGUF_Q6_K_BLOCK_SIZE;
    GgufParallelSupport.forEachRow(
        qWeight,
        rows,
        cols,
        row -> {
          float sum = 0.0f;
          long rowOffset = row * rowBytes;
          for (int block = 0; block < blocks; block++) {
            long blockOffset = rowOffset + (long) block * GGUF_Q6_K_BLOCK_BYTES;
            float d =
                Float.float16ToFloat(
                        qWeight.get(
                            GGUF_LE_SHORT,
                            blockOffset
                                + GGUF_Q6_K_QL_BYTES
                                + GGUF_Q6_K_QH_BYTES
                                + GGUF_Q6_K_SCALES))
                    * q8Scales[block];
            long qlOffset = blockOffset;
            long qhOffset = blockOffset + GGUF_Q6_K_QL_BYTES;
            long scaleOffset = qhOffset + GGUF_Q6_K_QH_BYTES;
            int activationOffset = block * GGUF_Q6_K_BLOCK_SIZE;
            int blockSum = 0;

            for (int superBlock = 0; superBlock < 2; superBlock++) {
              long qlBase = qlOffset + (long) superBlock * 64;
              long qhBase = qhOffset + (long) superBlock * 32;
              long scaleBase = scaleOffset + (long) superBlock * 8;
              int quantBase = activationOffset + superBlock * 128;
              for (int batch = 0; batch < 32; batch += 16) {
                int scaleIndex = batch / 16;
                int s1 = qWeight.get(ValueLayout.JAVA_BYTE, scaleBase + scaleIndex);
                int s2 = qWeight.get(ValueLayout.JAVA_BYTE, scaleBase + scaleIndex + 2L);
                int s3 = qWeight.get(ValueLayout.JAVA_BYTE, scaleBase + scaleIndex + 4L);
                int s4 = qWeight.get(ValueLayout.JAVA_BYTE, scaleBase + scaleIndex + 6L);
                blockSum +=
                    q6_KQ8_KIntegerDot(
                        qWeight,
                        qlBase + batch,
                        qlBase + 32L + batch,
                        qhBase + batch,
                        q8Quants,
                        quantBase + batch,
                        s1,
                        s2,
                        s3,
                        s4);
              }
            }

            sum = MathUtil.fma(d, blockSum, sum);
          }
          out[row] = sum;
        });
  }

  private static float ggufQ6_KQ8_KVectorRowDot(
      MemorySegment qWeight, long rowOffset, int blocks, byte[] q8Quants, float[] q8Scales) {
    float sum = 0.0f;
    for (int block = 0; block < blocks; block++) {
      long blockOffset = rowOffset + (long) block * GGUF_Q6_K_BLOCK_BYTES;
      float d =
          Float.float16ToFloat(
                  qWeight.get(
                      GGUF_LE_SHORT,
                      blockOffset + GGUF_Q6_K_QL_BYTES + GGUF_Q6_K_QH_BYTES + GGUF_Q6_K_SCALES))
              * q8Scales[block];
      long qlOffset = blockOffset;
      long qhOffset = blockOffset + GGUF_Q6_K_QL_BYTES;
      long scaleOffset = qhOffset + GGUF_Q6_K_QH_BYTES;
      int activationOffset = block * GGUF_Q6_K_BLOCK_SIZE;
      int blockSum = 0;

      for (int superBlock = 0; superBlock < 2; superBlock++) {
        long qlBase = qlOffset + (long) superBlock * 64;
        long qhBase = qhOffset + (long) superBlock * 32;
        long scaleBase = scaleOffset + (long) superBlock * 8;
        int quantBase = activationOffset + superBlock * 128;
        for (int batch = 0; batch < 32; batch += 16) {
          int scaleIndex = batch / 16;
          int s1 = qWeight.get(ValueLayout.JAVA_BYTE, scaleBase + scaleIndex);
          int s2 = qWeight.get(ValueLayout.JAVA_BYTE, scaleBase + scaleIndex + 2L);
          int s3 = qWeight.get(ValueLayout.JAVA_BYTE, scaleBase + scaleIndex + 4L);
          int s4 = qWeight.get(ValueLayout.JAVA_BYTE, scaleBase + scaleIndex + 6L);
          blockSum +=
              q6_KQ8_KIntegerDot(
                  qWeight,
                  qlBase + batch,
                  qlBase + 32L + batch,
                  qhBase + batch,
                  q8Quants,
                  quantBase + batch,
                  s1,
                  s2,
                  s3,
                  s4);
        }
      }

      sum = MathUtil.fma(d, blockSum, sum);
    }
    return sum;
  }

  static float ggufQ6_KQ8_KLongOffsetRowDot(
      MemorySegment qWeight, long rowOffset, long rowBytes, byte[] q8Quants, float[] q8Scales) {
    float sum = 0.0f;
    int block = 0;
    long rowLimit = rowOffset + rowBytes;
    for (long blockOffset = rowOffset;
        blockOffset < rowLimit;
        blockOffset += GGUF_Q6_K_BLOCK_BYTES, block++) {
      float d =
          Float.float16ToFloat(
                  qWeight.get(
                      GGUF_LE_SHORT,
                      blockOffset + GGUF_Q6_K_QL_BYTES + GGUF_Q6_K_QH_BYTES + GGUF_Q6_K_SCALES))
              * q8Scales[block];
      long qlOffset = blockOffset;
      long qhOffset = blockOffset + GGUF_Q6_K_QL_BYTES;
      long scaleOffset = qhOffset + GGUF_Q6_K_QH_BYTES;
      int activationOffset = block * GGUF_Q6_K_BLOCK_SIZE;
      int blockSum = 0;

      for (int superBlock = 0; superBlock < 2; superBlock++) {
        long qlBase = qlOffset + (long) superBlock * 64;
        long qhBase = qhOffset + (long) superBlock * 32;
        long scaleBase = scaleOffset + (long) superBlock * 8;
        int quantBase = activationOffset + superBlock * 128;
        for (int batch = 0; batch < 32; batch += 16) {
          int scaleIndex = batch / 16;
          int s1 = qWeight.get(ValueLayout.JAVA_BYTE, scaleBase + scaleIndex);
          int s2 = qWeight.get(ValueLayout.JAVA_BYTE, scaleBase + scaleIndex + 2L);
          int s3 = qWeight.get(ValueLayout.JAVA_BYTE, scaleBase + scaleIndex + 4L);
          int s4 = qWeight.get(ValueLayout.JAVA_BYTE, scaleBase + scaleIndex + 6L);
          blockSum +=
              q6_KQ8_KIntegerDot(
                  qWeight,
                  qlBase + batch,
                  qlBase + 32L + batch,
                  qhBase + batch,
                  q8Quants,
                  quantBase + batch,
                  s1,
                  s2,
                  s3,
                  s4);
        }
      }

      sum = MathUtil.fma(d, blockSum, sum);
    }
    return sum;
  }

  /** SIMD Q6_K by a batch of Q8_K activations with row-local weight reuse. */
  @Override
  public void ggufQ6_KQ8_KBatchedMatmul(
      float[] queries,
      MemorySegment qWeight,
      int batchSize,
      int rows,
      int cols,
      float[] out,
      byte[] q8Quants,
      float[] q8Scales) {
    if (batchSize == 1) {
      ggufQ6_KQ8_KMatVecDot(queries, qWeight, rows, cols, out, q8Quants, q8Scales);
      return;
    }
    if (VECTOR_BITSIZE < 256) {
      VectorUtilSupport.super.ggufQ6_KQ8_KBatchedMatmul(
          queries, qWeight, batchSize, rows, cols, out, q8Quants, q8Scales);
      return;
    }

    int blocks = cols / GGUF_Q6_K_BLOCK_SIZE;
    for (int batch = 0; batch < batchSize; batch++) {
      GgufQuantizationSupport.quantizeQ8_K(
          queries, batch * cols, cols, q8Quants, batch * cols, q8Scales, batch * blocks, null, 0);
    }

    long rowBytes = (long) blocks * GGUF_Q6_K_BLOCK_BYTES;
    GgufParallelSupport.forEachRow(
        qWeight,
        rows,
        cols,
        row -> {
          for (int batch = 0; batch < batchSize; batch++) {
            out[batch * rows + row] = 0.0f;
          }

          long rowOffset = row * rowBytes;
          for (int block = 0; block < blocks; block++) {
            long blockOffset = rowOffset + (long) block * GGUF_Q6_K_BLOCK_BYTES;
            float weightScale =
                Float.float16ToFloat(
                    qWeight.get(
                        GGUF_LE_SHORT,
                        blockOffset + GGUF_Q6_K_QL_BYTES + GGUF_Q6_K_QH_BYTES + GGUF_Q6_K_SCALES));
            long qlOffset = blockOffset;
            long qhOffset = blockOffset + GGUF_Q6_K_QL_BYTES;
            long scaleOffset = qhOffset + GGUF_Q6_K_QH_BYTES;
            int blockActivationOffset = block * GGUF_Q6_K_BLOCK_SIZE;

            for (int batch = 0; batch < batchSize; batch++) {
              int quantBatchOffset = batch * cols;
              int scaleBatchOffset = batch * blocks;
              int blockSum = 0;

              for (int superBlock = 0; superBlock < 2; superBlock++) {
                long qlBase = qlOffset + (long) superBlock * 64;
                long qhBase = qhOffset + (long) superBlock * 32;
                long scaleBase = scaleOffset + (long) superBlock * 8;
                int quantBase = quantBatchOffset + blockActivationOffset + superBlock * 128;
                for (int chunk = 0; chunk < 32; chunk += 16) {
                  int scaleIndex = chunk / 16;
                  int s1 = qWeight.get(ValueLayout.JAVA_BYTE, scaleBase + scaleIndex);
                  int s2 = qWeight.get(ValueLayout.JAVA_BYTE, scaleBase + scaleIndex + 2L);
                  int s3 = qWeight.get(ValueLayout.JAVA_BYTE, scaleBase + scaleIndex + 4L);
                  int s4 = qWeight.get(ValueLayout.JAVA_BYTE, scaleBase + scaleIndex + 6L);
                  blockSum +=
                      q6_KQ8_KIntegerDot(
                          qWeight,
                          qlBase + chunk,
                          qlBase + 32L + chunk,
                          qhBase + chunk,
                          q8Quants,
                          quantBase + chunk,
                          s1,
                          s2,
                          s3,
                          s4);
                }
              }

              int outputOffset = batch * rows + row;
              float d = weightScale * q8Scales[scaleBatchOffset + block];
              out[outputOffset] = MathUtil.fma(d, blockSum, out[outputOffset]);
            }
          }
        });
  }

  static int q6_KQ8_KIntegerDot(
      MemorySegment qWeight,
      long ql1Offset,
      long ql2Offset,
      long qhOffset,
      byte[] q8Quants,
      int quantOffset,
      int s1,
      int s2,
      int s3,
      int s4) {
    return s1 * q6_KQ8_KGroup0Dot(qWeight, ql1Offset, qhOffset, q8Quants, quantOffset)
        + s2 * q6_KQ8_KGroup2Dot(qWeight, ql2Offset, qhOffset, q8Quants, quantOffset + 32)
        + s3 * q6_KQ8_KGroup4Dot(qWeight, ql1Offset, qhOffset, q8Quants, quantOffset + 64)
        + s4 * q6_KQ8_KGroup6Dot(qWeight, ql2Offset, qhOffset, q8Quants, quantOffset + 96);
  }

  private static int q6_KQ8_KGroup0Dot(
      MemorySegment qWeight, long qlOffset, long qhOffset, byte[] q8Quants, int quantOffset) {
    int sum = 0;
    for (int index = 0; index < 16; index += 8) {
      ByteVector ql =
          ByteVector.fromMemorySegment(
              ByteVector.SPECIES_64, qWeight, qlOffset + index, ByteOrder.LITTLE_ENDIAN);
      ByteVector qh =
          ByteVector.fromMemorySegment(
              ByteVector.SPECIES_64, qWeight, qhOffset + index, ByteOrder.LITTLE_ENDIAN);
      ByteVector low = ql.and((byte) 0x0F);
      ByteVector high = qh.and((byte) 0x03).lanewise(VectorOperators.LSHL, 4);
      IntVector weights =
          (IntVector)
              low.add(high)
                  .sub((byte) 32)
                  .convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0);
      IntVector quants =
          (IntVector)
              ByteVector.fromArray(ByteVector.SPECIES_64, q8Quants, quantOffset + index)
                  .convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0);
      sum += weights.mul(quants).reduceLanes(VectorOperators.ADD);
    }
    return sum;
  }

  private static int q6_KQ8_KGroup2Dot(
      MemorySegment qWeight, long qlOffset, long qhOffset, byte[] q8Quants, int quantOffset) {
    int sum = 0;
    for (int index = 0; index < 16; index += 8) {
      ByteVector ql =
          ByteVector.fromMemorySegment(
              ByteVector.SPECIES_64, qWeight, qlOffset + index, ByteOrder.LITTLE_ENDIAN);
      ByteVector qh =
          ByteVector.fromMemorySegment(
              ByteVector.SPECIES_64, qWeight, qhOffset + index, ByteOrder.LITTLE_ENDIAN);
      ByteVector low = ql.and((byte) 0x0F);
      ByteVector high = qh.and((byte) 0x0C).lanewise(VectorOperators.LSHL, 2);
      IntVector weights =
          (IntVector)
              low.add(high)
                  .sub((byte) 32)
                  .convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0);
      IntVector quants =
          (IntVector)
              ByteVector.fromArray(ByteVector.SPECIES_64, q8Quants, quantOffset + index)
                  .convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0);
      sum += weights.mul(quants).reduceLanes(VectorOperators.ADD);
    }
    return sum;
  }

  private static int q6_KQ8_KGroup4Dot(
      MemorySegment qWeight, long qlOffset, long qhOffset, byte[] q8Quants, int quantOffset) {
    int sum = 0;
    for (int index = 0; index < 16; index += 8) {
      ByteVector ql =
          ByteVector.fromMemorySegment(
              ByteVector.SPECIES_64, qWeight, qlOffset + index, ByteOrder.LITTLE_ENDIAN);
      ByteVector qh =
          ByteVector.fromMemorySegment(
              ByteVector.SPECIES_64, qWeight, qhOffset + index, ByteOrder.LITTLE_ENDIAN);
      ByteVector low = ql.lanewise(VectorOperators.LSHR, 4).and((byte) 0x0F);
      ByteVector high = qh.and((byte) 0x30);
      IntVector weights =
          (IntVector)
              low.add(high)
                  .sub((byte) 32)
                  .convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0);
      IntVector quants =
          (IntVector)
              ByteVector.fromArray(ByteVector.SPECIES_64, q8Quants, quantOffset + index)
                  .convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0);
      sum += weights.mul(quants).reduceLanes(VectorOperators.ADD);
    }
    return sum;
  }

  private static int q6_KQ8_KGroup6Dot(
      MemorySegment qWeight, long qlOffset, long qhOffset, byte[] q8Quants, int quantOffset) {
    int sum = 0;
    for (int index = 0; index < 16; index += 8) {
      ByteVector ql =
          ByteVector.fromMemorySegment(
              ByteVector.SPECIES_64, qWeight, qlOffset + index, ByteOrder.LITTLE_ENDIAN);
      ByteVector qh =
          ByteVector.fromMemorySegment(
              ByteVector.SPECIES_64, qWeight, qhOffset + index, ByteOrder.LITTLE_ENDIAN);
      ByteVector low = ql.lanewise(VectorOperators.LSHR, 4).and((byte) 0x0F);
      ByteVector high = qh.and((byte) 0xC0).lanewise(VectorOperators.LSHR, 2);
      IntVector weights =
          (IntVector)
              low.add(high)
                  .sub((byte) 32)
                  .convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0);
      IntVector quants =
          (IntVector)
              ByteVector.fromArray(ByteVector.SPECIES_64, q8Quants, quantOffset + index)
                  .convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0);
      sum += weights.mul(quants).reduceLanes(VectorOperators.ADD);
    }
    return sum;
  }

  /** SIMD Q8_0 by Q8_0 GEMV with one activation quantization shared by all rows. */
  @Override
  public void ggufQ8_0Q8_0MatVecDot(
      float[] query,
      MemorySegment qWeight,
      int rows,
      int cols,
      float[] out,
      byte[] q8Quants,
      float[] q8Scales) {
    GgufQuantizationSupport.quantizeQ8_0(query, cols, q8Quants, q8Scales);

    long rowBytes = (long) (cols / GGUF_Q_BLOCK_SIZE) * GGUF_Q8_0_BLOCK_BYTES;
    int blocks = cols / GGUF_Q_BLOCK_SIZE;
    GgufParallelSupport.forEachRow(
        qWeight,
        rows,
        cols,
        GgufParallelSupport.Q8_MIN_ELEMENTS,
        row -> out[row] = q8_0Q8_0RowDot(qWeight, row * rowBytes, blocks, q8Quants, q8Scales));
  }

  /** SIMD Q8_0 by a batch of Q8_0 activations with row-local weight reuse. */
  @Override
  public void ggufQ8_0Q8_0BatchedMatmul(
      float[] queries,
      MemorySegment qWeight,
      int batchSize,
      int rows,
      int cols,
      float[] out,
      byte[] q8Quants,
      float[] q8Scales) {
    if (batchSize == 1) {
      ggufQ8_0Q8_0MatVecDot(queries, qWeight, rows, cols, out, q8Quants, q8Scales);
      return;
    }

    int blocks = cols / GGUF_Q_BLOCK_SIZE;
    for (int batch = 0; batch < batchSize; batch++) {
      GgufQuantizationSupport.quantizeQ8_0(
          queries, batch * cols, cols, q8Quants, batch * cols, q8Scales, batch * blocks);
    }

    long rowBytes = (long) blocks * GGUF_Q8_0_BLOCK_BYTES;
    int effectiveCols = Math.multiplyExact(cols, batchSize);
    GgufParallelSupport.forEachRow(
        qWeight,
        rows,
        effectiveCols,
        GgufParallelSupport.Q8_MIN_ELEMENTS,
        row -> {
          for (int batch = 0; batch < batchSize; batch++) {
            out[batch * rows + row] = 0.0f;
          }

          long rowOffset = row * rowBytes;
          for (int block = 0; block < blocks; block++) {
            long blockOffset = rowOffset + (long) block * GGUF_Q8_0_BLOCK_BYTES;
            float weightScale = Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset));
            long weightOffset = blockOffset + Short.BYTES;
            int blockActivationOffset = block * GGUF_Q_BLOCK_SIZE;
            q8_0Q8_0AccumulateBatchedBlock(
                qWeight,
                weightOffset,
                q8Quants,
                blockActivationOffset,
                cols,
                batchSize,
                weightScale,
                q8Scales,
                block,
                blocks,
                out,
                row,
                rows);
          }
        });
  }

  /** SIMD Q8_0 row-range multiplication over caller-prequantized activation rows. */
  @Override
  public void ggufQ8_0Q8_0BatchedMatmulRows(
      MemorySegment qWeight,
      int batchSize,
      int rows,
      int cols,
      int fromRow,
      int toRow,
      float[] out,
      GgufQ8_0Batch activation) {
    int blocks = cols / GGUF_Q_BLOCK_SIZE;
    byte[] q8Quants = activation.quants();
    float[] q8Scales = activation.scales();
    long rowBytes = (long) blocks * GGUF_Q8_0_BLOCK_BYTES;
    for (int row = fromRow; row < toRow; row++) {
      for (int batch = 0; batch < batchSize; batch++) {
        out[batch * rows + row] = 0.0f;
      }

      long rowOffset = row * rowBytes;
      for (int block = 0; block < blocks; block++) {
        long blockOffset = rowOffset + (long) block * GGUF_Q8_0_BLOCK_BYTES;
        float weightScale = Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset));
        long weightOffset = blockOffset + Short.BYTES;
        int blockActivationOffset = block * GGUF_Q_BLOCK_SIZE;
        q8_0Q8_0AccumulateBatchedBlock(
            qWeight,
            weightOffset,
            q8Quants,
            blockActivationOffset,
            cols,
            batchSize,
            weightScale,
            q8Scales,
            block,
            blocks,
            out,
            row,
            rows);
      }
    }
  }

  /** SIMD Q8_0 row-range multiplication over block-major caller-prequantized activation rows. */
  @Override
  public void ggufQ8_0Q8_0BlockMajorBatchedMatmulRows(
      MemorySegment qWeight,
      int batchSize,
      int rows,
      int cols,
      int fromRow,
      int toRow,
      float[] out,
      GgufQ8_0Batch activation) {
    if (batchSize == 1 || VECTOR_BITSIZE < 256) {
      ggufQ8_0Q8_0BatchedMatmulRows(
          qWeight, batchSize, rows, cols, fromRow, toRow, out, activation);
      return;
    }

    int blocks = cols / GGUF_Q_BLOCK_SIZE;
    byte[] blockMajorQuants = activation.blockMajorQuants();
    float[] q8Scales = activation.scales();
    long rowBytes = (long) blocks * GGUF_Q8_0_BLOCK_BYTES;
    for (int row = fromRow; row < toRow; row++) {
      for (int batch = 0; batch < batchSize; batch++) {
        out[batch * rows + row] = 0.0f;
      }

      long rowOffset = row * rowBytes;
      for (int block = 0; block < blocks; block++) {
        long blockOffset = rowOffset + (long) block * GGUF_Q8_0_BLOCK_BYTES;
        float weightScale = Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset));
        long weightOffset = blockOffset + Short.BYTES;
        int blockActivationOffset = block * activation.batchCapacity() * GGUF_Q_BLOCK_SIZE;
        q8_0Q8_0AccumulateBatchedBlock(
            qWeight,
            weightOffset,
            blockMajorQuants,
            blockActivationOffset,
            GGUF_Q_BLOCK_SIZE,
            batchSize,
            weightScale,
            q8Scales,
            block,
            blocks,
            out,
            row,
            rows);
      }
    }
  }

  @Override
  public void ggufQ8_0Q8_0DualBatchedMatmul(
      float[] queries,
      MemorySegment firstWeight,
      int firstRows,
      float[] firstOut,
      MemorySegment secondWeight,
      int secondRows,
      float[] secondOut,
      int batchSize,
      int cols,
      byte[] q8Quants,
      float[] q8Scales) {
    if (batchSize == 1) {
      ggufQ8_0Q8_0DualMatVecDot(
          queries,
          firstWeight,
          firstRows,
          firstOut,
          secondWeight,
          secondRows,
          secondOut,
          cols,
          q8Quants,
          q8Scales);
      return;
    }
    ggufQ8_0Q8_0GroupedBatchedMatmul(
        queries,
        firstWeight,
        firstRows,
        firstOut,
        secondWeight,
        secondRows,
        secondOut,
        secondWeight,
        0,
        secondOut,
        batchSize,
        cols,
        q8Quants,
        q8Scales);
  }

  @Override
  public void ggufQ8_0Q8_0TripleBatchedMatmul(
      float[] queries,
      MemorySegment firstWeight,
      int firstRows,
      float[] firstOut,
      MemorySegment secondWeight,
      int secondRows,
      float[] secondOut,
      MemorySegment thirdWeight,
      int thirdRows,
      float[] thirdOut,
      int batchSize,
      int cols,
      byte[] q8Quants,
      float[] q8Scales) {
    if (batchSize == 1) {
      ggufQ8_0Q8_0TripleMatVecDot(
          queries,
          firstWeight,
          firstRows,
          firstOut,
          secondWeight,
          secondRows,
          secondOut,
          thirdWeight,
          thirdRows,
          thirdOut,
          cols,
          q8Quants,
          q8Scales);
      return;
    }
    ggufQ8_0Q8_0GroupedBatchedMatmul(
        queries,
        firstWeight,
        firstRows,
        firstOut,
        secondWeight,
        secondRows,
        secondOut,
        thirdWeight,
        thirdRows,
        thirdOut,
        batchSize,
        cols,
        q8Quants,
        q8Scales);
  }

  private static void ggufQ8_0Q8_0GroupedBatchedMatmul(
      float[] queries,
      MemorySegment firstWeight,
      int firstRows,
      float[] firstOut,
      MemorySegment secondWeight,
      int secondRows,
      float[] secondOut,
      MemorySegment thirdWeight,
      int thirdRows,
      float[] thirdOut,
      int batchSize,
      int cols,
      byte[] q8Quants,
      float[] q8Scales) {
    int blocks = cols / GGUF_Q_BLOCK_SIZE;
    for (int batch = 0; batch < batchSize; batch++) {
      GgufQuantizationSupport.quantizeQ8_0(
          queries, batch * cols, cols, q8Quants, batch * cols, q8Scales, batch * blocks);
    }

    long rowBytes = (long) blocks * GGUF_Q8_0_BLOCK_BYTES;
    int secondStart = firstRows;
    int thirdStart = Math.addExact(firstRows, secondRows);
    int totalRows = Math.addExact(thirdStart, thirdRows);
    int effectiveCols = Math.multiplyExact(cols, batchSize);
    GgufParallelSupport.forEachRow(
        firstWeight,
        secondWeight,
        thirdWeight,
        totalRows,
        effectiveCols,
        GgufParallelSupport.Q8_MIN_ELEMENTS,
        row -> {
          MemorySegment qWeight;
          float[] out;
          int matrixRow;
          int matrixRows;
          if (row < secondStart) {
            qWeight = firstWeight;
            out = firstOut;
            matrixRow = row;
            matrixRows = firstRows;
          } else if (row < thirdStart) {
            qWeight = secondWeight;
            out = secondOut;
            matrixRow = row - secondStart;
            matrixRows = secondRows;
          } else {
            qWeight = thirdWeight;
            out = thirdOut;
            matrixRow = row - thirdStart;
            matrixRows = thirdRows;
          }

          for (int batch = 0; batch < batchSize; batch++) {
            out[batch * matrixRows + matrixRow] = 0.0f;
          }

          long rowOffset = matrixRow * rowBytes;
          for (int block = 0; block < blocks; block++) {
            long blockOffset = rowOffset + (long) block * GGUF_Q8_0_BLOCK_BYTES;
            float weightScale = Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset));
            long weightOffset = blockOffset + Short.BYTES;
            int blockActivationOffset = block * GGUF_Q_BLOCK_SIZE;
            q8_0Q8_0AccumulateBatchedBlock(
                qWeight,
                weightOffset,
                q8Quants,
                blockActivationOffset,
                cols,
                batchSize,
                weightScale,
                q8Scales,
                block,
                blocks,
                out,
                matrixRow,
                matrixRows);
          }
        });
  }

  @Override
  public void ggufQ8_0Q8_0DualMatVecDot(
      float[] query,
      MemorySegment firstWeight,
      int firstRows,
      float[] firstOut,
      MemorySegment secondWeight,
      int secondRows,
      float[] secondOut,
      int cols,
      byte[] q8Quants,
      float[] q8Scales) {
    ggufQ8_0Q8_0GroupedMatVecDot(
        query,
        firstWeight,
        firstRows,
        firstOut,
        secondWeight,
        secondRows,
        secondOut,
        secondWeight,
        0,
        secondOut,
        cols,
        q8Quants,
        q8Scales);
  }

  @Override
  public void ggufQ8_0Q8_0TripleMatVecDot(
      float[] query,
      MemorySegment firstWeight,
      int firstRows,
      float[] firstOut,
      MemorySegment secondWeight,
      int secondRows,
      float[] secondOut,
      MemorySegment thirdWeight,
      int thirdRows,
      float[] thirdOut,
      int cols,
      byte[] q8Quants,
      float[] q8Scales) {
    ggufQ8_0Q8_0GroupedMatVecDot(
        query,
        firstWeight,
        firstRows,
        firstOut,
        secondWeight,
        secondRows,
        secondOut,
        thirdWeight,
        thirdRows,
        thirdOut,
        cols,
        q8Quants,
        q8Scales);
  }

  private static void ggufQ8_0Q8_0GroupedMatVecDot(
      float[] query,
      MemorySegment firstWeight,
      int firstRows,
      float[] firstOut,
      MemorySegment secondWeight,
      int secondRows,
      float[] secondOut,
      MemorySegment thirdWeight,
      int thirdRows,
      float[] thirdOut,
      int cols,
      byte[] q8Quants,
      float[] q8Scales) {
    GgufQuantizationSupport.quantizeQ8_0(query, cols, q8Quants, q8Scales);

    long rowBytes = (long) (cols / GGUF_Q_BLOCK_SIZE) * GGUF_Q8_0_BLOCK_BYTES;
    int blocks = cols / GGUF_Q_BLOCK_SIZE;
    int secondStart = firstRows;
    int thirdStart = Math.addExact(firstRows, secondRows);
    int totalRows = Math.addExact(thirdStart, thirdRows);
    GgufParallelSupport.forEachRow(
        firstWeight,
        secondWeight,
        thirdWeight,
        totalRows,
        cols,
        GgufParallelSupport.Q8_MIN_ELEMENTS,
        row -> {
          MemorySegment qWeight;
          float[] out;
          int matrixRow;
          if (row < secondStart) {
            qWeight = firstWeight;
            out = firstOut;
            matrixRow = row;
          } else if (row < thirdStart) {
            qWeight = secondWeight;
            out = secondOut;
            matrixRow = row - secondStart;
          } else {
            qWeight = thirdWeight;
            out = thirdOut;
            matrixRow = row - thirdStart;
          }
          out[matrixRow] =
              q8_0Q8_0RowDot(qWeight, matrixRow * rowBytes, blocks, q8Quants, q8Scales);
        });
  }

  private static float q8_0Q8_0RowDot(
      MemorySegment qWeight, long rowOffset, int blocks, byte[] q8Quants, float[] q8Scales) {
    float sum = 0.0f;
    for (int block = 0; block < blocks; block++) {
      long blockOffset = rowOffset + (long) block * GGUF_Q8_0_BLOCK_BYTES;
      float scale = Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset)) * q8Scales[block];
      int integerSum =
          q8_0Q8_0IntegerDot(
              qWeight, blockOffset + Short.BYTES, q8Quants, block * GGUF_Q_BLOCK_SIZE);
      sum = MathUtil.fma(scale, integerSum, sum);
    }
    return sum;
  }

  private static void q8_0Q8_0AccumulateBatchedBlock(
      MemorySegment qWeight,
      long weightOffset,
      byte[] q8Quants,
      int quantOffset,
      int quantStride,
      int batchSize,
      float weightScale,
      float[] q8Scales,
      int scaleOffset,
      int scaleStride,
      float[] out,
      int outputOffset,
      int outputStride) {
    if (batchSize == 1) {
      int integerSum = q8_0Q8_0IntegerDot(qWeight, weightOffset, q8Quants, quantOffset);
      float scale = weightScale * q8Scales[scaleOffset];
      out[outputOffset] = MathUtil.fma(scale, integerSum, out[outputOffset]);
      return;
    }

    if (VECTOR_BITSIZE >= 512) {
      IntVector weight0 =
          (IntVector)
              ByteVector.fromMemorySegment(
                      ByteVector.SPECIES_128, qWeight, weightOffset, ByteOrder.LITTLE_ENDIAN)
                  .convertShape(VectorOperators.B2I, IntVector.SPECIES_512, 0);
      IntVector weight1 =
          (IntVector)
              ByteVector.fromMemorySegment(
                      ByteVector.SPECIES_128, qWeight, weightOffset + 16, ByteOrder.LITTLE_ENDIAN)
                  .convertShape(VectorOperators.B2I, IntVector.SPECIES_512, 0);
      for (int batch = 0; batch < batchSize; batch++) {
        int batchQuantOffset = quantOffset + batch * quantStride;
        IntVector accumulator = IntVector.zero(IntVector.SPECIES_512);
        IntVector quant0 =
            (IntVector)
                ByteVector.fromArray(ByteVector.SPECIES_128, q8Quants, batchQuantOffset)
                    .convertShape(VectorOperators.B2I, IntVector.SPECIES_512, 0);
        IntVector quant1 =
            (IntVector)
                ByteVector.fromArray(ByteVector.SPECIES_128, q8Quants, batchQuantOffset + 16)
                    .convertShape(VectorOperators.B2I, IntVector.SPECIES_512, 0);
        accumulator = accumulator.add(weight0.mul(quant0));
        accumulator = accumulator.add(weight1.mul(quant1));
        int integerSum = accumulator.reduceLanes(VectorOperators.ADD);
        int outputIndex = outputOffset + batch * outputStride;
        float scale = weightScale * q8Scales[scaleOffset + batch * scaleStride];
        out[outputIndex] = MathUtil.fma(scale, integerSum, out[outputIndex]);
      }
      return;
    }

    if (VECTOR_BITSIZE >= 256) {
      IntVector weight0 =
          (IntVector)
              ByteVector.fromMemorySegment(
                      ByteVector.SPECIES_64, qWeight, weightOffset, ByteOrder.LITTLE_ENDIAN)
                  .convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0);
      IntVector weight1 =
          (IntVector)
              ByteVector.fromMemorySegment(
                      ByteVector.SPECIES_64, qWeight, weightOffset + 8, ByteOrder.LITTLE_ENDIAN)
                  .convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0);
      IntVector weight2 =
          (IntVector)
              ByteVector.fromMemorySegment(
                      ByteVector.SPECIES_64, qWeight, weightOffset + 16, ByteOrder.LITTLE_ENDIAN)
                  .convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0);
      IntVector weight3 =
          (IntVector)
              ByteVector.fromMemorySegment(
                      ByteVector.SPECIES_64, qWeight, weightOffset + 24, ByteOrder.LITTLE_ENDIAN)
                  .convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0);
      for (int batch = 0; batch < batchSize; batch++) {
        int batchQuantOffset = quantOffset + batch * quantStride;
        IntVector accumulator = IntVector.zero(IntVector.SPECIES_256);
        IntVector quant0 =
            (IntVector)
                ByteVector.fromArray(ByteVector.SPECIES_64, q8Quants, batchQuantOffset)
                    .convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0);
        IntVector quant1 =
            (IntVector)
                ByteVector.fromArray(ByteVector.SPECIES_64, q8Quants, batchQuantOffset + 8)
                    .convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0);
        IntVector quant2 =
            (IntVector)
                ByteVector.fromArray(ByteVector.SPECIES_64, q8Quants, batchQuantOffset + 16)
                    .convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0);
        IntVector quant3 =
            (IntVector)
                ByteVector.fromArray(ByteVector.SPECIES_64, q8Quants, batchQuantOffset + 24)
                    .convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0);
        accumulator = accumulator.add(weight0.mul(quant0));
        accumulator = accumulator.add(weight1.mul(quant1));
        accumulator = accumulator.add(weight2.mul(quant2));
        accumulator = accumulator.add(weight3.mul(quant3));
        int integerSum = accumulator.reduceLanes(VectorOperators.ADD);
        int outputIndex = outputOffset + batch * outputStride;
        float scale = weightScale * q8Scales[scaleOffset + batch * scaleStride];
        out[outputIndex] = MathUtil.fma(scale, integerSum, out[outputIndex]);
      }
      return;
    }

    for (int batch = 0; batch < batchSize; batch++) {
      int integerSum =
          q8_0Q8_0IntegerDot(qWeight, weightOffset, q8Quants, quantOffset + batch * quantStride);
      int outputIndex = outputOffset + batch * outputStride;
      float scale = weightScale * q8Scales[scaleOffset + batch * scaleStride];
      out[outputIndex] = MathUtil.fma(scale, integerSum, out[outputIndex]);
    }
  }

  private static int q8_0Q8_0IntegerDot(
      MemorySegment qWeight, long weightOffset, byte[] q8Quants, int quantOffset) {
    if (VECTOR_BITSIZE >= 512) {
      IntVector accumulator = IntVector.zero(IntVector.SPECIES_512);
      for (int index = 0; index < GGUF_Q_BLOCK_SIZE; index += 16) {
        ByteVector weights =
            ByteVector.fromMemorySegment(
                ByteVector.SPECIES_128, qWeight, weightOffset + index, ByteOrder.LITTLE_ENDIAN);
        ByteVector quants =
            ByteVector.fromArray(ByteVector.SPECIES_128, q8Quants, quantOffset + index);
        IntVector weightInts =
            (IntVector) weights.convertShape(VectorOperators.B2I, IntVector.SPECIES_512, 0);
        IntVector quantInts =
            (IntVector) quants.convertShape(VectorOperators.B2I, IntVector.SPECIES_512, 0);
        accumulator = accumulator.add(weightInts.mul(quantInts));
      }
      return accumulator.reduceLanes(VectorOperators.ADD);
    }

    if (VECTOR_BITSIZE >= 256) {
      IntVector accumulator = IntVector.zero(IntVector.SPECIES_256);
      for (int index = 0; index < GGUF_Q_BLOCK_SIZE; index += 8) {
        ByteVector weights =
            ByteVector.fromMemorySegment(
                ByteVector.SPECIES_64, qWeight, weightOffset + index, ByteOrder.LITTLE_ENDIAN);
        ByteVector quants =
            ByteVector.fromArray(ByteVector.SPECIES_64, q8Quants, quantOffset + index);
        IntVector weightInts =
            (IntVector) weights.convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0);
        IntVector quantInts =
            (IntVector) quants.convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0);
        accumulator = accumulator.add(weightInts.mul(quantInts));
      }
      return accumulator.reduceLanes(VectorOperators.ADD);
    }

    if (VECTOR_BITSIZE >= 128) {
      IntVector lowAccumulator = IntVector.zero(IntVector.SPECIES_128);
      IntVector highAccumulator = IntVector.zero(IntVector.SPECIES_128);
      for (int index = 0; index < GGUF_Q_BLOCK_SIZE; index += 8) {
        ShortVector weights =
            (ShortVector)
                ByteVector.fromMemorySegment(
                        ByteVector.SPECIES_64,
                        qWeight,
                        weightOffset + index,
                        ByteOrder.LITTLE_ENDIAN)
                    .convertShape(VectorOperators.B2S, ShortVector.SPECIES_128, 0);
        ShortVector quants =
            (ShortVector)
                ByteVector.fromArray(ByteVector.SPECIES_64, q8Quants, quantOffset + index)
                    .convertShape(VectorOperators.B2S, ShortVector.SPECIES_128, 0);
        IntVector weightLow =
            (IntVector) weights.convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 0);
        IntVector weightHigh =
            (IntVector) weights.convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 1);
        IntVector quantLow =
            (IntVector) quants.convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 0);
        IntVector quantHigh =
            (IntVector) quants.convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 1);
        lowAccumulator = lowAccumulator.add(weightLow.mul(quantLow));
        highAccumulator = highAccumulator.add(weightHigh.mul(quantHigh));
      }
      return lowAccumulator.add(highAccumulator).reduceLanes(VectorOperators.ADD);
    }

    int sum = 0;
    for (int index = 0; index < GGUF_Q_BLOCK_SIZE; index++) {
      sum +=
          qWeight.get(ValueLayout.JAVA_BYTE, weightOffset + index) * q8Quants[quantOffset + index];
    }
    return sum;
  }

  // --- Byte dot product: widening to avoid overflow ---
  //
  // Tiered widening strategy (B2I = byte-to-int, 4x lane expansion):
  //   512-bit path: ByteVector.SPECIES_128 (16 bytes) → IntVector.SPECIES_512 (16 ints)
  //   256-bit path: ByteVector.SPECIES_64  ( 8 bytes) → IntVector.SPECIES_256 ( 8 ints)
  //   128-bit path: ByteVector.SPECIES_64  ( 8 bytes) → ShortVector.SPECIES_128 (8 shorts) →
  //                 2× IntVector.SPECIES_128 (lo/hi halves) via S2I; split accumulators
  // Each tier accumulates into its own IntVector and adds to `res` via +=, so tiers compose safely.
  // Int32 overflow is safe: max per-lane accumulation ≤ (dim/lanes) * 127^2 which for dim=1536
  // is ≤ 96 * 16129 ≈ 1.5M, well within int32 range (2.1B).

  @Override
  public int dotProduct(byte[] a, byte[] b) {
    int i = 0;
    int res = 0;

    // 512-bit path: 16 bytes → 16 ints per iteration
    if (VECTOR_BITSIZE >= 512 && a.length >= ByteVector.SPECIES_128.length()) {
      int limit = i + ByteVector.SPECIES_128.loopBound(a.length - i);
      IntVector acc = IntVector.zero(IntVector.SPECIES_512);
      for (; i < limit; i += ByteVector.SPECIES_128.length()) {
        ByteVector va = ByteVector.fromArray(ByteVector.SPECIES_128, a, i);
        ByteVector vb = ByteVector.fromArray(ByteVector.SPECIES_128, b, i);
        IntVector ia = (IntVector) va.convertShape(VectorOperators.B2I, IntVector.SPECIES_512, 0);
        IntVector ib = (IntVector) vb.convertShape(VectorOperators.B2I, IntVector.SPECIES_512, 0);
        acc = acc.add(ia.mul(ib));
      }
      res += acc.reduceLanes(VectorOperators.ADD);
    }

    // 256-bit path: 8 bytes → 8 ints per iteration (handles remainder after 512-bit or standalone)
    if (VECTOR_BITSIZE >= 256 && a.length - i >= ByteVector.SPECIES_64.length()) {
      int limit = i + ByteVector.SPECIES_64.loopBound(a.length - i);
      IntVector acc = IntVector.zero(IntVector.SPECIES_256);
      for (; i < limit; i += ByteVector.SPECIES_64.length()) {
        ByteVector va = ByteVector.fromArray(ByteVector.SPECIES_64, a, i);
        ByteVector vb = ByteVector.fromArray(ByteVector.SPECIES_64, b, i);
        IntVector ia = (IntVector) va.convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0);
        IntVector ib = (IntVector) vb.convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0);
        acc = acc.add(ia.mul(ib));
      }
      res += acc.reduceLanes(VectorOperators.ADD);
    }

    // 128-bit path (NEON, RVV-128): 8 bytes → 8 shorts → 2× 4 ints per iteration.
    // Byte*byte products fit in int16 (|a*b| ≤ 127*127 = 16129), so short mul is safe.
    if (VECTOR_BITSIZE == 128 && a.length - i >= ByteVector.SPECIES_64.length()) {
      int limit = i + ByteVector.SPECIES_64.loopBound(a.length - i);
      IntVector acc1 = IntVector.zero(IntVector.SPECIES_128);
      IntVector acc2 = IntVector.zero(IntVector.SPECIES_128);
      for (; i < limit; i += ByteVector.SPECIES_64.length()) {
        ByteVector va = ByteVector.fromArray(ByteVector.SPECIES_64, a, i);
        ByteVector vb = ByteVector.fromArray(ByteVector.SPECIES_64, b, i);
        ShortVector va16 =
            (ShortVector) va.convertShape(VectorOperators.B2S, ShortVector.SPECIES_128, 0);
        ShortVector vb16 =
            (ShortVector) vb.convertShape(VectorOperators.B2S, ShortVector.SPECIES_128, 0);
        ShortVector prod16 = va16.mul(vb16);
        acc1 =
            acc1.add(
                (IntVector) prod16.convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 0));
        acc2 =
            acc2.add(
                (IntVector) prod16.convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 1));
      }
      res += acc1.add(acc2).reduceLanes(VectorOperators.ADD);
    }

    // Scalar tail
    for (; i < a.length; i++) {
      res += a[i] * b[i];
    }
    return res;
  }

  @Override
  public int squareDistance(byte[] a, byte[] b) {
    int i = 0;
    int res = 0;

    // 512-bit path
    if (VECTOR_BITSIZE >= 512 && a.length >= ByteVector.SPECIES_128.length()) {
      int limit = i + ByteVector.SPECIES_128.loopBound(a.length - i);
      IntVector acc = IntVector.zero(IntVector.SPECIES_512);
      for (; i < limit; i += ByteVector.SPECIES_128.length()) {
        ByteVector va = ByteVector.fromArray(ByteVector.SPECIES_128, a, i);
        ByteVector vb = ByteVector.fromArray(ByteVector.SPECIES_128, b, i);
        IntVector ia = (IntVector) va.convertShape(VectorOperators.B2I, IntVector.SPECIES_512, 0);
        IntVector ib = (IntVector) vb.convertShape(VectorOperators.B2I, IntVector.SPECIES_512, 0);
        IntVector diff = ia.sub(ib);
        acc = acc.add(diff.mul(diff));
      }
      res += acc.reduceLanes(VectorOperators.ADD);
    }

    // 256-bit path
    if (VECTOR_BITSIZE >= 256 && a.length - i >= ByteVector.SPECIES_64.length()) {
      int limit = i + ByteVector.SPECIES_64.loopBound(a.length - i);
      IntVector acc = IntVector.zero(IntVector.SPECIES_256);
      for (; i < limit; i += ByteVector.SPECIES_64.length()) {
        ByteVector va = ByteVector.fromArray(ByteVector.SPECIES_64, a, i);
        ByteVector vb = ByteVector.fromArray(ByteVector.SPECIES_64, b, i);
        IntVector ia = (IntVector) va.convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0);
        IntVector ib = (IntVector) vb.convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0);
        IntVector diff = ia.sub(ib);
        acc = acc.add(diff.mul(diff));
      }
      res += acc.reduceLanes(VectorOperators.ADD);
    }

    // 128-bit path: 8 bytes → 8 shorts → sub in short, then split into 2× 4 ints for mul+add.
    // |a - b| ≤ 255 fits in int16; diff*diff ≤ 65025 does not fit in int16, hence mul in int.
    if (VECTOR_BITSIZE == 128 && a.length - i >= ByteVector.SPECIES_64.length()) {
      int limit = i + ByteVector.SPECIES_64.loopBound(a.length - i);
      IntVector acc1 = IntVector.zero(IntVector.SPECIES_128);
      IntVector acc2 = IntVector.zero(IntVector.SPECIES_128);
      for (; i < limit; i += ByteVector.SPECIES_64.length()) {
        ByteVector va = ByteVector.fromArray(ByteVector.SPECIES_64, a, i);
        ByteVector vb = ByteVector.fromArray(ByteVector.SPECIES_64, b, i);
        ShortVector va16 =
            (ShortVector) va.convertShape(VectorOperators.B2S, ShortVector.SPECIES_128, 0);
        ShortVector vb16 =
            (ShortVector) vb.convertShape(VectorOperators.B2S, ShortVector.SPECIES_128, 0);
        ShortVector diff16 = va16.sub(vb16);
        IntVector diff32_lo =
            (IntVector) diff16.convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 0);
        IntVector diff32_hi =
            (IntVector) diff16.convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 1);
        acc1 = acc1.add(diff32_lo.mul(diff32_lo));
        acc2 = acc2.add(diff32_hi.mul(diff32_hi));
      }
      res += acc1.add(acc2).reduceLanes(VectorOperators.ADD);
    }

    for (; i < a.length; i++) {
      int diff = a[i] - b[i];
      res += diff * diff;
    }
    return res;
  }

  @Override
  public float cosine(byte[] a, byte[] b) {
    int sum = 0;
    int norm1 = 0;
    int norm2 = 0;
    int i = 0;

    // 512-bit path
    if (VECTOR_BITSIZE >= 512 && a.length >= ByteVector.SPECIES_128.length()) {
      int limit = i + ByteVector.SPECIES_128.loopBound(a.length - i);
      IntVector vSum = IntVector.zero(IntVector.SPECIES_512);
      IntVector vNorm1 = IntVector.zero(IntVector.SPECIES_512);
      IntVector vNorm2 = IntVector.zero(IntVector.SPECIES_512);
      for (; i < limit; i += ByteVector.SPECIES_128.length()) {
        ByteVector va = ByteVector.fromArray(ByteVector.SPECIES_128, a, i);
        ByteVector vb = ByteVector.fromArray(ByteVector.SPECIES_128, b, i);
        IntVector ia = (IntVector) va.convertShape(VectorOperators.B2I, IntVector.SPECIES_512, 0);
        IntVector ib = (IntVector) vb.convertShape(VectorOperators.B2I, IntVector.SPECIES_512, 0);
        vSum = vSum.add(ia.mul(ib));
        vNorm1 = vNorm1.add(ia.mul(ia));
        vNorm2 = vNorm2.add(ib.mul(ib));
      }
      sum += vSum.reduceLanes(VectorOperators.ADD);
      norm1 += vNorm1.reduceLanes(VectorOperators.ADD);
      norm2 += vNorm2.reduceLanes(VectorOperators.ADD);
    }

    // 256-bit path
    if (VECTOR_BITSIZE >= 256 && a.length - i >= ByteVector.SPECIES_64.length()) {
      int limit = i + ByteVector.SPECIES_64.loopBound(a.length - i);
      IntVector vSum = IntVector.zero(IntVector.SPECIES_256);
      IntVector vNorm1 = IntVector.zero(IntVector.SPECIES_256);
      IntVector vNorm2 = IntVector.zero(IntVector.SPECIES_256);
      for (; i < limit; i += ByteVector.SPECIES_64.length()) {
        ByteVector va = ByteVector.fromArray(ByteVector.SPECIES_64, a, i);
        ByteVector vb = ByteVector.fromArray(ByteVector.SPECIES_64, b, i);
        IntVector ia = (IntVector) va.convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0);
        IntVector ib = (IntVector) vb.convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0);
        vSum = vSum.add(ia.mul(ib));
        vNorm1 = vNorm1.add(ia.mul(ia));
        vNorm2 = vNorm2.add(ib.mul(ib));
      }
      sum += vSum.reduceLanes(VectorOperators.ADD);
      norm1 += vNorm1.reduceLanes(VectorOperators.ADD);
      norm2 += vNorm2.reduceLanes(VectorOperators.ADD);
    }

    // 128-bit path: 8 bytes → 8 shorts, multiply in short, widen halves to int for accumulation.
    // All three quantities (a*b, a*a, b*b) stay within int16 range (≤ 16129 for int8 inputs).
    if (VECTOR_BITSIZE == 128 && a.length - i >= ByteVector.SPECIES_64.length()) {
      int limit = i + ByteVector.SPECIES_64.loopBound(a.length - i);
      IntVector vSum1 = IntVector.zero(IntVector.SPECIES_128);
      IntVector vSum2 = IntVector.zero(IntVector.SPECIES_128);
      IntVector vNorm1Lo = IntVector.zero(IntVector.SPECIES_128);
      IntVector vNorm1Hi = IntVector.zero(IntVector.SPECIES_128);
      IntVector vNorm2Lo = IntVector.zero(IntVector.SPECIES_128);
      IntVector vNorm2Hi = IntVector.zero(IntVector.SPECIES_128);
      for (; i < limit; i += ByteVector.SPECIES_64.length()) {
        ByteVector va = ByteVector.fromArray(ByteVector.SPECIES_64, a, i);
        ByteVector vb = ByteVector.fromArray(ByteVector.SPECIES_64, b, i);
        ShortVector va16 =
            (ShortVector) va.convertShape(VectorOperators.B2S, ShortVector.SPECIES_128, 0);
        ShortVector vb16 =
            (ShortVector) vb.convertShape(VectorOperators.B2S, ShortVector.SPECIES_128, 0);
        ShortVector dot16 = va16.mul(vb16);
        ShortVector n1_16 = va16.mul(va16);
        ShortVector n2_16 = vb16.mul(vb16);
        vSum1 =
            vSum1.add(
                (IntVector) dot16.convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 0));
        vSum2 =
            vSum2.add(
                (IntVector) dot16.convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 1));
        vNorm1Lo =
            vNorm1Lo.add(
                (IntVector) n1_16.convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 0));
        vNorm1Hi =
            vNorm1Hi.add(
                (IntVector) n1_16.convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 1));
        vNorm2Lo =
            vNorm2Lo.add(
                (IntVector) n2_16.convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 0));
        vNorm2Hi =
            vNorm2Hi.add(
                (IntVector) n2_16.convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 1));
      }
      sum += vSum1.add(vSum2).reduceLanes(VectorOperators.ADD);
      norm1 += vNorm1Lo.add(vNorm1Hi).reduceLanes(VectorOperators.ADD);
      norm2 += vNorm2Lo.add(vNorm2Hi).reduceLanes(VectorOperators.ADD);
    }

    for (; i < a.length; i++) {
      sum += a[i] * b[i];
      norm1 += a[i] * a[i];
      norm2 += b[i] * b[i];
    }

    return (float) (sum / Math.sqrt((double) norm1 * (double) norm2));
  }

  // --- MemorySegment distance kernels ---

  @Override
  public float dotProduct(MemorySegment a, MemorySegment b, int dimensions) {
    int i = 0;
    float res = 0f;

    if (dimensions > 2 * FLOAT_SPECIES.length()) {
      int limit = FLOAT_SPECIES.loopBound(dimensions);
      FloatVector acc1 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector acc2 = FloatVector.zero(FLOAT_SPECIES);
      int unrolledLimit = limit - FLOAT_SPECIES.length();

      for (; i < unrolledLimit; i += 2 * FLOAT_SPECIES.length()) {
        long off1 = (long) i * Float.BYTES;
        long off2 = (long) (i + FLOAT_SPECIES.length()) * Float.BYTES;
        FloatVector va1 =
            FloatVector.fromMemorySegment(FLOAT_SPECIES, a, off1, ByteOrder.LITTLE_ENDIAN);
        FloatVector vb1 =
            FloatVector.fromMemorySegment(FLOAT_SPECIES, b, off1, ByteOrder.LITTLE_ENDIAN);
        acc1 = fma(va1, vb1, acc1);
        FloatVector va2 =
            FloatVector.fromMemorySegment(FLOAT_SPECIES, a, off2, ByteOrder.LITTLE_ENDIAN);
        FloatVector vb2 =
            FloatVector.fromMemorySegment(FLOAT_SPECIES, b, off2, ByteOrder.LITTLE_ENDIAN);
        acc2 = fma(va2, vb2, acc2);
      }

      for (; i < limit; i += FLOAT_SPECIES.length()) {
        long off = (long) i * Float.BYTES;
        FloatVector va =
            FloatVector.fromMemorySegment(FLOAT_SPECIES, a, off, ByteOrder.LITTLE_ENDIAN);
        FloatVector vb =
            FloatVector.fromMemorySegment(FLOAT_SPECIES, b, off, ByteOrder.LITTLE_ENDIAN);
        acc1 = fma(va, vb, acc1);
      }
      res = acc1.add(acc2).reduceLanes(VectorOperators.ADD);
    }

    // Scalar tail
    for (; i < dimensions; i++) {
      float ai = a.getAtIndex(ValueLayout.JAVA_FLOAT, i);
      float bi = b.getAtIndex(ValueLayout.JAVA_FLOAT, i);
      res = MathUtil.fma(ai, bi, res);
    }
    return res;
  }

  @Override
  public float squareDistance(MemorySegment a, MemorySegment b, int dimensions) {
    int i = 0;
    float res = 0f;

    if (dimensions > 2 * FLOAT_SPECIES.length()) {
      int limit = FLOAT_SPECIES.loopBound(dimensions);
      FloatVector acc1 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector acc2 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector acc3 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector acc4 = FloatVector.zero(FLOAT_SPECIES);
      int unrolledLimit = limit - 3 * FLOAT_SPECIES.length();

      // 4x unrolled for ILP (matches array squareDistance body)
      for (; i < unrolledLimit; i += 4 * FLOAT_SPECIES.length()) {
        long off1 = (long) i * Float.BYTES;
        long off2 = (long) (i + FLOAT_SPECIES.length()) * Float.BYTES;
        long off3 = (long) (i + 2 * FLOAT_SPECIES.length()) * Float.BYTES;
        long off4 = (long) (i + 3 * FLOAT_SPECIES.length()) * Float.BYTES;

        FloatVector diff1 =
            FloatVector.fromMemorySegment(FLOAT_SPECIES, a, off1, ByteOrder.LITTLE_ENDIAN)
                .sub(
                    FloatVector.fromMemorySegment(FLOAT_SPECIES, b, off1, ByteOrder.LITTLE_ENDIAN));
        acc1 = fma(diff1, diff1, acc1);

        FloatVector diff2 =
            FloatVector.fromMemorySegment(FLOAT_SPECIES, a, off2, ByteOrder.LITTLE_ENDIAN)
                .sub(
                    FloatVector.fromMemorySegment(FLOAT_SPECIES, b, off2, ByteOrder.LITTLE_ENDIAN));
        acc2 = fma(diff2, diff2, acc2);

        FloatVector diff3 =
            FloatVector.fromMemorySegment(FLOAT_SPECIES, a, off3, ByteOrder.LITTLE_ENDIAN)
                .sub(
                    FloatVector.fromMemorySegment(FLOAT_SPECIES, b, off3, ByteOrder.LITTLE_ENDIAN));
        acc3 = fma(diff3, diff3, acc3);

        FloatVector diff4 =
            FloatVector.fromMemorySegment(FLOAT_SPECIES, a, off4, ByteOrder.LITTLE_ENDIAN)
                .sub(
                    FloatVector.fromMemorySegment(FLOAT_SPECIES, b, off4, ByteOrder.LITTLE_ENDIAN));
        acc4 = fma(diff4, diff4, acc4);
      }

      // Vector tail: remaining full vectors
      for (; i < limit; i += FLOAT_SPECIES.length()) {
        long off = (long) i * Float.BYTES;
        FloatVector diff =
            FloatVector.fromMemorySegment(FLOAT_SPECIES, a, off, ByteOrder.LITTLE_ENDIAN)
                .sub(FloatVector.fromMemorySegment(FLOAT_SPECIES, b, off, ByteOrder.LITTLE_ENDIAN));
        acc1 = fma(diff, diff, acc1);
      }

      FloatVector r1 = acc1.add(acc2);
      FloatVector r2 = acc3.add(acc4);
      res = r1.add(r2).reduceLanes(VectorOperators.ADD);
    }

    // Scalar tail
    for (; i < dimensions; i++) {
      float ai = a.getAtIndex(ValueLayout.JAVA_FLOAT, i);
      float bi = b.getAtIndex(ValueLayout.JAVA_FLOAT, i);
      float diff = ai - bi;
      res += diff * diff;
    }
    return res;
  }

  // --- MemorySegment cosine: 2x unrolled, 3 FMAs per iteration ---

  @Override
  public float cosine(MemorySegment a, MemorySegment b, int dimensions) {
    int i = 0;
    float sum = 0f;
    float norm1 = 0f;
    float norm2 = 0f;

    int lanes = FLOAT_SPECIES.length();
    if (dimensions >= 4 * lanes) {
      // 4x unrolled: 12 independent FMA accumulators hide FMA latency — parity with the float[]
      // cosine path (cosineBody4x). 3 FMAs per lane-group (dot, ‖a‖², ‖b‖²).
      int limit = FLOAT_SPECIES.loopBound(dimensions);
      FloatVector s0 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector s1 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector s2 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector s3 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector n1_0 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector n1_1 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector n1_2 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector n1_3 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector n2_0 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector n2_1 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector n2_2 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector n2_3 = FloatVector.zero(FLOAT_SPECIES);
      int unrolledLimit = limit - 3 * lanes;

      for (; i < unrolledLimit; i += 4 * lanes) {
        long o0 = (long) i * Float.BYTES;
        long o1 = (long) (i + lanes) * Float.BYTES;
        long o2 = (long) (i + 2 * lanes) * Float.BYTES;
        long o3 = (long) (i + 3 * lanes) * Float.BYTES;
        FloatVector va0 =
            FloatVector.fromMemorySegment(FLOAT_SPECIES, a, o0, ByteOrder.LITTLE_ENDIAN);
        FloatVector vb0 =
            FloatVector.fromMemorySegment(FLOAT_SPECIES, b, o0, ByteOrder.LITTLE_ENDIAN);
        FloatVector va1 =
            FloatVector.fromMemorySegment(FLOAT_SPECIES, a, o1, ByteOrder.LITTLE_ENDIAN);
        FloatVector vb1 =
            FloatVector.fromMemorySegment(FLOAT_SPECIES, b, o1, ByteOrder.LITTLE_ENDIAN);
        FloatVector va2 =
            FloatVector.fromMemorySegment(FLOAT_SPECIES, a, o2, ByteOrder.LITTLE_ENDIAN);
        FloatVector vb2 =
            FloatVector.fromMemorySegment(FLOAT_SPECIES, b, o2, ByteOrder.LITTLE_ENDIAN);
        FloatVector va3 =
            FloatVector.fromMemorySegment(FLOAT_SPECIES, a, o3, ByteOrder.LITTLE_ENDIAN);
        FloatVector vb3 =
            FloatVector.fromMemorySegment(FLOAT_SPECIES, b, o3, ByteOrder.LITTLE_ENDIAN);
        s0 = fma(va0, vb0, s0);
        s1 = fma(va1, vb1, s1);
        s2 = fma(va2, vb2, s2);
        s3 = fma(va3, vb3, s3);
        n1_0 = fma(va0, va0, n1_0);
        n1_1 = fma(va1, va1, n1_1);
        n1_2 = fma(va2, va2, n1_2);
        n1_3 = fma(va3, va3, n1_3);
        n2_0 = fma(vb0, vb0, n2_0);
        n2_1 = fma(vb1, vb1, n2_1);
        n2_2 = fma(vb2, vb2, n2_2);
        n2_3 = fma(vb3, vb3, n2_3);
      }

      // Vector tail (1 lane-width at a time)
      for (; i < limit; i += lanes) {
        long off = (long) i * Float.BYTES;
        FloatVector va =
            FloatVector.fromMemorySegment(FLOAT_SPECIES, a, off, ByteOrder.LITTLE_ENDIAN);
        FloatVector vb =
            FloatVector.fromMemorySegment(FLOAT_SPECIES, b, off, ByteOrder.LITTLE_ENDIAN);
        s0 = fma(va, vb, s0);
        n1_0 = fma(va, va, n1_0);
        n2_0 = fma(vb, vb, n2_0);
      }

      sum = s0.add(s1).add(s2.add(s3)).reduceLanes(VectorOperators.ADD);
      norm1 = n1_0.add(n1_1).add(n1_2.add(n1_3)).reduceLanes(VectorOperators.ADD);
      norm2 = n2_0.add(n2_1).add(n2_2.add(n2_3)).reduceLanes(VectorOperators.ADD);
    } else if (dimensions >= lanes) {
      // Short vectors: single-accumulator vector body (no unroll prologue cost).
      int limit = FLOAT_SPECIES.loopBound(dimensions);
      FloatVector vs = FloatVector.zero(FLOAT_SPECIES);
      FloatVector vn1 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector vn2 = FloatVector.zero(FLOAT_SPECIES);
      for (; i < limit; i += lanes) {
        long off = (long) i * Float.BYTES;
        FloatVector va =
            FloatVector.fromMemorySegment(FLOAT_SPECIES, a, off, ByteOrder.LITTLE_ENDIAN);
        FloatVector vb =
            FloatVector.fromMemorySegment(FLOAT_SPECIES, b, off, ByteOrder.LITTLE_ENDIAN);
        vs = fma(va, vb, vs);
        vn1 = fma(va, va, vn1);
        vn2 = fma(vb, vb, vn2);
      }
      sum = vs.reduceLanes(VectorOperators.ADD);
      norm1 = vn1.reduceLanes(VectorOperators.ADD);
      norm2 = vn2.reduceLanes(VectorOperators.ADD);
    }

    // Scalar tail
    for (; i < dimensions; i++) {
      float ai = a.getAtIndex(ValueLayout.JAVA_FLOAT, i);
      float bi = b.getAtIndex(ValueLayout.JAVA_FLOAT, i);
      sum = MathUtil.fma(ai, bi, sum);
      norm1 = MathUtil.fma(ai, ai, norm1);
      norm2 = MathUtil.fma(bi, bi, norm2);
    }

    return (float) (sum / Math.sqrt((double) norm1 * (double) norm2));
  }

  // --- Hamming distance: XOR + BIT_COUNT on LongVector ---

  @Override
  public int hammingDistance(long[] a, long[] b) {
    int i = 0;
    long dist = 0;

    if (a.length >= LONG_SPECIES.length()) {
      int limit = LONG_SPECIES.loopBound(a.length);
      LongVector acc = LongVector.zero(LONG_SPECIES);

      for (; i < limit; i += LONG_SPECIES.length()) {
        LongVector va = LongVector.fromArray(LONG_SPECIES, a, i);
        LongVector vb = LongVector.fromArray(LONG_SPECIES, b, i);
        acc = acc.add(va.lanewise(VectorOperators.XOR, vb).lanewise(VectorOperators.BIT_COUNT));
      }
      dist = acc.reduceLanes(VectorOperators.ADD);
    }

    // Scalar tail
    for (; i < a.length; i++) {
      dist += Long.bitCount(a[i] ^ b[i]);
    }
    return (int) dist;
  }

  // --- Vector arithmetic ---

  @Override
  public void addInPlace(float[] v1, float[] v2) {
    int i = 0;
    int limit = FLOAT_SPECIES.loopBound(v1.length);
    for (; i < limit; i += FLOAT_SPECIES.length()) {
      FloatVector va = FloatVector.fromArray(FLOAT_SPECIES, v1, i);
      FloatVector vb = FloatVector.fromArray(FLOAT_SPECIES, v2, i);
      va.add(vb).intoArray(v1, i);
    }
    for (; i < v1.length; i++) {
      v1[i] += v2[i];
    }
  }

  @Override
  public void addScaledInPlace(
      float[] out, int outOffset, float[] vector, int vectorOffset, int length, float scale) {
    int i = 0;
    int limit = FLOAT_SPECIES.loopBound(length);
    FloatVector scaleVector = FloatVector.broadcast(FLOAT_SPECIES, scale);
    for (; i < limit; i += FLOAT_SPECIES.length()) {
      FloatVector outVector = FloatVector.fromArray(FLOAT_SPECIES, out, outOffset + i);
      FloatVector addend = FloatVector.fromArray(FLOAT_SPECIES, vector, vectorOffset + i);
      fma(addend, scaleVector, outVector).intoArray(out, outOffset + i);
    }
    for (; i < length; i++) {
      int outIndex = outOffset + i;
      out[outIndex] = MathUtil.fma(vector[vectorOffset + i], scale, out[outIndex]);
    }
  }

  @Override
  public void addWeightedRowsInPlace(
      float[] out,
      int outOffset,
      float[] matrix,
      int matrixOffset,
      int rowStride,
      float[] weights,
      int weightsOffset,
      int rows,
      int columns) {
    int row = 0;
    int fourRowLimit = rows & ~3;
    int vectorLimit = FLOAT_SPECIES.loopBound(columns);
    for (; row < fourRowLimit; row += 4) {
      int row0 = matrixOffset + row * rowStride;
      int row1 = row0 + rowStride;
      int row2 = row1 + rowStride;
      int row3 = row2 + rowStride;
      FloatVector weight0 = FloatVector.broadcast(FLOAT_SPECIES, weights[weightsOffset + row]);
      FloatVector weight1 = FloatVector.broadcast(FLOAT_SPECIES, weights[weightsOffset + row + 1]);
      FloatVector weight2 = FloatVector.broadcast(FLOAT_SPECIES, weights[weightsOffset + row + 2]);
      FloatVector weight3 = FloatVector.broadcast(FLOAT_SPECIES, weights[weightsOffset + row + 3]);

      int column = 0;
      for (; column < vectorLimit; column += FLOAT_SPECIES.length()) {
        FloatVector result = FloatVector.fromArray(FLOAT_SPECIES, out, outOffset + column);
        result = fma(FloatVector.fromArray(FLOAT_SPECIES, matrix, row0 + column), weight0, result);
        result = fma(FloatVector.fromArray(FLOAT_SPECIES, matrix, row1 + column), weight1, result);
        result = fma(FloatVector.fromArray(FLOAT_SPECIES, matrix, row2 + column), weight2, result);
        result = fma(FloatVector.fromArray(FLOAT_SPECIES, matrix, row3 + column), weight3, result);
        result.intoArray(out, outOffset + column);
      }
      for (; column < columns; column++) {
        int outIndex = outOffset + column;
        float result = out[outIndex];
        result = MathUtil.fma(matrix[row0 + column], weights[weightsOffset + row], result);
        result = MathUtil.fma(matrix[row1 + column], weights[weightsOffset + row + 1], result);
        result = MathUtil.fma(matrix[row2 + column], weights[weightsOffset + row + 2], result);
        result = MathUtil.fma(matrix[row3 + column], weights[weightsOffset + row + 3], result);
        out[outIndex] = result;
      }
    }
    for (; row < rows; row++) {
      addScaledInPlace(
          out,
          outOffset,
          matrix,
          matrixOffset + row * rowStride,
          columns,
          weights[weightsOffset + row]);
    }
  }

  @Override
  public void subInPlace(float[] v1, float[] v2) {
    int i = 0;
    int limit = FLOAT_SPECIES.loopBound(v1.length);
    for (; i < limit; i += FLOAT_SPECIES.length()) {
      FloatVector va = FloatVector.fromArray(FLOAT_SPECIES, v1, i);
      FloatVector vb = FloatVector.fromArray(FLOAT_SPECIES, v2, i);
      va.sub(vb).intoArray(v1, i);
    }
    for (; i < v1.length; i++) {
      v1[i] -= v2[i];
    }
  }

  @Override
  public void scale(float[] vector, float multiplier) {
    int i = 0;
    int limit = FLOAT_SPECIES.loopBound(vector.length);
    FloatVector mul = FloatVector.broadcast(FLOAT_SPECIES, multiplier);
    for (; i < limit; i += FLOAT_SPECIES.length()) {
      FloatVector v = FloatVector.fromArray(FLOAT_SPECIES, vector, i);
      v.mul(mul).intoArray(vector, i);
    }
    for (; i < vector.length; i++) {
      vector[i] *= multiplier;
    }
  }

  @Override
  public float sum(float[] vector) {
    int i = 0;
    float res = 0f;
    int limit = FLOAT_SPECIES.loopBound(vector.length);
    if (limit > 0) {
      FloatVector acc = FloatVector.zero(FLOAT_SPECIES);
      for (; i < limit; i += FLOAT_SPECIES.length()) {
        acc = acc.add(FloatVector.fromArray(FLOAT_SPECIES, vector, i));
      }
      res = acc.reduceLanes(VectorOperators.ADD);
    }
    for (; i < vector.length; i++) {
      res += vector[i];
    }
    return res;
  }

  // --- L2 normalization: 4x unrolled scale ---

  @Override
  public float[] l2normalize(float[] v, boolean throwOnZero) {
    double squaredNorm = dotProduct(v, v);
    if (squaredNorm == 0.0) {
      if (throwOnZero) {
        throw new IllegalArgumentException("Cannot normalize a zero-length vector");
      }
      return v;
    }
    if (MathUtil.isUnitVector((float) squaredNorm)) {
      return v;
    }
    float invNorm = 1.0f / (float) Math.sqrt(squaredNorm);
    int i = 0;

    // Vectorize if large enough
    if (v.length > 2 * FLOAT_SPECIES.length()) {
      int limit = FLOAT_SPECIES.loopBound(v.length);
      l2normalizeBody(v, invNorm, limit);
      i = limit;
    }

    for (; i < v.length; i++) {
      v[i] *= invNorm;
    }
    return v;
  }

  private void l2normalizeBody(float[] v, float invNorm, int limit) {
    FloatVector invNormVector = FloatVector.broadcast(FLOAT_SPECIES, invNorm);
    int i = 0;
    int unrolledLimit = limit - 3 * FLOAT_SPECIES.length();

    // 4x unrolled scale
    for (; i < unrolledLimit; i += 4 * FLOAT_SPECIES.length()) {
      FloatVector.fromArray(FLOAT_SPECIES, v, i).mul(invNormVector).intoArray(v, i);
      FloatVector.fromArray(FLOAT_SPECIES, v, i + FLOAT_SPECIES.length())
          .mul(invNormVector)
          .intoArray(v, i + FLOAT_SPECIES.length());
      FloatVector.fromArray(FLOAT_SPECIES, v, i + 2 * FLOAT_SPECIES.length())
          .mul(invNormVector)
          .intoArray(v, i + 2 * FLOAT_SPECIES.length());
      FloatVector.fromArray(FLOAT_SPECIES, v, i + 3 * FLOAT_SPECIES.length())
          .mul(invNormVector)
          .intoArray(v, i + 3 * FLOAT_SPECIES.length());
    }

    for (; i < limit; i += FLOAT_SPECIES.length()) {
      FloatVector.fromArray(FLOAT_SPECIES, v, i).mul(invNormVector).intoArray(v, i);
    }
  }

  // --- Fused batch matrix-vector kernels (GEMV) ---

  /** Two-row strided GEMV with the same accumulator chains as the independent dot kernel. */
  @Override
  public void matVecDotExact(
      float[] query,
      int queryOffset,
      float[] matrix,
      int matrixOffset,
      int rowStride,
      int rows,
      int columns,
      float[] out,
      int outOffset) {
    int pairedRows = rows & ~1;
    int speciesLength = FLOAT_SPECIES.length();

    for (int row = 0; row < pairedRows; row += 2) {
      int base0 = matrixOffset + row * rowStride;
      int base1 = base0 + rowStride;
      int column = 0;
      float sum0 = 0.0f;
      float sum1 = 0.0f;

      if (columns > 2 * speciesLength) {
        int vectorLimit = FLOAT_SPECIES.loopBound(columns);
        int unrolledLimit = vectorLimit - 3 * speciesLength;
        FloatVector acc00 = FloatVector.zero(FLOAT_SPECIES);
        FloatVector acc01 = FloatVector.zero(FLOAT_SPECIES);
        FloatVector acc02 = FloatVector.zero(FLOAT_SPECIES);
        FloatVector acc03 = FloatVector.zero(FLOAT_SPECIES);
        FloatVector acc10 = FloatVector.zero(FLOAT_SPECIES);
        FloatVector acc11 = FloatVector.zero(FLOAT_SPECIES);
        FloatVector acc12 = FloatVector.zero(FLOAT_SPECIES);
        FloatVector acc13 = FloatVector.zero(FLOAT_SPECIES);

        for (; column < unrolledLimit; column += 4 * speciesLength) {
          FloatVector query0 = FloatVector.fromArray(FLOAT_SPECIES, query, queryOffset + column);
          acc00 = fma(query0, FloatVector.fromArray(FLOAT_SPECIES, matrix, base0 + column), acc00);
          acc10 = fma(query0, FloatVector.fromArray(FLOAT_SPECIES, matrix, base1 + column), acc10);

          FloatVector query1 =
              FloatVector.fromArray(FLOAT_SPECIES, query, queryOffset + column + speciesLength);
          acc01 =
              fma(
                  query1,
                  FloatVector.fromArray(FLOAT_SPECIES, matrix, base0 + column + speciesLength),
                  acc01);
          acc11 =
              fma(
                  query1,
                  FloatVector.fromArray(FLOAT_SPECIES, matrix, base1 + column + speciesLength),
                  acc11);

          FloatVector query2 =
              FloatVector.fromArray(FLOAT_SPECIES, query, queryOffset + column + 2 * speciesLength);
          acc02 =
              fma(
                  query2,
                  FloatVector.fromArray(FLOAT_SPECIES, matrix, base0 + column + 2 * speciesLength),
                  acc02);
          acc12 =
              fma(
                  query2,
                  FloatVector.fromArray(FLOAT_SPECIES, matrix, base1 + column + 2 * speciesLength),
                  acc12);

          FloatVector query3 =
              FloatVector.fromArray(FLOAT_SPECIES, query, queryOffset + column + 3 * speciesLength);
          acc03 =
              fma(
                  query3,
                  FloatVector.fromArray(FLOAT_SPECIES, matrix, base0 + column + 3 * speciesLength),
                  acc03);
          acc13 =
              fma(
                  query3,
                  FloatVector.fromArray(FLOAT_SPECIES, matrix, base1 + column + 3 * speciesLength),
                  acc13);
        }

        for (; column < vectorLimit; column += speciesLength) {
          FloatVector queryVector =
              FloatVector.fromArray(FLOAT_SPECIES, query, queryOffset + column);
          acc00 =
              fma(queryVector, FloatVector.fromArray(FLOAT_SPECIES, matrix, base0 + column), acc00);
          acc10 =
              fma(queryVector, FloatVector.fromArray(FLOAT_SPECIES, matrix, base1 + column), acc10);
        }

        FloatVector row0First = acc00.add(acc01);
        FloatVector row0Second = acc02.add(acc03);
        FloatVector row1First = acc10.add(acc11);
        FloatVector row1Second = acc12.add(acc13);
        sum0 += reduceAdd(row0First.add(row0Second));
        sum1 += reduceAdd(row1First.add(row1Second));
      }

      for (; column < columns; column++) {
        float queryValue = query[queryOffset + column];
        sum0 = MathUtil.fma(queryValue, matrix[base0 + column], sum0);
        sum1 = MathUtil.fma(queryValue, matrix[base1 + column], sum1);
      }
      out[outOffset + row] = sum0;
      out[outOffset + row + 1] = sum1;
    }

    for (int row = pairedRows; row < rows; row++) {
      out[outOffset + row] =
          dotProduct(query, queryOffset, matrix, matrixOffset + row * rowStride, columns);
    }
  }

  /**
   * SIMD 4-row-unrolled fused GEMV for dot product.
   *
   * <p>For each group of 4 rows the inner dimension loop loads one SIMD chunk of {@code query}
   * <em>once</em> and multiplies it with the corresponding chunk of all 4 rows simultaneously. This
   * cuts query memory traffic by 4× vs calling {@link #dotProduct} per row and saturates 4 FMA
   * execution ports with independent chains.
   *
   * <p>Rows that do not form a full group of 4 fall back to {@link #dotProduct}.
   */
  @Override
  public void matVecDot(float[] query, float[][] matrix, float[] out, int numRows) {
    int dim = query.length;
    int rowGroup = numRows & ~3;
    int limit = FLOAT_SPECIES.loopBound(dim);

    for (int r = 0; r < rowGroup; r += 4) {
      float[] r0 = matrix[r], r1 = matrix[r + 1], r2 = matrix[r + 2], r3 = matrix[r + 3];
      FloatVector acc0 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector acc1 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector acc2 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector acc3 = FloatVector.zero(FLOAT_SPECIES);

      for (int i = 0; i < limit; i += FLOAT_SPECIES.length()) {
        FloatVector qv = FloatVector.fromArray(FLOAT_SPECIES, query, i);
        acc0 = fma(qv, FloatVector.fromArray(FLOAT_SPECIES, r0, i), acc0);
        acc1 = fma(qv, FloatVector.fromArray(FLOAT_SPECIES, r1, i), acc1);
        acc2 = fma(qv, FloatVector.fromArray(FLOAT_SPECIES, r2, i), acc2);
        acc3 = fma(qv, FloatVector.fromArray(FLOAT_SPECIES, r3, i), acc3);
      }

      float s0 = acc0.reduceLanes(VectorOperators.ADD);
      float s1 = acc1.reduceLanes(VectorOperators.ADD);
      float s2 = acc2.reduceLanes(VectorOperators.ADD);
      float s3 = acc3.reduceLanes(VectorOperators.ADD);

      // Scalar tail (remaining elements after SIMD loop bound)
      for (int i = limit; i < dim; i++) {
        float q = query[i];
        s0 = MathUtil.fma(q, r0[i], s0);
        s1 = MathUtil.fma(q, r1[i], s1);
        s2 = MathUtil.fma(q, r2[i], s2);
        s3 = MathUtil.fma(q, r3[i], s3);
      }

      out[r] = s0;
      out[r + 1] = s1;
      out[r + 2] = s2;
      out[r + 3] = s3;
    }

    // Tail rows (0–3 remaining)
    for (int r = rowGroup; r < numRows; r++) {
      out[r] = dotProduct(query, 0, matrix[r], 0, dim);
    }
  }

  /**
   * SIMD 4-row-unrolled fused GEMV for a flat row-major matrix. This is the dense-tensor variant of
   * {@link #matVecDot(float[], float[][], float[], int)} and avoids allocating per-row arrays when
   * the caller already stores rows contiguously.
   */
  @Override
  public void matVecDot(float[] query, float[] rowMajorMatrix, int rows, int cols, float[] out) {
    int rowGroup = rows & ~3;
    int limit = FLOAT_SPECIES.loopBound(cols);

    for (int r = 0; r < rowGroup; r += 4) {
      int base0 = r * cols;
      int base1 = base0 + cols;
      int base2 = base1 + cols;
      int base3 = base2 + cols;
      FloatVector acc0 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector acc1 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector acc2 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector acc3 = FloatVector.zero(FLOAT_SPECIES);

      for (int i = 0; i < limit; i += FLOAT_SPECIES.length()) {
        FloatVector qv = FloatVector.fromArray(FLOAT_SPECIES, query, i);
        acc0 = fma(qv, FloatVector.fromArray(FLOAT_SPECIES, rowMajorMatrix, base0 + i), acc0);
        acc1 = fma(qv, FloatVector.fromArray(FLOAT_SPECIES, rowMajorMatrix, base1 + i), acc1);
        acc2 = fma(qv, FloatVector.fromArray(FLOAT_SPECIES, rowMajorMatrix, base2 + i), acc2);
        acc3 = fma(qv, FloatVector.fromArray(FLOAT_SPECIES, rowMajorMatrix, base3 + i), acc3);
      }

      float s0 = acc0.reduceLanes(VectorOperators.ADD);
      float s1 = acc1.reduceLanes(VectorOperators.ADD);
      float s2 = acc2.reduceLanes(VectorOperators.ADD);
      float s3 = acc3.reduceLanes(VectorOperators.ADD);

      for (int i = limit; i < cols; i++) {
        float q = query[i];
        s0 = MathUtil.fma(q, rowMajorMatrix[base0 + i], s0);
        s1 = MathUtil.fma(q, rowMajorMatrix[base1 + i], s1);
        s2 = MathUtil.fma(q, rowMajorMatrix[base2 + i], s2);
        s3 = MathUtil.fma(q, rowMajorMatrix[base3 + i], s3);
      }

      out[r] = s0;
      out[r + 1] = s1;
      out[r + 2] = s2;
      out[r + 3] = s3;
    }

    for (int r = rowGroup; r < rows; r++) {
      out[r] = dotProduct(query, 0, rowMajorMatrix, r * cols, cols);
    }
  }

  /** SIMD 4-row-unrolled GEMV over little-endian mapped GGUF F32 weights. */
  @Override
  public void ggufF32MatVecDot(
      float[] query, MemorySegment weight, int rows, int cols, float[] out) {
    int rowGroup = rows & ~3;
    int limit = FLOAT_SPECIES.loopBound(cols);
    long rowBytes = (long) cols * Float.BYTES;

    for (int row = 0; row < rowGroup; row += 4) {
      long base0 = row * rowBytes;
      long base1 = base0 + rowBytes;
      long base2 = base1 + rowBytes;
      long base3 = base2 + rowBytes;
      FloatVector acc0 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector acc1 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector acc2 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector acc3 = FloatVector.zero(FLOAT_SPECIES);

      for (int col = 0; col < limit; col += FLOAT_SPECIES.length()) {
        FloatVector queryVector = FloatVector.fromArray(FLOAT_SPECIES, query, col);
        long byteOffset = (long) col * Float.BYTES;
        acc0 =
            fma(
                queryVector,
                FloatVector.fromMemorySegment(
                    FLOAT_SPECIES, weight, base0 + byteOffset, ByteOrder.LITTLE_ENDIAN),
                acc0);
        acc1 =
            fma(
                queryVector,
                FloatVector.fromMemorySegment(
                    FLOAT_SPECIES, weight, base1 + byteOffset, ByteOrder.LITTLE_ENDIAN),
                acc1);
        acc2 =
            fma(
                queryVector,
                FloatVector.fromMemorySegment(
                    FLOAT_SPECIES, weight, base2 + byteOffset, ByteOrder.LITTLE_ENDIAN),
                acc2);
        acc3 =
            fma(
                queryVector,
                FloatVector.fromMemorySegment(
                    FLOAT_SPECIES, weight, base3 + byteOffset, ByteOrder.LITTLE_ENDIAN),
                acc3);
      }

      float sum0 = acc0.reduceLanes(VectorOperators.ADD);
      float sum1 = acc1.reduceLanes(VectorOperators.ADD);
      float sum2 = acc2.reduceLanes(VectorOperators.ADD);
      float sum3 = acc3.reduceLanes(VectorOperators.ADD);
      for (int col = limit; col < cols; col++) {
        float queryValue = query[col];
        long byteOffset = (long) col * Float.BYTES;
        sum0 = MathUtil.fma(queryValue, weight.get(GGUF_LE_FLOAT, base0 + byteOffset), sum0);
        sum1 = MathUtil.fma(queryValue, weight.get(GGUF_LE_FLOAT, base1 + byteOffset), sum1);
        sum2 = MathUtil.fma(queryValue, weight.get(GGUF_LE_FLOAT, base2 + byteOffset), sum2);
        sum3 = MathUtil.fma(queryValue, weight.get(GGUF_LE_FLOAT, base3 + byteOffset), sum3);
      }

      out[row] = sum0;
      out[row + 1] = sum1;
      out[row + 2] = sum2;
      out[row + 3] = sum3;
    }

    for (int row = rowGroup; row < rows; row++) {
      long base = row * rowBytes;
      FloatVector accumulator = FloatVector.zero(FLOAT_SPECIES);
      for (int col = 0; col < limit; col += FLOAT_SPECIES.length()) {
        accumulator =
            fma(
                FloatVector.fromArray(FLOAT_SPECIES, query, col),
                FloatVector.fromMemorySegment(
                    FLOAT_SPECIES,
                    weight,
                    base + (long) col * Float.BYTES,
                    ByteOrder.LITTLE_ENDIAN),
                accumulator);
      }
      float sum = accumulator.reduceLanes(VectorOperators.ADD);
      for (int col = limit; col < cols; col++) {
        sum =
            MathUtil.fma(
                query[col], weight.get(GGUF_LE_FLOAT, base + (long) col * Float.BYTES), sum);
      }
      out[row] = sum;
    }
  }

  /**
   * SIMD 4-row-unrolled fused GEMV for squared L2 distance.
   *
   * <p>Same strategy as {@link #matVecDot}: loads each query SIMD chunk once and accumulates
   * squared differences for 4 rows simultaneously.
   */
  @Override
  public void matVecSquaredL2(float[] query, float[][] matrix, float[] out, int numRows) {
    int dim = query.length;
    int rowGroup = numRows & ~3;
    int limit = FLOAT_SPECIES.loopBound(dim);

    for (int r = 0; r < rowGroup; r += 4) {
      float[] r0 = matrix[r], r1 = matrix[r + 1], r2 = matrix[r + 2], r3 = matrix[r + 3];
      FloatVector acc0 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector acc1 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector acc2 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector acc3 = FloatVector.zero(FLOAT_SPECIES);

      for (int i = 0; i < limit; i += FLOAT_SPECIES.length()) {
        FloatVector qv = FloatVector.fromArray(FLOAT_SPECIES, query, i);
        FloatVector d0 = qv.sub(FloatVector.fromArray(FLOAT_SPECIES, r0, i));
        FloatVector d1 = qv.sub(FloatVector.fromArray(FLOAT_SPECIES, r1, i));
        FloatVector d2 = qv.sub(FloatVector.fromArray(FLOAT_SPECIES, r2, i));
        FloatVector d3 = qv.sub(FloatVector.fromArray(FLOAT_SPECIES, r3, i));
        acc0 = fma(d0, d0, acc0);
        acc1 = fma(d1, d1, acc1);
        acc2 = fma(d2, d2, acc2);
        acc3 = fma(d3, d3, acc3);
      }

      float s0 = acc0.reduceLanes(VectorOperators.ADD);
      float s1 = acc1.reduceLanes(VectorOperators.ADD);
      float s2 = acc2.reduceLanes(VectorOperators.ADD);
      float s3 = acc3.reduceLanes(VectorOperators.ADD);

      for (int i = limit; i < dim; i++) {
        float q = query[i];
        float e0 = q - r0[i];
        s0 = MathUtil.fma(e0, e0, s0);
        float e1 = q - r1[i];
        s1 = MathUtil.fma(e1, e1, s1);
        float e2 = q - r2[i];
        s2 = MathUtil.fma(e2, e2, s2);
        float e3 = q - r3[i];
        s3 = MathUtil.fma(e3, e3, s3);
      }

      out[r] = s0;
      out[r + 1] = s1;
      out[r + 2] = s2;
      out[r + 3] = s3;
    }

    for (int r = rowGroup; r < numRows; r++) {
      out[r] = squareDistance(query, 0, matrix[r], 0, dim);
    }
  }

  /**
   * Off-heap SIMD 4-row-unrolled fused GEMV for dot product. Mirrors {@link #matVecDot(float[],
   * float[][], float[], int)} exactly, but reads each of the 4 rows from a {@link MemorySegment}
   * (zero-copy mmap/off-heap slice) via {@code FloatVector.fromMemorySegment} instead of {@code
   * fromArray}. The query is still loaded once per 4-row group from the on-heap {@code float[]}, so
   * the segment-scoring path gets the same 4× query-load amortization as the {@code float[][]}
   * path.
   */
  @Override
  public void matVecDot(float[] query, MemorySegment[] rows, int dim, float[] out, int count) {
    int rowGroup = count & ~3;
    int limit = FLOAT_SPECIES.loopBound(dim);

    for (int r = 0; r < rowGroup; r += 4) {
      MemorySegment r0 = rows[r], r1 = rows[r + 1], r2 = rows[r + 2], r3 = rows[r + 3];
      FloatVector acc0 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector acc1 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector acc2 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector acc3 = FloatVector.zero(FLOAT_SPECIES);

      for (int i = 0; i < limit; i += FLOAT_SPECIES.length()) {
        long byteOff = (long) i * Float.BYTES;
        FloatVector qv = FloatVector.fromArray(FLOAT_SPECIES, query, i);
        acc0 =
            fma(
                qv,
                FloatVector.fromMemorySegment(FLOAT_SPECIES, r0, byteOff, ByteOrder.LITTLE_ENDIAN),
                acc0);
        acc1 =
            fma(
                qv,
                FloatVector.fromMemorySegment(FLOAT_SPECIES, r1, byteOff, ByteOrder.LITTLE_ENDIAN),
                acc1);
        acc2 =
            fma(
                qv,
                FloatVector.fromMemorySegment(FLOAT_SPECIES, r2, byteOff, ByteOrder.LITTLE_ENDIAN),
                acc2);
        acc3 =
            fma(
                qv,
                FloatVector.fromMemorySegment(FLOAT_SPECIES, r3, byteOff, ByteOrder.LITTLE_ENDIAN),
                acc3);
      }

      float s0 = acc0.reduceLanes(VectorOperators.ADD);
      float s1 = acc1.reduceLanes(VectorOperators.ADD);
      float s2 = acc2.reduceLanes(VectorOperators.ADD);
      float s3 = acc3.reduceLanes(VectorOperators.ADD);

      // Scalar tail (remaining elements after SIMD loop bound)
      for (int i = limit; i < dim; i++) {
        float q = query[i];
        s0 = MathUtil.fma(q, r0.getAtIndex(ValueLayout.JAVA_FLOAT, i), s0);
        s1 = MathUtil.fma(q, r1.getAtIndex(ValueLayout.JAVA_FLOAT, i), s1);
        s2 = MathUtil.fma(q, r2.getAtIndex(ValueLayout.JAVA_FLOAT, i), s2);
        s3 = MathUtil.fma(q, r3.getAtIndex(ValueLayout.JAVA_FLOAT, i), s3);
      }

      out[r] = s0;
      out[r + 1] = s1;
      out[r + 2] = s2;
      out[r + 3] = s3;
    }

    // Tail rows (0-3 remaining)
    for (int r = rowGroup; r < count; r++) {
      out[r] = dotArraySeg(query, rows[r], dim, limit);
    }
  }

  /**
   * Off-heap SIMD 4-row-unrolled fused GEMV for squared L2 distance. Mirrors {@link
   * #matVecSquaredL2(float[], float[][], float[], int)}, reading rows from {@link MemorySegment}s.
   */
  @Override
  public void matVecSquaredL2(
      float[] query, MemorySegment[] rows, int dim, float[] out, int count) {
    int rowGroup = count & ~3;
    int limit = FLOAT_SPECIES.loopBound(dim);

    for (int r = 0; r < rowGroup; r += 4) {
      MemorySegment r0 = rows[r], r1 = rows[r + 1], r2 = rows[r + 2], r3 = rows[r + 3];
      FloatVector acc0 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector acc1 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector acc2 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector acc3 = FloatVector.zero(FLOAT_SPECIES);

      for (int i = 0; i < limit; i += FLOAT_SPECIES.length()) {
        long byteOff = (long) i * Float.BYTES;
        FloatVector qv = FloatVector.fromArray(FLOAT_SPECIES, query, i);
        FloatVector d0 =
            qv.sub(
                FloatVector.fromMemorySegment(FLOAT_SPECIES, r0, byteOff, ByteOrder.LITTLE_ENDIAN));
        FloatVector d1 =
            qv.sub(
                FloatVector.fromMemorySegment(FLOAT_SPECIES, r1, byteOff, ByteOrder.LITTLE_ENDIAN));
        FloatVector d2 =
            qv.sub(
                FloatVector.fromMemorySegment(FLOAT_SPECIES, r2, byteOff, ByteOrder.LITTLE_ENDIAN));
        FloatVector d3 =
            qv.sub(
                FloatVector.fromMemorySegment(FLOAT_SPECIES, r3, byteOff, ByteOrder.LITTLE_ENDIAN));
        acc0 = fma(d0, d0, acc0);
        acc1 = fma(d1, d1, acc1);
        acc2 = fma(d2, d2, acc2);
        acc3 = fma(d3, d3, acc3);
      }

      float s0 = acc0.reduceLanes(VectorOperators.ADD);
      float s1 = acc1.reduceLanes(VectorOperators.ADD);
      float s2 = acc2.reduceLanes(VectorOperators.ADD);
      float s3 = acc3.reduceLanes(VectorOperators.ADD);

      for (int i = limit; i < dim; i++) {
        float q = query[i];
        float e0 = q - r0.getAtIndex(ValueLayout.JAVA_FLOAT, i);
        s0 = MathUtil.fma(e0, e0, s0);
        float e1 = q - r1.getAtIndex(ValueLayout.JAVA_FLOAT, i);
        s1 = MathUtil.fma(e1, e1, s1);
        float e2 = q - r2.getAtIndex(ValueLayout.JAVA_FLOAT, i);
        s2 = MathUtil.fma(e2, e2, s2);
        float e3 = q - r3.getAtIndex(ValueLayout.JAVA_FLOAT, i);
        s3 = MathUtil.fma(e3, e3, s3);
      }

      out[r] = s0;
      out[r + 1] = s1;
      out[r + 2] = s2;
      out[r + 3] = s3;
    }

    for (int r = rowGroup; r < count; r++) {
      out[r] = squaredL2ArraySeg(query, rows[r], dim, limit);
    }
  }

  /** Single-row {@code float[]}-query vs {@code MemorySegment}-row dot product (GEMV tail rows). */
  private float dotArraySeg(float[] query, MemorySegment row, int dim, int limit) {
    FloatVector acc = FloatVector.zero(FLOAT_SPECIES);
    for (int i = 0; i < limit; i += FLOAT_SPECIES.length()) {
      acc =
          fma(
              FloatVector.fromArray(FLOAT_SPECIES, query, i),
              FloatVector.fromMemorySegment(
                  FLOAT_SPECIES, row, (long) i * Float.BYTES, ByteOrder.LITTLE_ENDIAN),
              acc);
    }
    float s = acc.reduceLanes(VectorOperators.ADD);
    for (int i = limit; i < dim; i++) {
      s = MathUtil.fma(query[i], row.getAtIndex(ValueLayout.JAVA_FLOAT, i), s);
    }
    return s;
  }

  /** Single-row {@code float[]}-query vs {@code MemorySegment}-row squared L2 (GEMV tail rows). */
  private float squaredL2ArraySeg(float[] query, MemorySegment row, int dim, int limit) {
    FloatVector acc = FloatVector.zero(FLOAT_SPECIES);
    for (int i = 0; i < limit; i += FLOAT_SPECIES.length()) {
      FloatVector d =
          FloatVector.fromArray(FLOAT_SPECIES, query, i)
              .sub(
                  FloatVector.fromMemorySegment(
                      FLOAT_SPECIES, row, (long) i * Float.BYTES, ByteOrder.LITTLE_ENDIAN));
      acc = fma(d, d, acc);
    }
    float s = acc.reduceLanes(VectorOperators.ADD);
    for (int i = limit; i < dim; i++) {
      float e = query[i] - row.getAtIndex(ValueLayout.JAVA_FLOAT, i);
      s = MathUtil.fma(e, e, s);
    }
    return s;
  }

  // --- Fused batch cosine kernels ---

  /**
   * Combines a per-row dot product and row norm with the shared query norm into a raw cosine value,
   * exactly matching the arithmetic of the per-row {@link #cosine(float[], float[])} reference
   * ({@code (float)(dot / sqrt(qNorm2 * rowNorm2))}). A zero row or zero query gives {@code 0/0 =
   * NaN}, identical to the per-row kernel.
   */
  private static float cosineValue(float dot, float qNorm2, float rowNorm2) {
    return (float) (dot / Math.sqrt((double) qNorm2 * (double) rowNorm2));
  }

  /**
   * SIMD 4-row-unrolled fused cosine GEMV. The squared query norm {@code ‖query‖²} is computed ONCE
   * (single reduction over {@code query}); then each group of 4 rows loads one SIMD chunk of {@code
   * query} <em>once</em> and accumulates, for all 4 rows simultaneously, the dot product {@code
   * fma(qv, rowv, dot[r])} and the row norm {@code fma(rowv, rowv, rn[r])}. This mirrors {@link
   * #matVecDot(float[], float[][], float[], int)} but carries a second (row-norm) accumulator set
   * plus the single shared query norm, so non-normalized cosine collections get the same 4×
   * query-load amortization DOT/EUCLIDEAN already have.
   *
   * <p>Rows that do not form a full group of 4 fall back to a single-row fused pass sharing the
   * same {@code qNorm2}.
   */
  @Override
  public void batchCosine(float[] query, float[][] rows, float[] out, int count) {
    int dim = query.length;
    float qNorm2 = dotProduct(query, 0, query, 0, dim); // ‖query‖² — computed once for the batch
    int rowGroup = count & ~3;
    int limit = FLOAT_SPECIES.loopBound(dim);

    for (int r = 0; r < rowGroup; r += 4) {
      float[] r0 = rows[r], r1 = rows[r + 1], r2 = rows[r + 2], r3 = rows[r + 3];
      FloatVector d0 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector d1 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector d2 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector d3 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector n0 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector n1 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector n2 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector n3 = FloatVector.zero(FLOAT_SPECIES);

      for (int i = 0; i < limit; i += FLOAT_SPECIES.length()) {
        FloatVector qv = FloatVector.fromArray(FLOAT_SPECIES, query, i); // query chunk loaded ONCE
        FloatVector v0 = FloatVector.fromArray(FLOAT_SPECIES, r0, i);
        FloatVector v1 = FloatVector.fromArray(FLOAT_SPECIES, r1, i);
        FloatVector v2 = FloatVector.fromArray(FLOAT_SPECIES, r2, i);
        FloatVector v3 = FloatVector.fromArray(FLOAT_SPECIES, r3, i);
        d0 = fma(qv, v0, d0);
        d1 = fma(qv, v1, d1);
        d2 = fma(qv, v2, d2);
        d3 = fma(qv, v3, d3);
        n0 = fma(v0, v0, n0);
        n1 = fma(v1, v1, n1);
        n2 = fma(v2, v2, n2);
        n3 = fma(v3, v3, n3);
      }

      float dot0 = d0.reduceLanes(VectorOperators.ADD);
      float dot1 = d1.reduceLanes(VectorOperators.ADD);
      float dot2 = d2.reduceLanes(VectorOperators.ADD);
      float dot3 = d3.reduceLanes(VectorOperators.ADD);
      float rn0 = n0.reduceLanes(VectorOperators.ADD);
      float rn1 = n1.reduceLanes(VectorOperators.ADD);
      float rn2 = n2.reduceLanes(VectorOperators.ADD);
      float rn3 = n3.reduceLanes(VectorOperators.ADD);

      // Scalar tail (remaining elements after SIMD loop bound)
      for (int i = limit; i < dim; i++) {
        float q = query[i];
        float e0 = r0[i];
        dot0 = MathUtil.fma(q, e0, dot0);
        rn0 = MathUtil.fma(e0, e0, rn0);
        float e1 = r1[i];
        dot1 = MathUtil.fma(q, e1, dot1);
        rn1 = MathUtil.fma(e1, e1, rn1);
        float e2 = r2[i];
        dot2 = MathUtil.fma(q, e2, dot2);
        rn2 = MathUtil.fma(e2, e2, rn2);
        float e3 = r3[i];
        dot3 = MathUtil.fma(q, e3, dot3);
        rn3 = MathUtil.fma(e3, e3, rn3);
      }

      out[r] = cosineValue(dot0, qNorm2, rn0);
      out[r + 1] = cosineValue(dot1, qNorm2, rn1);
      out[r + 2] = cosineValue(dot2, qNorm2, rn2);
      out[r + 3] = cosineValue(dot3, qNorm2, rn3);
    }

    // Tail rows (0-3 remaining)
    for (int r = rowGroup; r < count; r++) {
      out[r] = cosineArrayRow(query, rows[r], qNorm2, limit, dim);
    }
  }

  /** Single-row fused cosine over {@code float[]} rows sharing the batch {@code qNorm2}. */
  private float cosineArrayRow(float[] query, float[] row, float qNorm2, int limit, int dim) {
    FloatVector dot = FloatVector.zero(FLOAT_SPECIES);
    FloatVector rn = FloatVector.zero(FLOAT_SPECIES);
    for (int i = 0; i < limit; i += FLOAT_SPECIES.length()) {
      FloatVector qv = FloatVector.fromArray(FLOAT_SPECIES, query, i);
      FloatVector rv = FloatVector.fromArray(FLOAT_SPECIES, row, i);
      dot = fma(qv, rv, dot);
      rn = fma(rv, rv, rn);
    }
    float d = dot.reduceLanes(VectorOperators.ADD);
    float n = rn.reduceLanes(VectorOperators.ADD);
    for (int i = limit; i < dim; i++) {
      float rv = row[i];
      d = MathUtil.fma(query[i], rv, d);
      n = MathUtil.fma(rv, rv, n);
    }
    return cosineValue(d, qNorm2, n);
  }

  /**
   * Off-heap SIMD 4-row-unrolled fused cosine GEMV. Mirrors {@link #batchCosine(float[], float[][],
   * float[], int)}, reading the 4 rows from {@link MemorySegment}s (zero-copy mmap/off-heap slices)
   * via {@code FloatVector.fromMemorySegment} instead of {@code fromArray}. The query is still an
   * on-heap {@code float[]} whose norm is computed once and whose SIMD chunk is loaded once per
   * 4-row group.
   */
  @Override
  public void batchCosine(float[] query, MemorySegment[] rows, int dim, float[] out, int count) {
    float qNorm2 = dotProduct(query, 0, query, 0, dim); // ‖query‖² — computed once for the batch
    int rowGroup = count & ~3;
    int limit = FLOAT_SPECIES.loopBound(dim);

    for (int r = 0; r < rowGroup; r += 4) {
      MemorySegment r0 = rows[r], r1 = rows[r + 1], r2 = rows[r + 2], r3 = rows[r + 3];
      FloatVector d0 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector d1 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector d2 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector d3 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector n0 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector n1 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector n2 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector n3 = FloatVector.zero(FLOAT_SPECIES);

      for (int i = 0; i < limit; i += FLOAT_SPECIES.length()) {
        long byteOff = (long) i * Float.BYTES;
        FloatVector qv = FloatVector.fromArray(FLOAT_SPECIES, query, i); // query chunk loaded ONCE
        FloatVector v0 =
            FloatVector.fromMemorySegment(FLOAT_SPECIES, r0, byteOff, ByteOrder.LITTLE_ENDIAN);
        FloatVector v1 =
            FloatVector.fromMemorySegment(FLOAT_SPECIES, r1, byteOff, ByteOrder.LITTLE_ENDIAN);
        FloatVector v2 =
            FloatVector.fromMemorySegment(FLOAT_SPECIES, r2, byteOff, ByteOrder.LITTLE_ENDIAN);
        FloatVector v3 =
            FloatVector.fromMemorySegment(FLOAT_SPECIES, r3, byteOff, ByteOrder.LITTLE_ENDIAN);
        d0 = fma(qv, v0, d0);
        d1 = fma(qv, v1, d1);
        d2 = fma(qv, v2, d2);
        d3 = fma(qv, v3, d3);
        n0 = fma(v0, v0, n0);
        n1 = fma(v1, v1, n1);
        n2 = fma(v2, v2, n2);
        n3 = fma(v3, v3, n3);
      }

      float dot0 = d0.reduceLanes(VectorOperators.ADD);
      float dot1 = d1.reduceLanes(VectorOperators.ADD);
      float dot2 = d2.reduceLanes(VectorOperators.ADD);
      float dot3 = d3.reduceLanes(VectorOperators.ADD);
      float rn0 = n0.reduceLanes(VectorOperators.ADD);
      float rn1 = n1.reduceLanes(VectorOperators.ADD);
      float rn2 = n2.reduceLanes(VectorOperators.ADD);
      float rn3 = n3.reduceLanes(VectorOperators.ADD);

      for (int i = limit; i < dim; i++) {
        float q = query[i];
        float e0 = r0.getAtIndex(ValueLayout.JAVA_FLOAT, i);
        dot0 = MathUtil.fma(q, e0, dot0);
        rn0 = MathUtil.fma(e0, e0, rn0);
        float e1 = r1.getAtIndex(ValueLayout.JAVA_FLOAT, i);
        dot1 = MathUtil.fma(q, e1, dot1);
        rn1 = MathUtil.fma(e1, e1, rn1);
        float e2 = r2.getAtIndex(ValueLayout.JAVA_FLOAT, i);
        dot2 = MathUtil.fma(q, e2, dot2);
        rn2 = MathUtil.fma(e2, e2, rn2);
        float e3 = r3.getAtIndex(ValueLayout.JAVA_FLOAT, i);
        dot3 = MathUtil.fma(q, e3, dot3);
        rn3 = MathUtil.fma(e3, e3, rn3);
      }

      out[r] = cosineValue(dot0, qNorm2, rn0);
      out[r + 1] = cosineValue(dot1, qNorm2, rn1);
      out[r + 2] = cosineValue(dot2, qNorm2, rn2);
      out[r + 3] = cosineValue(dot3, qNorm2, rn3);
    }

    // Tail rows (0-3 remaining)
    for (int r = rowGroup; r < count; r++) {
      out[r] = cosineSegRow(query, rows[r], qNorm2, limit, dim);
    }
  }

  /** Single-row fused cosine over a {@code MemorySegment} row sharing the batch {@code qNorm2}. */
  private float cosineSegRow(float[] query, MemorySegment row, float qNorm2, int limit, int dim) {
    FloatVector dot = FloatVector.zero(FLOAT_SPECIES);
    FloatVector rn = FloatVector.zero(FLOAT_SPECIES);
    for (int i = 0; i < limit; i += FLOAT_SPECIES.length()) {
      FloatVector qv = FloatVector.fromArray(FLOAT_SPECIES, query, i);
      FloatVector rv =
          FloatVector.fromMemorySegment(
              FLOAT_SPECIES, row, (long) i * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
      dot = fma(qv, rv, dot);
      rn = fma(rv, rv, rn);
    }
    float d = dot.reduceLanes(VectorOperators.ADD);
    float n = rn.reduceLanes(VectorOperators.ADD);
    for (int i = limit; i < dim; i++) {
      float rv = row.getAtIndex(ValueLayout.JAVA_FLOAT, i);
      d = MathUtil.fma(query[i], rv, d);
      n = MathUtil.fma(rv, rv, n);
    }
    return cosineValue(d, qNorm2, n);
  }
}
