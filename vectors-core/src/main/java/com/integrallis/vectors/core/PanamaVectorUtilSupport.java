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
      float[] q8Scales) {
    GgufQuantizationSupport.quantizeQ8_0(query, cols, q8Quants, q8Scales);

    long rowBytes = (long) (cols / GGUF_Q_BLOCK_SIZE) * GGUF_Q4_0_BLOCK_BYTES;
    int blocks = cols / GGUF_Q_BLOCK_SIZE;
    GgufParallelSupport.forEachRow(
        qWeight,
        rows,
        cols,
        row -> {
          if (VECTOR_BITSIZE >= 256) {
            FloatVector accumulator = FloatVector.zero(FloatVector.SPECIES_256);
            long rowOffset = row * rowBytes;
            for (int block = 0; block < blocks; block++) {
              long blockOffset = rowOffset + (long) block * GGUF_Q4_0_BLOCK_BYTES;
              float scale =
                  Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset)) * q8Scales[block];
              IntVector integerLanes =
                  q4_0Q8_0IntegerLanes(
                      qWeight, blockOffset + Short.BYTES, q8Quants, block * GGUF_Q_BLOCK_SIZE);
              FloatVector products =
                  (FloatVector)
                      integerLanes.convertShape(VectorOperators.I2F, FloatVector.SPECIES_256, 0);
              accumulator =
                  fma(products, FloatVector.broadcast(FloatVector.SPECIES_256, scale), accumulator);
            }
            out[row] = reduceAdd(accumulator);
            return;
          }

          float sum = 0.0f;
          long rowOffset = row * rowBytes;
          for (int block = 0; block < blocks; block++) {
            long blockOffset = rowOffset + (long) block * GGUF_Q4_0_BLOCK_BYTES;
            float scale =
                Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset)) * q8Scales[block];
            int integerSum =
                q4_0Q8_0IntegerDot(
                    qWeight, blockOffset + Short.BYTES, q8Quants, block * GGUF_Q_BLOCK_SIZE);
            sum = MathUtil.fma(scale, integerSum, sum);
          }
          out[row] = sum;
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
      float[] laneScratch) {
    if (batchSize == 1) {
      ggufQ4_0Q8_0MatVecDot(queries, qWeight, rows, cols, out, q8Quants, q8Scales);
      return;
    }
    if (VECTOR_BITSIZE < 256) {
      VectorUtilSupport.super.ggufQ4_0Q8_0BatchedMatmul(
          queries, qWeight, batchSize, rows, cols, out, q8Quants, q8Scales, laneScratch);
      return;
    }

    int blocks = cols / GGUF_Q_BLOCK_SIZE;
    for (int batch = 0; batch < batchSize; batch++) {
      GgufQuantizationSupport.quantizeQ8_0(
          queries, batch * cols, cols, q8Quants, batch * cols, q8Scales, batch * blocks);
    }

    long rowBytes = (long) blocks * GGUF_Q4_0_BLOCK_BYTES;
    GgufParallelSupport.forEachRow(
        qWeight,
        rows,
        cols,
        row -> {
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
                    ByteVector.SPECIES_128,
                    qWeight,
                    blockOffset + Short.BYTES,
                    ByteOrder.LITTLE_ENDIAN);
            ShortVector low =
                (ShortVector)
                    packed
                        .and((byte) 0x0F)
                        .sub((byte) 8)
                        .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0);
            ShortVector high =
                (ShortVector)
                    packed
                        .lanewise(VectorOperators.LSHR, 4)
                        .and((byte) 0x0F)
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
                  weightScale * q8Scales[batch * blocks + block]);
            }
          }

          for (int batch = 0; batch < batchSize; batch++) {
            int laneOffset = rowLaneOffset + batch * FloatVector.SPECIES_256.length();
            out[batch * rows + row] =
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
      float scale) {
    ShortVector lowQuants =
        (ShortVector)
            ByteVector.fromArray(ByteVector.SPECIES_128, q8Quants, quantOffset)
                .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0);
    ShortVector highQuants =
        (ShortVector)
            ByteVector.fromArray(ByteVector.SPECIES_128, q8Quants, quantOffset + 16)
                .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0);
    IntVector integerLanes = fourProductLanes(low.mul(lowQuants), high.mul(highQuants));
    FloatVector products =
        (FloatVector) integerLanes.convertShape(VectorOperators.I2F, FloatVector.SPECIES_256, 0);
    FloatVector accumulator =
        FloatVector.fromArray(FloatVector.SPECIES_256, laneScratch, laneOffset);
    fma(products, FloatVector.broadcast(FloatVector.SPECIES_256, scale), accumulator)
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

  private static int q4_0Q8_0IntegerDot(
      MemorySegment qWeight, long nibbleOffset, byte[] q8Quants, int quantOffset) {
    if (VECTOR_BITSIZE >= 256) {
      ByteVector packed =
          ByteVector.fromMemorySegment(
              ByteVector.SPECIES_128, qWeight, nibbleOffset, ByteOrder.LITTLE_ENDIAN);
      ByteVector low = packed.and((byte) 0x0F).sub((byte) 8);
      ByteVector high = packed.lanewise(VectorOperators.LSHR, 4).and((byte) 0x0F).sub((byte) 8);
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
      ByteVector high = packed.lanewise(VectorOperators.LSHR, 4).and((byte) 0x0F).sub((byte) 8);
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
    ByteVector high = packed.lanewise(VectorOperators.LSHR, 4).and((byte) 0x0F).sub((byte) 8);
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
    int blocks = cols / GGUF_Q4_K_BLOCK_SIZE;
    GgufParallelSupport.forEachRow(
        qWeight,
        rows,
        cols,
        row -> {
          if (VECTOR_BITSIZE >= 256) {
            FloatVector accumulator = FloatVector.zero(FloatVector.SPECIES_256);
            float minimumContribution = 0.0f;
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
              IntVector blockLanes = IntVector.zero(IntVector.SPECIES_256);
              int minimumSum = 0;

              for (int group = 0; group < 8; group++) {
                int scale = GgufQuantizationSupport.qKScale(qWeight, scalesOffset, group);
                int min = GgufQuantizationSupport.qKMin(qWeight, scalesOffset, group);
                long packedOffset = quantsOffset + (long) (group >>> 1) * 32;
                int shift = (group & 1) * 4;
                int groupActivationOffset = activationOffset + group * 32;
                IntVector groupLanes =
                    q4_KQ8_KIntegerLanes(
                        qWeight, packedOffset, shift, q8Quants, groupActivationOffset);
                blockLanes = blockLanes.add(groupLanes.mul(scale));
                int sumOffset = groupActivationOffset / GGUF_Q8_K_SUM_BLOCK_SIZE;
                minimumSum += min * (q8Sums[sumOffset] + q8Sums[sumOffset + 1]);
              }

              FloatVector products =
                  (FloatVector)
                      blockLanes.convertShape(VectorOperators.I2F, FloatVector.SPECIES_256, 0);
              accumulator =
                  fma(products, FloatVector.broadcast(FloatVector.SPECIES_256, d), accumulator);
              minimumContribution = MathUtil.fma(-dMin, minimumSum, minimumContribution);
            }
            out[row] = accumulator.reduceLanes(VectorOperators.ADD) + minimumContribution;
            return;
          }

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

  private static int q4_KQ8_KIntegerDot(
      MemorySegment qWeight, long packedOffset, int shift, byte[] q8Quants, int quantOffset) {
    if (VECTOR_BITSIZE >= 256) {
      int sum = 0;
      for (int index = 0; index < 32; index += 16) {
        ByteVector packed =
            ByteVector.fromMemorySegment(
                ByteVector.SPECIES_128, qWeight, packedOffset + index, ByteOrder.LITTLE_ENDIAN);
        ByteVector q4 = packed.lanewise(VectorOperators.LSHR, shift).and((byte) 0x0F);
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
        ByteVector q4 = packed.lanewise(VectorOperators.LSHR, shift).and((byte) 0x0F);
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
      sum += ((packed >>> shift) & 0x0F) * q8Quants[quantOffset + index];
    }
    return sum;
  }

  static IntVector q4_KQ8_KIntegerLanes(
      MemorySegment qWeight, long packedOffset, int shift, byte[] q8Quants, int quantOffset) {
    ByteVector firstPacked =
        ByteVector.fromMemorySegment(
            ByteVector.SPECIES_128, qWeight, packedOffset, ByteOrder.LITTLE_ENDIAN);
    ByteVector secondPacked =
        ByteVector.fromMemorySegment(
            ByteVector.SPECIES_128, qWeight, packedOffset + 16, ByteOrder.LITTLE_ENDIAN);
    ShortVector firstProducts =
        ((ShortVector)
                firstPacked
                    .lanewise(VectorOperators.LSHR, shift)
                    .and((byte) 0x0F)
                    .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0))
            .mul(
                (ShortVector)
                    ByteVector.fromArray(ByteVector.SPECIES_128, q8Quants, quantOffset)
                        .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0));
    ShortVector secondProducts =
        ((ShortVector)
                secondPacked
                    .lanewise(VectorOperators.LSHR, shift)
                    .and((byte) 0x0F)
                    .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0))
            .mul(
                (ShortVector)
                    ByteVector.fromArray(ByteVector.SPECIES_128, q8Quants, quantOffset + 16)
                        .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0));
    return fourProductLanes(firstProducts, secondProducts);
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

  static IntVector q5_0Q8_0IntegerLanes(
      MemorySegment qWeight, long packedOffset, int highBits, byte[] q8Quants, int quantOffset) {
    IntVector accumulator = IntVector.zero(IntVector.SPECIES_256);
    for (int index = 0; index < 16; index += 8) {
      ByteVector packed =
          ByteVector.fromMemorySegment(
              ByteVector.SPECIES_64, qWeight, packedOffset + index, ByteOrder.LITTLE_ENDIAN);
      ByteVector low = q5Values(packed.and((byte) 0x0F), (byte) (highBits >>> index));
      ByteVector high =
          q5Values(
              packed.lanewise(VectorOperators.LSHR, 4).and((byte) 0x0F),
              (byte) (highBits >>> (index + 16)));
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

  private static ByteVector q5Values(ByteVector lowBits, byte highBits) {
    VectorMask<Byte> highMask =
        Q5_HIGH_BIT_MASKS.and(highBits).compare(VectorOperators.NE, (byte) 0);
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
    return s1 * q6_KQ8_KGroupDot(qWeight, ql1Offset, qhOffset, 0, 0, q8Quants, quantOffset)
        + s2 * q6_KQ8_KGroupDot(qWeight, ql2Offset, qhOffset, 0, 2, q8Quants, quantOffset + 32)
        + s3 * q6_KQ8_KGroupDot(qWeight, ql1Offset, qhOffset, 4, 4, q8Quants, quantOffset + 64)
        + s4 * q6_KQ8_KGroupDot(qWeight, ql2Offset, qhOffset, 4, 6, q8Quants, quantOffset + 96);
  }

  private static int q6_KQ8_KGroupDot(
      MemorySegment qWeight,
      long qlOffset,
      long qhOffset,
      int qlShift,
      int qhShift,
      byte[] q8Quants,
      int quantOffset) {
    int sum = 0;
    for (int index = 0; index < 16; index += 8) {
      ByteVector ql =
          ByteVector.fromMemorySegment(
              ByteVector.SPECIES_64, qWeight, qlOffset + index, ByteOrder.LITTLE_ENDIAN);
      ByteVector qh =
          ByteVector.fromMemorySegment(
              ByteVector.SPECIES_64, qWeight, qhOffset + index, ByteOrder.LITTLE_ENDIAN);
      ByteVector low = ql.lanewise(VectorOperators.LSHR, qlShift).and((byte) 0x0F);
      ByteVector high =
          qh.lanewise(VectorOperators.LSHR, qhShift)
              .and((byte) 0x03)
              .lanewise(VectorOperators.LSHL, 4);
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
        row -> {
          float sum = 0.0f;
          long rowOffset = row * rowBytes;
          for (int block = 0; block < blocks; block++) {
            long blockOffset = rowOffset + (long) block * GGUF_Q8_0_BLOCK_BYTES;
            float scale =
                Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset)) * q8Scales[block];
            int integerSum =
                q8_0Q8_0IntegerDot(
                    qWeight, blockOffset + Short.BYTES, q8Quants, block * GGUF_Q_BLOCK_SIZE);
            sum = MathUtil.fma(scale, integerSum, sum);
          }
          out[row] = sum;
        });
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
