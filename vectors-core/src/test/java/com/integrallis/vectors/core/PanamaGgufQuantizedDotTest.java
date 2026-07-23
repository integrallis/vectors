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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.data.Offset.offset;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Random;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class PanamaGgufQuantizedDotTest {

  private static final ValueLayout.OfShort LE_SHORT =
      ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

  @Test
  void q4ShortPairwiseRequiresExplicitSelectionAvx2WidthAndModelSizedRows() {
    assertThat(PanamaVectorUtilSupport.useQ4ShortPairwise(GgufQ4Kernel.WIDENED, 256, 64)).isFalse();
    assertThat(PanamaVectorUtilSupport.useQ4ShortPairwise(GgufQ4Kernel.SHORT_PAIRWISE, 128, 64))
        .isFalse();
    assertThat(PanamaVectorUtilSupport.useQ4ShortPairwise(GgufQ4Kernel.SHORT_PAIRWISE, 256, 31))
        .isFalse();
    assertThat(PanamaVectorUtilSupport.useQ4ShortPairwise(GgufQ4Kernel.SHORT_PAIRWISE, 256, 32))
        .isTrue();
  }

  @Test
  void explicitQ4KernelSelectionCannotBypassHardwareEligibility() {
    assertThat(PanamaVectorUtilSupport.useQ4ShortPairwise(GgufQ4Kernel.WIDENED, 256, 64)).isFalse();
    assertThat(PanamaVectorUtilSupport.useQ4ShortPairwise(GgufQ4Kernel.SHORT_PAIRWISE, 256, 64))
        .isTrue();
    assertThat(PanamaVectorUtilSupport.useQ4ShortPairwise(GgufQ4Kernel.SHORT_PAIRWISE, 128, 64))
        .isFalse();
    assertThat(PanamaVectorUtilSupport.useQ4ShortPairwise(GgufQ4Kernel.SHORT_PAIRWISE, 256, 31))
        .isFalse();
    assertThat(
            PanamaVectorUtilSupport.useQ4UnsignedPairwise(GgufQ4Kernel.UNSIGNED_PAIRWISE, 128, 64))
        .isFalse();
    assertThat(
            PanamaVectorUtilSupport.useQ4UnsignedPairwise(GgufQ4Kernel.UNSIGNED_PAIRWISE, 256, 31))
        .isFalse();
    assertThat(
            PanamaVectorUtilSupport.useQ4UnsignedPairwise(GgufQ4Kernel.UNSIGNED_PAIRWISE, 256, 32))
        .isTrue();
    assertThat(PanamaVectorUtilSupport.useQ4UnsignedPairwise(GgufQ4Kernel.SHORT_PAIRWISE, 256, 64))
        .isFalse();
  }

  @Test
  void unsignedPairwiseFallbackDoesNotComputeUnusedCorrections() {
    int blocks = 31;
    int cols = blocks * 32;
    float[] query = new float[cols];
    byte[] weights = new byte[blocks * 18];
    ByteBuffer weightBuffer = ByteBuffer.wrap(weights).order(ByteOrder.LITTLE_ENDIAN);
    for (int block = 0; block < blocks; block++) {
      weightBuffer.putShort(block * 18, Float.floatToFloat16(0.125f));
      for (int index = 0; index < 32; index++) {
        query[block * 32 + index] = (index - 15) / 17.0f;
      }
    }
    int sentinel = 0x5148;
    int[] corrections = new int[cols / 4];
    java.util.Arrays.fill(corrections, sentinel);

    new PanamaVectorUtilSupport()
        .ggufQ4_0Q8_0MatVecDot(
            query,
            MemorySegment.ofArray(weights),
            1,
            cols,
            new float[1],
            new byte[cols],
            new float[blocks],
            corrections,
            GgufQ4Kernel.UNSIGNED_PAIRWISE);

    assertThat(corrections).containsOnly(sentinel);
  }

  @Test
  void q4HighNibblesDecodeEveryUnsignedByteWithoutCrossLaneBits() {
    assertQ4HighNibbles(ByteVector.SPECIES_64);
    assertQ4HighNibbles(ByteVector.SPECIES_128);
  }

  private static void assertQ4HighNibbles(VectorSpecies<Byte> species) {
    byte[] packed = new byte[species.length()];
    byte[] expected = new byte[packed.length];

    for (int first = 0; first < 256; first += packed.length) {
      for (int lane = 0; lane < packed.length; lane++) {
        int unsigned = first + lane;
        packed[lane] = (byte) unsigned;
        expected[lane] = (byte) (unsigned >>> 4);
      }

      ByteVector actual =
          PanamaVectorUtilSupport.q4HighNibbles(ByteVector.fromArray(species, packed, 0));

      assertThat(actual.toArray()).containsExactly(expected);
    }
  }

  @Test
  void q4_0Q8_0IntegerLanesSumFourProductsPerLane() {
    byte[] packed = new byte[16];
    byte[] q8 = new byte[32];
    int[] q4 = new int[32];
    for (int index = 0; index < 16; index++) {
      int low = index % 16;
      int high = 15 - index;
      packed[index] = (byte) (low | (high << 4));
      q4[index] = low - 8;
      q4[index + 16] = high - 8;
    }
    for (int index = 0; index < q8.length; index++) {
      q8[index] = (byte) (index - 16);
    }

    IntVector actual =
        PanamaVectorUtilSupport.q4_0Q8_0IntegerLanes(MemorySegment.ofArray(packed), 0, q8, 0);

    int[] expected = new int[8];
    for (int lane = 0; lane < expected.length; lane++) {
      for (int index = lane * 4; index < lane * 4 + 4; index++) {
        expected[lane] += q4[index] * q8[index];
      }
    }
    assertThat(actual.toArray()).containsExactly(expected);
  }

  @Test
  void q4_0Q8_0ShortPairwiseIntegerLanesMatchWidenedKernelExactly() {
    byte[] packed = new byte[16];
    byte[] q8 = new byte[32];
    Random random = new Random(0x514750414952L);

    for (int iteration = 0; iteration < 10_000; iteration++) {
      random.nextBytes(packed);
      random.nextBytes(q8);
      if (iteration == 0) {
        for (int index = 0; index < packed.length; index++) {
          packed[index] = (byte) (index | ((15 - index) << 4));
        }
        for (int index = 0; index < q8.length; index++) {
          q8[index] =
              switch (index & 3) {
                case 0 -> Byte.MIN_VALUE;
                case 1 -> -1;
                case 2 -> 0;
                default -> 127;
              };
        }
      }

      MemorySegment weights = MemorySegment.ofArray(packed);
      IntVector expected = PanamaVectorUtilSupport.q4_0Q8_0IntegerLanes(weights, 0, q8, 0);
      IntVector shortPairwise =
          PanamaVectorUtilSupport.q4_0Q8_0ShortPairwiseIntegerLanes(weights, 0, q8, 0);
      assertThat(shortPairwise.toArray()).containsExactly(expected.toArray());
    }
  }

  @Test
  void q4_0Q8_0UnsignedPairwiseRowPreservesSplitHalfAccumulation() {
    int blocks = 64;
    byte[] weightBytes = new byte[blocks * 18];
    byte[] q8 = new byte[blocks * 32];
    float[] q8Scales = new float[blocks];
    int[] zeroPointCorrections = new int[blocks * 8];
    Random random = new Random(0x554e5349474e4544L);

    for (int trial = 0; trial < 1_000; trial++) {
      for (int block = 0; block < blocks; block++) {
        int weightOffset = block * 18;
        short scale = Float.floatToFloat16(0.001f + random.nextFloat() * 0.05f);
        weightBytes[weightOffset] = (byte) scale;
        weightBytes[weightOffset + 1] = (byte) (scale >>> 8);
        for (int index = 0; index < 16; index++) {
          weightBytes[weightOffset + Short.BYTES + index] = (byte) random.nextInt();
        }
        for (int index = 0; index < 32; index++) {
          q8[block * 32 + index] = (byte) random.nextInt(-127, 128);
        }
        for (int group = 0; group < 8; group++) {
          int offset = block * 32 + group * 4;
          zeroPointCorrections[block * 8 + group] =
              8 * (q8[offset] + q8[offset + 1] + q8[offset + 2] + q8[offset + 3]);
        }
        q8Scales[block] = 0.001f + random.nextFloat() * 0.05f;
      }

      MemorySegment weights = MemorySegment.ofArray(weightBytes);
      FloatVector expectedLanes = FloatVector.zero(FloatVector.SPECIES_256);
      for (int block = 0; block < blocks; block++) {
        long blockOffset = (long) block * 18;
        float scale = Float.float16ToFloat(weights.get(LE_SHORT, blockOffset)) * q8Scales[block];
        IntVector groups =
            PanamaVectorUtilSupport.q4_0Q8_0ShortPairwiseIntegerLanes(
                weights, blockOffset + Short.BYTES, q8, block * 32);
        expectedLanes =
            PanamaVectorUtilSupport.fma(
                (FloatVector) groups.convertShape(VectorOperators.I2F, FloatVector.SPECIES_256, 0),
                FloatVector.broadcast(FloatVector.SPECIES_256, scale),
                expectedLanes);
      }
      float even =
          (expectedLanes.lane(4) + expectedLanes.lane(0))
              + (expectedLanes.lane(6) + expectedLanes.lane(2));
      float odd =
          (expectedLanes.lane(5) + expectedLanes.lane(1))
              + (expectedLanes.lane(7) + expectedLanes.lane(3));
      float expected = even + odd;

      float actual =
          PanamaVectorUtilSupport.q4_0Q8_0UnsignedPairwiseRowDot(
              weights, 0, blocks, q8, q8Scales, zeroPointCorrections);

      assertThat(Float.floatToRawIntBits(actual)).isEqualTo(Float.floatToRawIntBits(expected));
    }
  }

  @Test
  void q4_0Q8_0UnsignedPairwiseBatchedMatmulCombinesIntegerHalvesBeforeScaling() {
    int blocks = 65;
    int cols = blocks * 32;
    int batchSize = 2;
    byte[] weightBytes = new byte[blocks * 18];
    float[] queries = new float[batchSize * cols];
    Random random = new Random(0x424154434845444cL);
    for (int block = 0; block < blocks; block++) {
      int weightOffset = block * 18;
      short scale = Float.floatToFloat16(0.001f + random.nextFloat() * 0.05f);
      weightBytes[weightOffset] = (byte) scale;
      weightBytes[weightOffset + 1] = (byte) (scale >>> 8);
      for (int index = 0; index < 16; index++) {
        weightBytes[weightOffset + Short.BYTES + index] = (byte) random.nextInt();
      }
    }
    for (int index = 0; index < queries.length; index++) {
      queries[index] = random.nextFloat() * 4.0f - 2.0f;
    }

    MemorySegment weights = MemorySegment.ofArray(weightBytes);
    float[] actual = new float[batchSize];
    byte[] q8 = new byte[batchSize * cols];
    float[] q8Scales = new float[batchSize * blocks];
    int[] zeroPointCorrections = new int[batchSize * blocks * 8];
    VectorUtil.ggufQ4_0Q8_0BatchedMatmul(
        queries,
        weights,
        batchSize,
        1,
        cols,
        actual,
        q8,
        q8Scales,
        zeroPointCorrections,
        new float[batchSize * 8],
        GgufQ4Kernel.UNSIGNED_PAIRWISE);

    for (int batch = 0; batch < batchSize; batch++) {
      FloatVector expectedLanes = FloatVector.zero(FloatVector.SPECIES_128);
      for (int block = 0; block < blocks; block++) {
        long blockOffset = (long) block * 18;
        float scale =
            Float.float16ToFloat(weights.get(LE_SHORT, blockOffset))
                * q8Scales[batch * blocks + block];
        int[] groups =
            PanamaVectorUtilSupport.q4_0Q8_0ShortPairwiseIntegerLanes(
                    weights, blockOffset + Short.BYTES, q8, batch * cols + block * 32)
                .toArray();
        IntVector combinedGroups =
            IntVector.fromArray(
                IntVector.SPECIES_128,
                new int[] {
                  groups[0] + groups[4],
                  groups[1] + groups[5],
                  groups[2] + groups[6],
                  groups[3] + groups[7]
                },
                0);
        expectedLanes =
            PanamaVectorUtilSupport.fma(
                (FloatVector)
                    combinedGroups.convertShape(VectorOperators.I2F, FloatVector.SPECIES_128, 0),
                FloatVector.broadcast(FloatVector.SPECIES_128, scale),
                expectedLanes);
      }
      float expected =
          (expectedLanes.lane(2) + expectedLanes.lane(0))
              + (expectedLanes.lane(3) + expectedLanes.lane(1));
      assertThat(Float.floatToRawIntBits(actual[batch]))
          .isEqualTo(Float.floatToRawIntBits(expected));
    }
  }

  @Test
  void q8_0QuantizationProducesExactQ4ZeroPointCorrections() {
    float[] query = new float[96];
    Random random = new Random(0x51385a504f494e54L);
    for (int index = 0; index < query.length; index++) {
      query[index] = random.nextFloat() * 4.0f - 2.0f;
    }
    byte[] quants = new byte[query.length];
    float[] scales = new float[query.length / 32];
    int[] zeroPointCorrections = new int[query.length / 4];
    byte[] referenceQuants = new byte[query.length];
    float[] referenceScales = new float[query.length / 32];

    GgufQuantizationSupport.quantizeQ8_0(query, query.length, referenceQuants, referenceScales);

    GgufQuantizationSupport.quantizeQ8_0WithQ4Corrections(
        query, query.length, quants, scales, zeroPointCorrections);

    assertThat(quants).containsExactly(referenceQuants);
    assertThat(scales).containsExactly(referenceScales);
    for (int group = 0; group < zeroPointCorrections.length; group++) {
      int offset = group * 4;
      assertThat(zeroPointCorrections[group])
          .isEqualTo(
              8 * (quants[offset] + quants[offset + 1] + quants[offset + 2] + quants[offset + 3]));
    }
  }

  @Test
  void q8_0BatchQuantizationCombinesQ4HalfCorrections() {
    int batchSize = 2;
    int dimensions = 64;
    int blocks = dimensions / 32;
    float[] values = new float[batchSize * dimensions];
    Random random = new Random(0x434f4d42494e4544L);
    for (int index = 0; index < values.length; index++) {
      values[index] = random.nextFloat() * 4.0f - 2.0f;
    }

    GgufQ8_0Batch activation = GgufQ8_0Batch.allocate(batchSize, dimensions);
    activation.quantizeForQ4(values, batchSize, GgufQ4Kernel.UNSIGNED_PAIRWISE);

    byte[] quants = activation.quants();
    int[] corrections = activation.zeroPointCorrections();
    assertThat(corrections).hasSize(batchSize * blocks * 4);
    for (int batch = 0; batch < batchSize; batch++) {
      for (int block = 0; block < blocks; block++) {
        int quantOffset = batch * dimensions + block * 32;
        int correctionOffset = (batch * blocks + block) * 4;
        for (int group = 0; group < 4; group++) {
          int groupSum = 0;
          for (int lane = 0; lane < 4; lane++) {
            groupSum += quants[quantOffset + group * 4 + lane];
            groupSum += quants[quantOffset + 16 + group * 4 + lane];
          }
          assertThat(corrections[correctionOffset + group]).isEqualTo(8 * groupSum);
        }
      }
    }
  }

  @Test
  void q4_KQ8_KIntegerDotDecodesBothNibbleHalves() {
    byte[] packed = new byte[32];
    byte[] q8 = new byte[32];
    int expectedLow = 0;
    int expectedHigh = 0;
    for (int index = 0; index < packed.length; index++) {
      int high = (index * 3) % 16;
      int low = 15 - high;
      packed[index] = (byte) ((high << 4) | low);
      q8[index] = (byte) (index - 16);
      expectedLow += low * q8[index];
      expectedHigh += high * q8[index];
    }

    MemorySegment weights = MemorySegment.ofArray(packed);
    assertThat(PanamaVectorUtilSupport.q4_KQ8_KIntegerDot(weights, 0, 0, q8, 0))
        .isEqualTo(expectedLow);
    assertThat(PanamaVectorUtilSupport.q4_KQ8_KIntegerDot(weights, 0, 4, q8, 0))
        .isEqualTo(expectedHigh);
  }

  @Test
  void kQuantPackedDotHelpersReturnPrimitivesToPreserveScalarReplacement() {
    assertThat(PanamaVectorUtilSupport.class.getDeclaredMethods())
        .filteredOn(
            method ->
                method.getName().equals("q4_KQ8_KIntegerDot")
                    || method.getName().equals("q4_KQ8_KLowIntegerDot")
                    || method.getName().equals("q4_KQ8_KHighIntegerDot")
                    || method.getName().matches("q6_KQ8_KGroup[0246]Dot"))
        .isNotEmpty()
        .allSatisfy(method -> assertThat(method.getReturnType()).isEqualTo(int.class));
  }

  @Test
  void q5_KQ8_KIntegerDotDecodesPerElementHighBits() {
    byte[] weights = new byte[64];
    byte[] q8 = new byte[32];
    int highBit = 1 << 3;
    for (int index = 0; index < 32; index++) {
      int low = index * 3 % 16;
      int high = index * 5 % 16;
      weights[32 + index] = (byte) (low | (high << 4));
      if ((index & 1) != 0) {
        weights[index] = (byte) highBit;
      }
      q8[index] = (byte) (index - 16);
    }

    int actual =
        PanamaVectorUtilSupport.q5_KQ8_KIntegerDot(
            MemorySegment.ofArray(weights), 32, 4, 0, highBit, q8, 0);

    int expected = 0;
    for (int index = 0; index < 32; index++) {
      int packed = weights[32 + index] & 0xFF;
      int quant = (packed >>> 4) & 0x0F;
      if ((weights[index] & highBit) != 0) {
        quant += 16;
      }
      expected += quant * q8[index];
    }
    assertThat(actual).isEqualTo(expected);
  }

  @Test
  void q6_KQ8_KIntegerDotDecodesPackedWeightsAndSignedScales() {
    byte[] weights = new byte[80];
    byte[] q8 = new byte[128];
    Random random = new Random(0x516B48L);
    random.nextBytes(weights);
    random.nextBytes(q8);
    int quantOffset = 7;
    int s1 = -11;
    int s2 = 7;
    int s3 = -3;
    int s4 = 13;

    int actual =
        PanamaVectorUtilSupport.q6_KQ8_KIntegerDot(
            MemorySegment.ofArray(weights), 0, 32, 64, q8, quantOffset, s1, s2, s3, s4);

    int expected = 0;
    for (int index = 0; index < 16; index++) {
      int ql1 = weights[index] & 0xFF;
      int ql2 = weights[32 + index] & 0xFF;
      int high = weights[64 + index] & 0xFF;
      int q1 = ((ql1 & 0x0F) | ((high & 0x03) << 4)) - 32;
      int q2 = ((ql2 & 0x0F) | (((high >>> 2) & 0x03) << 4)) - 32;
      int q3 = ((ql1 >>> 4) | (((high >>> 4) & 0x03) << 4)) - 32;
      int q4 = ((ql2 >>> 4) | (((high >>> 6) & 0x03) << 4)) - 32;
      expected += s1 * q1 * q8[quantOffset + index];
      expected += s2 * q2 * q8[quantOffset + index + 32];
      expected += s3 * q3 * q8[quantOffset + index + 64];
      expected += s4 * q4 * q8[quantOffset + index + 96];
    }
    assertThat(actual).isEqualTo(expected);
  }

  @Test
  void q5_0Q8_0IntegerLanesDecodeHighBits() {
    byte[] packed = new byte[16];
    byte[] q8 = new byte[32];
    Random random = new Random(0x515048L);
    random.nextBytes(packed);
    random.nextBytes(q8);
    int highBits = 0xA5C33C5A;

    IntVector actual =
        PanamaVectorUtilSupport.q5_0Q8_0IntegerLanes(
            MemorySegment.ofArray(packed), 0, highBits, q8, 0);

    int[] expected = new int[8];
    for (int index = 0; index < packed.length; index++) {
      int value = packed[index] & 0xFF;
      int low = (value & 0x0F) | (((highBits >>> index) & 1) << 4);
      int high = (value >>> 4) | (((highBits >>> (index + 16)) & 1) << 4);
      int lane = index & 7;
      expected[lane] += (low - 16) * q8[index];
      expected[lane] += (high - 16) * q8[index + 16];
    }
    assertThat(actual.toArray()).containsExactly(expected);
  }

  @Test
  void panamaProviderOwnsQ4_0Q8_0Kernel() {
    assertThat(PanamaVectorUtilSupport.class.getDeclaredMethods())
        .extracting(Method::getName)
        .contains("ggufQ4_0Q8_0MatVecDot");
  }

  @Test
  void panamaProviderOwnsQ8_0Q8_0Kernel() {
    assertThat(PanamaVectorUtilSupport.class.getDeclaredMethods())
        .extracting(Method::getName)
        .contains("ggufQ8_0Q8_0MatVecDot");
  }

  @Test
  void panamaProviderOwnsQ8_0Q8_0BatchedKernel() {
    assertThat(PanamaVectorUtilSupport.class.getDeclaredMethods())
        .extracting(Method::getName)
        .contains("ggufQ8_0Q8_0BatchedMatmul");
  }

  @Test
  void panamaProviderOwnsQ4_KQ8_KKernel() {
    assertThat(PanamaVectorUtilSupport.class.getDeclaredMethods())
        .extracting(Method::getName)
        .contains("ggufQ4_KQ8_KMatVecDot");
  }

  @Test
  void panamaProviderOwnsQ4_KQ8_KBatchedKernel() {
    assertThat(PanamaVectorUtilSupport.class.getDeclaredMethods())
        .extracting(Method::getName)
        .contains("ggufQ4_KQ8_KBatchedMatmul");
  }

  @Test
  void panamaProviderOwnsQ5_KQ8_KKernel() {
    assertThat(PanamaVectorUtilSupport.class.getDeclaredMethods())
        .extracting(Method::getName)
        .contains("ggufQ5_KQ8_KMatVecDot");
  }

  @Test
  void panamaProviderOwnsQ5_KQ8_KBatchedKernel() {
    assertThat(PanamaVectorUtilSupport.class.getDeclaredMethods())
        .extracting(Method::getName)
        .contains("ggufQ5_KQ8_KBatchedMatmul");
  }

  @Test
  void panamaProviderOwnsQ6_KQ8_KKernel() {
    assertThat(PanamaVectorUtilSupport.class.getDeclaredMethods())
        .extracting(Method::getName)
        .contains("ggufQ6_KQ8_KMatVecDot");
  }

  @Test
  void panamaProviderOwnsQ6_KQ8_KBatchedKernel() {
    assertThat(PanamaVectorUtilSupport.class.getDeclaredMethods())
        .extracting(Method::getName)
        .contains("ggufQ6_KQ8_KBatchedMatmul");
  }

  @Test
  void panamaProviderOwnsQ5_0Q8_0Kernel() {
    assertThat(PanamaVectorUtilSupport.class.getDeclaredMethods())
        .extracting(Method::getName)
        .contains("ggufQ5_0Q8_0MatVecDot", "ggufQ5_0Q8_0BatchedMatmul");
  }

  @Test
  void q4_0Q8_0KernelMatchesScalarReferenceAcrossRowsAndBlocks() {
    int rows = 512;
    int cols = 2048;
    Random random = new Random(0x5148L);
    float[] query = new float[cols];
    for (int index = 0; index < cols; index++) {
      query[index] = random.nextFloat() * 4.0f - 2.0f;
    }

    byte[] weights = new byte[rows * (cols / 32) * 18];
    random.nextBytes(weights);
    ByteBuffer buffer = ByteBuffer.wrap(weights).order(ByteOrder.LITTLE_ENDIAN);
    for (int offset = 0; offset < weights.length; offset += 18) {
      float scale = random.nextFloat() * 0.05f + 0.001f;
      buffer.putShort(offset, Float.floatToFloat16(scale));
    }

    float[] expected = new float[rows];
    float[] actual = new float[rows];
    byte[] expectedQuants = new byte[cols];
    byte[] actualQuants = new byte[cols];
    float[] expectedScales = new float[cols / 32];
    float[] actualScales = new float[cols / 32];
    MemorySegment weightSegment = MemorySegment.ofArray(weights);

    new ScalarVectorUtilSupport()
        .ggufQ4_0Q8_0MatVecDot(
            query,
            weightSegment,
            rows,
            cols,
            expected,
            expectedQuants,
            expectedScales,
            new int[cols / 4],
            GgufQ4Kernel.WIDENED);
    new PanamaVectorUtilSupport()
        .ggufQ4_0Q8_0MatVecDot(
            query,
            weightSegment,
            rows,
            cols,
            actual,
            actualQuants,
            actualScales,
            new int[cols / 4],
            GgufQ4Kernel.WIDENED);

    assertThat(actualQuants).containsExactly(expectedQuants);
    assertThat(actualScales).containsExactly(expectedScales);
    assertThat(actual).hasSameSizeAs(expected);
    for (int row = 0; row < rows; row++) {
      assertThat(actual[row]).isCloseTo(expected[row], offset(1e-4f));
    }
  }

  @Test
  void q4_0UnsignedBlockPairsStayNumericallyCloseToGemvForOddAndEvenBlockCounts() {
    int batchSize = 3;
    int rows = 5;
    Random random = new Random(0x5144_424c_4f43_4b53L);
    PanamaVectorUtilSupport support = new PanamaVectorUtilSupport();

    for (int blocks : new int[] {32, 33, 64}) {
      int cols = blocks * 32;
      float[] queries = new float[batchSize * cols];
      for (int index = 0; index < queries.length; index++) {
        queries[index] = random.nextFloat() * 4.0f - 2.0f;
      }
      byte[] weights = new byte[rows * blocks * 18];
      random.nextBytes(weights);
      ByteBuffer buffer = ByteBuffer.wrap(weights).order(ByteOrder.LITTLE_ENDIAN);
      for (int block = 0; block < rows * blocks; block++) {
        buffer.putShort(block * 18, Float.floatToFloat16(random.nextFloat() * 0.1f - 0.05f));
      }

      GgufQ8_0Batch activation = GgufQ8_0Batch.allocate(batchSize, cols);
      activation.quantizeForQ4(queries, batchSize, GgufQ4Kernel.UNSIGNED_PAIRWISE);
      MemorySegment weightSegment = MemorySegment.ofArray(weights);
      float[] expected = new float[batchSize * rows];
      float[] actual = new float[batchSize * rows];
      float[] query = new float[cols];
      float[] gemv = new float[rows];
      for (int batch = 0; batch < batchSize; batch++) {
        System.arraycopy(queries, batch * cols, query, 0, cols);
        support.ggufQ4_0Q8_0MatVecDot(
            query,
            weightSegment,
            rows,
            cols,
            gemv,
            new byte[cols],
            new float[blocks],
            new int[blocks * 8],
            GgufQ4Kernel.UNSIGNED_PAIRWISE);
        System.arraycopy(gemv, 0, expected, batch * rows, rows);
      }

      support.ggufQ4_0Q8_0BatchedMatmulRows(
          weightSegment,
          batchSize,
          rows,
          cols,
          0,
          rows,
          actual,
          activation,
          new float[batchSize * rows * 8],
          GgufQ4Kernel.UNSIGNED_PAIRWISE);

      for (int index = 0; index < actual.length; index++) {
        assertThat(actual[index]).isCloseTo(expected[index], offset(1e-5f));
      }
    }
  }

  @Test
  void q8_0Q8_0KernelMatchesScalarReferenceAcrossRowsAndBlocks() {
    int rows = 512;
    int cols = 2048;
    Random random = new Random(0x5188L);
    float[] query = new float[cols];
    for (int index = 0; index < cols; index++) {
      query[index] = random.nextFloat() * 4.0f - 2.0f;
    }

    byte[] weights = new byte[rows * (cols / 32) * 34];
    random.nextBytes(weights);
    ByteBuffer buffer = ByteBuffer.wrap(weights).order(ByteOrder.LITTLE_ENDIAN);
    for (int offset = 0; offset < weights.length; offset += 34) {
      float scale = random.nextFloat() * 0.05f + 0.001f;
      buffer.putShort(offset, Float.floatToFloat16(scale));
    }

    float[] expected = new float[rows];
    float[] actual = new float[rows];
    byte[] expectedQuants = new byte[cols];
    byte[] actualQuants = new byte[cols];
    float[] expectedScales = new float[cols / 32];
    float[] actualScales = new float[cols / 32];
    MemorySegment weightSegment = MemorySegment.ofArray(weights);

    new ScalarVectorUtilSupport()
        .ggufQ8_0Q8_0MatVecDot(
            query, weightSegment, rows, cols, expected, expectedQuants, expectedScales);
    new PanamaVectorUtilSupport()
        .ggufQ8_0Q8_0MatVecDot(
            query, weightSegment, rows, cols, actual, actualQuants, actualScales);

    assertThat(actualQuants).containsExactly(expectedQuants);
    assertThat(actualScales).containsExactly(expectedScales);
    assertThat(actual).containsExactly(expected);
  }

  @Test
  void q8_0Q8_0PrequantizedRowRangesMatchScalarReferenceExactly() {
    int batchSize = 3;
    int rows = 17;
    int cols = 96;
    Random random = new Random(0x5188_524f_5753L);
    float[] queries = new float[batchSize * cols];
    for (int index = 0; index < queries.length; index++) {
      queries[index] = random.nextFloat() * 4.0f - 2.0f;
    }

    byte[] weights = new byte[rows * (cols / 32) * 34];
    random.nextBytes(weights);
    ByteBuffer buffer = ByteBuffer.wrap(weights).order(ByteOrder.LITTLE_ENDIAN);
    for (int offset = 0; offset < weights.length; offset += 34) {
      buffer.putShort(offset, Float.floatToFloat16(random.nextFloat() * 0.05f + 0.001f));
    }

    GgufQ8_0Batch activation = GgufQ8_0Batch.allocate(batchSize, cols);
    activation.quantize(queries, batchSize);
    MemorySegment weightSegment = MemorySegment.ofArray(weights);
    float[] expected = new float[batchSize * rows];
    float[] actual = new float[batchSize * rows];

    new ScalarVectorUtilSupport()
        .ggufQ8_0Q8_0BatchedMatmulRows(
            weightSegment, batchSize, rows, cols, 0, rows, expected, activation);
    PanamaVectorUtilSupport support = new PanamaVectorUtilSupport();
    support.ggufQ8_0Q8_0BatchedMatmulRows(
        weightSegment, batchSize, rows, cols, 0, 7, actual, activation);
    support.ggufQ8_0Q8_0BatchedMatmulRows(
        weightSegment, batchSize, rows, cols, 7, rows, actual, activation);

    assertThat(actual).containsExactly(expected);
  }

  @Test
  void q4_KQ8_KKernelMatchesScalarReferenceAcrossRowsAndBlocks() {
    int rows = 512;
    int cols = 2048;
    Random random = new Random(0x514B48L);
    float[] query = new float[cols];
    for (int index = 0; index < cols; index++) {
      query[index] = random.nextFloat() * 4.0f - 2.0f;
    }

    byte[] weights = new byte[rows * (cols / 256) * 144];
    random.nextBytes(weights);
    ByteBuffer buffer = ByteBuffer.wrap(weights).order(ByteOrder.LITTLE_ENDIAN);
    for (int offset = 0; offset < weights.length; offset += 144) {
      float scale = random.nextFloat() * 0.05f + 0.001f;
      float minScale = random.nextFloat() * 0.05f;
      buffer.putShort(offset, Float.floatToFloat16(scale));
      buffer.putShort(offset + Short.BYTES, Float.floatToFloat16(minScale));
    }

    float[] expected = new float[rows];
    float[] actual = new float[rows];
    byte[] expectedQuants = new byte[cols];
    byte[] actualQuants = new byte[cols];
    float[] expectedScales = new float[cols / 256];
    float[] actualScales = new float[cols / 256];
    short[] expectedSums = new short[cols / 16];
    short[] actualSums = new short[cols / 16];
    MemorySegment weightSegment = MemorySegment.ofArray(weights);

    new ScalarVectorUtilSupport()
        .ggufQ4_KQ8_KMatVecDot(
            query,
            weightSegment,
            rows,
            cols,
            expected,
            expectedQuants,
            expectedScales,
            expectedSums);
    new PanamaVectorUtilSupport()
        .ggufQ4_KQ8_KMatVecDot(
            query, weightSegment, rows, cols, actual, actualQuants, actualScales, actualSums);

    assertThat(actualQuants).containsExactly(expectedQuants);
    assertThat(actualScales).containsExactly(expectedScales);
    assertThat(actualSums).containsExactly(expectedSums);
    assertThat(actual).containsExactly(expected);
  }

  @Test
  void q5_KQ8_KKernelMatchesScalarReferenceAcrossRowsAndBlocks() {
    int rows = 512;
    int cols = 2048;
    Random random = new Random(0x515B48L);
    float[] query = new float[cols];
    for (int index = 0; index < cols; index++) {
      query[index] = random.nextFloat() * 4.0f - 2.0f;
    }

    byte[] weights = new byte[rows * (cols / 256) * 176];
    random.nextBytes(weights);
    ByteBuffer buffer = ByteBuffer.wrap(weights).order(ByteOrder.LITTLE_ENDIAN);
    for (int offset = 0; offset < weights.length; offset += 176) {
      float scale = random.nextFloat() * 0.05f + 0.001f;
      float minScale = random.nextFloat() * 0.05f;
      buffer.putShort(offset, Float.floatToFloat16(scale));
      buffer.putShort(offset + Short.BYTES, Float.floatToFloat16(minScale));
    }

    float[] expected = new float[rows];
    float[] actual = new float[rows];
    byte[] expectedQuants = new byte[cols];
    byte[] actualQuants = new byte[cols];
    float[] expectedScales = new float[cols / 256];
    float[] actualScales = new float[cols / 256];
    short[] expectedSums = new short[cols / 16];
    short[] actualSums = new short[cols / 16];
    MemorySegment weightSegment = MemorySegment.ofArray(weights);

    new ScalarVectorUtilSupport()
        .ggufQ5_KQ8_KMatVecDot(
            query,
            weightSegment,
            rows,
            cols,
            expected,
            expectedQuants,
            expectedScales,
            expectedSums);
    new PanamaVectorUtilSupport()
        .ggufQ5_KQ8_KMatVecDot(
            query, weightSegment, rows, cols, actual, actualQuants, actualScales, actualSums);

    assertThat(actualQuants).containsExactly(expectedQuants);
    assertThat(actualScales).containsExactly(expectedScales);
    assertThat(actualSums).containsExactly(expectedSums);
    assertThat(actual).containsExactly(expected);
  }

  @Test
  void q6_KQ8_KKernelMatchesScalarReferenceAcrossRowsAndBlocks() {
    int rows = 512;
    int cols = 2048;
    Random random = new Random(0x516B48L);
    float[] query = new float[cols];
    for (int index = 0; index < cols; index++) {
      query[index] = random.nextFloat() * 4.0f - 2.0f;
    }

    byte[] weights = new byte[rows * (cols / 256) * 210];
    random.nextBytes(weights);
    ByteBuffer buffer = ByteBuffer.wrap(weights).order(ByteOrder.LITTLE_ENDIAN);
    for (int offset = 0; offset < weights.length; offset += 210) {
      float scale = random.nextFloat() * 0.05f + 0.001f;
      buffer.putShort(offset + 208, Float.floatToFloat16(scale));
    }

    float[] expected = new float[rows];
    float[] actual = new float[rows];
    byte[] expectedQuants = new byte[cols];
    byte[] actualQuants = new byte[cols];
    float[] expectedScales = new float[cols / 256];
    float[] actualScales = new float[cols / 256];
    MemorySegment weightSegment = MemorySegment.ofArray(weights);

    new ScalarVectorUtilSupport()
        .ggufQ6_KQ8_KMatVecDot(
            query, weightSegment, rows, cols, expected, expectedQuants, expectedScales);
    new PanamaVectorUtilSupport()
        .ggufQ6_KQ8_KMatVecDot(
            query, weightSegment, rows, cols, actual, actualQuants, actualScales);

    assertThat(actualQuants).containsExactly(expectedQuants);
    assertThat(actualScales).containsExactly(expectedScales);
    assertThat(actual).hasSameSizeAs(expected);
    for (int row = 0; row < rows; row++) {
      assertThat(actual[row]).isCloseTo(expected[row], offset(1e-3f));
    }
  }

  @Test
  void q5_0Q8_0KernelMatchesScalarReferenceAcrossRowsAndBlocks() {
    int rows = 512;
    int cols = 2048;
    Random random = new Random(0x515048L);
    float[] query = new float[cols];
    for (int index = 0; index < cols; index++) {
      query[index] = random.nextFloat() * 4.0f - 2.0f;
    }

    byte[] weights = new byte[rows * (cols / 32) * 22];
    random.nextBytes(weights);
    ByteBuffer buffer = ByteBuffer.wrap(weights).order(ByteOrder.LITTLE_ENDIAN);
    for (int offset = 0; offset < weights.length; offset += 22) {
      float scale = random.nextFloat() * 0.05f + 0.001f;
      buffer.putShort(offset, Float.floatToFloat16(scale));
    }

    float[] expected = new float[rows];
    float[] actual = new float[rows];
    byte[] expectedQuants = new byte[cols];
    byte[] actualQuants = new byte[cols];
    float[] expectedScales = new float[cols / 32];
    float[] actualScales = new float[cols / 32];
    MemorySegment weightSegment = MemorySegment.ofArray(weights);

    new ScalarVectorUtilSupport()
        .ggufQ5_0Q8_0MatVecDot(
            query, weightSegment, rows, cols, expected, expectedQuants, expectedScales);
    new PanamaVectorUtilSupport()
        .ggufQ5_0Q8_0MatVecDot(
            query, weightSegment, rows, cols, actual, actualQuants, actualScales);

    assertThat(actualQuants).containsExactly(expectedQuants);
    assertThat(actualScales).containsExactly(expectedScales);
    assertThat(actual).containsExactly(expected);
  }

  @Test
  void q5_0Q8_0BatchedKernelMatchesScalarReferenceExactly() {
    int batchSize = 4;
    int rows = 17;
    int cols = 256;
    Random random = new Random(0xB515048L);
    float[] queries = new float[batchSize * cols];
    for (int index = 0; index < queries.length; index++) {
      queries[index] = random.nextFloat() * 4.0f - 2.0f;
    }

    byte[] weights = new byte[rows * (cols / 32) * 22];
    random.nextBytes(weights);
    ByteBuffer buffer = ByteBuffer.wrap(weights).order(ByteOrder.LITTLE_ENDIAN);
    for (int offset = 0; offset < weights.length; offset += 22) {
      buffer.putShort(offset, Float.floatToFloat16(random.nextFloat() * 0.05f + 0.001f));
    }

    float[] expected = new float[batchSize * rows];
    float[] actual = new float[batchSize * rows];
    byte[] expectedQuants = new byte[batchSize * cols];
    byte[] actualQuants = new byte[batchSize * cols];
    float[] expectedScales = new float[batchSize * (cols / 32)];
    float[] actualScales = new float[batchSize * (cols / 32)];
    MemorySegment weightSegment = MemorySegment.ofArray(weights);

    new ScalarVectorUtilSupport()
        .ggufQ5_0Q8_0BatchedMatmul(
            queries,
            weightSegment,
            batchSize,
            rows,
            cols,
            expected,
            expectedQuants,
            expectedScales);
    new PanamaVectorUtilSupport()
        .ggufQ5_0Q8_0BatchedMatmul(
            queries, weightSegment, batchSize, rows, cols, actual, actualQuants, actualScales);

    assertThat(actualQuants).containsExactly(expectedQuants);
    assertThat(actualScales).containsExactly(expectedScales);
    assertThat(actual).containsExactly(expected);
  }
}
