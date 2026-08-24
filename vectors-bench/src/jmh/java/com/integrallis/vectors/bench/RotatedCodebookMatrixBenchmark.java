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
package com.integrallis.vectors.bench;

import com.integrallis.vectors.core.RotatedCodebookMatrix;
import com.integrallis.vectors.core.VectorUtil;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

/** Compares compressed rotated-codebook matvec with an expanded F32 matrix. */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Thread)
@Fork(
    value = 1,
    jvmArgsPrepend = {"--add-modules", "jdk.incubator.vector"})
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
public class RotatedCodebookMatrixBenchmark {

  private static final int GROUP_SIZE = 128;

  /** Needle projection and embedding-row counts. */
  @Param({"512", "8192"})
  int rows;

  /** Needle 2 model width. */
  @Param("512")
  int columns;

  /** Mixed storage encodings present in the released Needle 2 artifact. */
  @Param({"CQ2", "CQ4"})
  String encodingName;

  private RotatedCodebookMatrix matrix;
  private RotatedCodebookMatrix.PreparedActivation prepared;
  private float[] input;
  private float[] dense;
  private float[] output;

  /** Creates one deterministic matrix and verifies the compressed and expanded paths agree. */
  @Setup(Level.Trial)
  public void setUp() {
    RotatedCodebookMatrix.Encoding encoding = RotatedCodebookMatrix.Encoding.valueOf(encodingName);
    int packedBits = encoding == RotatedCodebookMatrix.Encoding.CQ2 ? 2 : 4;
    int levels = 1 << packedBits;
    float[] codebook = codebook(levels);
    int rowBytes = columns * packedBits / Byte.SIZE;
    byte[] codes = new byte[Math.multiplyExact(rows, rowBytes)];
    byte[] norms = new byte[Math.multiplyExact(rows, columns / GROUP_SIZE * Short.BYTES)];
    Random random = new Random(42L + rows + packedBits);
    random.nextBytes(codes);
    ByteBuffer normBuffer = ByteBuffer.wrap(norms).order(ByteOrder.LITTLE_ENDIAN);
    while (normBuffer.hasRemaining()) {
      normBuffer.putShort(Float.floatToFloat16(0.75f + random.nextFloat() * 0.5f));
    }

    matrix =
        RotatedCodebookMatrix.of(
            MemorySegment.ofArray(codes),
            MemorySegment.ofArray(norms),
            rows,
            columns,
            GROUP_SIZE,
            encoding,
            codebook);
    input = new float[columns];
    for (int index = 0; index < columns; index++) {
      input[index] = random.nextFloat() * 2.0f - 1.0f;
    }
    prepared = matrix.prepare(input);
    dense = dequantize(codes, norms, rows, columns, packedBits, codebook);
    output = new float[rows];

    float[] packedOutput = new float[rows];
    float[] denseOutput = new float[rows];
    matrix.multiply(prepared, packedOutput);
    VectorUtil.batchDotProduct(input, dense, rows, columns, denseOutput);
    for (int row = 0; row < rows; row++) {
      if (Math.abs(packedOutput[row] - denseOutput[row]) > 1.0e-3f) {
        throw new IllegalStateException("compressed and expanded benchmark matrices disagree");
      }
    }
  }

  /** Measures the one-time activation transform and byte-lookup construction. */
  @Benchmark
  public void prepareActivation(Blackhole blackhole) {
    blackhole.consume(matrix.prepare(input));
  }

  /** Measures compressed matvec after activation preparation. */
  @Benchmark
  public void compressedPreparedMatVec(Blackhole blackhole) {
    matrix.multiply(prepared, output);
    blackhole.consume(output);
  }

  /** Measures a one-off compressed matvec including activation preparation. */
  @Benchmark
  public void compressedOneShotMatVec(Blackhole blackhole) {
    matrix.multiply(matrix.prepare(input), output);
    blackhole.consume(output);
  }

  /** Measures the existing SIMD F32 matvec over a fully expanded matrix. */
  @Benchmark
  public void expandedF32MatVec(Blackhole blackhole) {
    VectorUtil.batchDotProduct(input, dense, rows, columns, output);
    blackhole.consume(output);
  }

  private static float[] codebook(int levels) {
    float[] codebook = new float[levels];
    float scale = (float) (1.0 / Math.sqrt(GROUP_SIZE));
    for (int index = 0; index < levels; index++) {
      codebook[index] = (-1.0f + 2.0f * index / (levels - 1)) * scale;
    }
    return codebook;
  }

  private static float[] dequantize(
      byte[] codes, byte[] norms, int rows, int columns, int packedBits, float[] codebook) {
    int groups = columns / GROUP_SIZE;
    int rowBytes = columns * packedBits / Byte.SIZE;
    int mask = (1 << packedBits) - 1;
    ByteBuffer normBuffer = ByteBuffer.wrap(norms).order(ByteOrder.LITTLE_ENDIAN);
    float[] dense = new float[Math.multiplyExact(rows, columns)];
    float[] group = new float[GROUP_SIZE];
    float hadamardScale = (float) (1.0 / Math.sqrt(GROUP_SIZE));
    for (int row = 0; row < rows; row++) {
      for (int groupIndex = 0; groupIndex < groups; groupIndex++) {
        int columnBase = groupIndex * GROUP_SIZE;
        for (int index = 0; index < GROUP_SIZE; index++) {
          int valueIndex = columnBase + index;
          int bitIndex = valueIndex * packedBits;
          int byteIndex = row * rowBytes + (bitIndex >>> 3);
          int shift = bitIndex & 7;
          int code = Byte.toUnsignedInt(codes[byteIndex]) >>> shift;
          if (Byte.SIZE - shift < packedBits) {
            code |= Byte.toUnsignedInt(codes[byteIndex + 1]) << (Byte.SIZE - shift);
          }
          group[index] = codebook[code & mask];
        }
        hadamardInPlace(group);
        float norm =
            Float.float16ToFloat(normBuffer.getShort((row * groups + groupIndex) * Short.BYTES));
        int outputBase = row * columns + columnBase;
        for (int index = 0; index < GROUP_SIZE; index++) {
          dense[outputBase + index] = group[index] * hadamardScale * norm;
        }
      }
    }
    return dense;
  }

  private static void hadamardInPlace(float[] values) {
    for (int stride = 1; stride < values.length; stride *= 2) {
      for (int base = 0; base < values.length; base += 2 * stride) {
        for (int index = 0; index < stride; index++) {
          float left = values[base + index];
          float right = values[base + stride + index];
          values[base + index] = left + right;
          values[base + stride + index] = left - right;
        }
      }
    }
  }
}
