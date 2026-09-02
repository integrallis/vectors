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

import com.integrallis.vectors.core.BFloat16Matrix;
import com.integrallis.vectors.core.VectorUtil;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;

/** Compares compact MobileMoE-style {@code x @ W} INT4 execution with expanded BF16. */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(2)
public class PackedInt4RightMatrixBenchmark {

  @State(Scope.Thread)
  public static class MatrixState {

    @Param({"gate-up", "down"})
    String shape;

    @Param({"1", "4", "8", "16"})
    int batchSize;

    int inputs;
    int outputs;
    MemorySegment packed;
    MemorySegment scales;
    MemorySegment q8Weights;
    BFloat16Matrix bfloat16;
    float[] activations;
    float[] result;
    byte[] q8Activations;
    float[] q8ActivationScales;

    @Setup
    public void setup() {
      inputs = "gate-up".equals(shape) ? 768 : 384;
      outputs = 768;
      int groupSize = 32;
      int groupsPerInput = outputs / groupSize;
      Random random = new Random(0x4d4f42494c454d4fL);
      byte[] packedBytes = new byte[inputs * outputs / 2];
      ByteBuffer scaleBytes =
          ByteBuffer.allocate(inputs * groupsPerInput * Short.BYTES).order(ByteOrder.LITTLE_ENDIAN);
      float[] scaleValues = new float[inputs * groupsPerInput];
      for (int index = 0; index < scaleValues.length; index++) {
        float scale =
            Float.float16ToFloat(Float.floatToFloat16(0.01f + random.nextFloat() * 0.09f));
        scaleValues[index] = scale;
        scaleBytes.putShort(Float.floatToFloat16(scale));
      }
      byte[] denseBytes = new byte[outputs * inputs * Short.BYTES];
      float[] denseValues = new float[outputs * inputs];
      for (int input = 0; input < inputs; input++) {
        for (int output = 0; output < outputs; output += 2) {
          int even = random.nextInt(16) - 8;
          int odd = random.nextInt(16) - 8;
          packedBytes[(input * outputs + output) / 2] =
              (byte) ((even & 0x0f) | ((odd & 0x0f) << 4));
          float evenValue = even * scaleValues[input * groupsPerInput + output / groupSize];
          float oddValue = odd * scaleValues[input * groupsPerInput + output / groupSize];
          denseValues[output * inputs + input] = evenValue;
          denseValues[(output + 1) * inputs + input] = oddValue;
          putBfloat16(denseBytes, output * inputs + input, evenValue);
          putBfloat16(denseBytes, (output + 1) * inputs + input, oddValue);
        }
      }
      packed = MemorySegment.ofArray(packedBytes);
      scales = MemorySegment.ofArray(scaleBytes.array());
      bfloat16 = BFloat16Matrix.of(MemorySegment.ofArray(denseBytes), outputs, inputs);
      q8Weights = MemorySegment.ofArray(quantizeQ8Weights(denseValues, outputs, inputs));
      activations = new float[batchSize * inputs];
      for (int index = 0; index < activations.length; index++) {
        activations[index] = random.nextFloat(-1.0f, 1.0f);
      }
      result = new float[batchSize * outputs];
      q8Activations = new byte[batchSize * inputs];
      q8ActivationScales = new float[batchSize * inputs / 32];
    }

    private static void putBfloat16(byte[] bytes, int index, float value) {
      int bits = Float.floatToRawIntBits(value);
      short bfloat16 = (short) ((bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16);
      int offset = index * Short.BYTES;
      bytes[offset] = (byte) bfloat16;
      bytes[offset + 1] = (byte) (bfloat16 >>> Byte.SIZE);
    }

    private static byte[] quantizeQ8Weights(float[] values, int rows, int columns) {
      int blocks = columns / 32;
      ByteBuffer bytes = ByteBuffer.allocate(rows * blocks * 34).order(ByteOrder.LITTLE_ENDIAN);
      for (int row = 0; row < rows; row++) {
        for (int block = 0; block < blocks; block++) {
          int offset = row * columns + block * 32;
          float maximum = 0.0f;
          for (int index = 0; index < 32; index++) {
            maximum = Math.max(maximum, Math.abs(values[offset + index]));
          }
          float scale = maximum / 127.0f;
          bytes.putShort(Float.floatToFloat16(scale));
          float inverse = maximum == 0.0f ? 0.0f : 127.0f / maximum;
          for (int index = 0; index < 32; index++) {
            bytes.put((byte) Math.clamp(Math.round(values[offset + index] * inverse), -127, 127));
          }
        }
      }
      return bytes.array();
    }
  }

  @Benchmark
  public float[] packedInt4(MatrixState state) {
    VectorUtil.packedInt4GroupRightMatVecBatch(
        state.activations,
        state.batchSize,
        state.packed,
        state.scales,
        state.inputs,
        state.outputs,
        32,
        state.result);
    return state.result;
  }

  @Benchmark
  public float[] expandedBfloat16(MatrixState state) {
    state.bfloat16.multiplyBatch(state.activations, 0, state.batchSize, state.result, 0);
    return state.result;
  }

  @Benchmark
  public float[] requantizedQ8(MatrixState state) {
    VectorUtil.ggufQ8_0Q8_0BatchedMatmul(
        state.activations,
        state.q8Weights,
        state.batchSize,
        state.outputs,
        state.inputs,
        state.result,
        state.q8Activations,
        state.q8ActivationScales);
    return state.result;
  }
}
