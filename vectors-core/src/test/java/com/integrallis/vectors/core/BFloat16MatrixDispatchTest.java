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

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Random;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;

class BFloat16MatrixDispatchTest {
  @Test
  void sharedConcurrentMatvecMatchesConfinedCallingThreadExecution() throws Exception {
    int rows = 1027;
    int columns = 1025;
    byte[] bytes = new byte[rows * columns * Short.BYTES];
    var packed = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
    var random = new Random(0xBF16D15CL);
    while (packed.hasRemaining()) {
      packed.putShort((short) (Float.floatToRawIntBits(random.nextFloat() - 0.5f) >>> 16));
    }
    float[][] input = new float[2][columns];
    float[][] expected = new float[2][rows];
    for (float[] vector : input) {
      for (int column = 0; column < columns; column++) vector[column] = random.nextFloat() - 0.5f;
    }
    try (var confined = Arena.ofConfined();
        var shared = Arena.ofShared()) {
      var localWeights = confined.allocate(bytes.length, Long.BYTES);
      localWeights.copyFrom(MemorySegment.ofArray(bytes));
      var sharedWeights = shared.allocate(bytes.length, Long.BYTES);
      sharedWeights.copyFrom(localWeights);
      var localMatrix = BFloat16Matrix.of(localWeights, rows, columns);
      var sharedMatrix = BFloat16Matrix.of(sharedWeights, rows, columns);
      for (int index = 0; index < input.length; index++) {
        localMatrix.multiply(input[index], expected[index]);
      }
      try (var callers = Executors.newFixedThreadPool(2)) {
        var first = callers.submit(() -> repeat(sharedMatrix, input[0], expected[0]));
        var second = callers.submit(() -> repeat(sharedMatrix, input[1], expected[1]));
        first.get(60, TimeUnit.SECONDS);
        second.get(60, TimeUnit.SECONDS);
      }
    }
  }

  private static void repeat(BFloat16Matrix matrix, float[] input, float[] expected) {
    float[] output = new float[expected.length];
    for (int iteration = 0; iteration < 20; iteration++) {
      matrix.multiply(input, output);
      assertThat(output).containsExactly(expected);
    }
  }
}
