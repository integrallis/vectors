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

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.management.ManagementFactory;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Random;

/** Fresh-JVM allocation check for compiled public BF16 prompt batches. */
final class BFloat16MatrixAllocationProbe {
  private static volatile float sink;

  private BFloat16MatrixAllocationProbe() {}

  public static void main(String[] args) {
    var bean = (com.sun.management.ThreadMXBean) ManagementFactory.getThreadMXBean();
    if (!bean.isThreadAllocatedMemorySupported())
      throw new AssertionError("allocation accounting unavailable");
    bean.setThreadAllocatedMemoryEnabled(true);
    int rows = 48;
    int columns = 5120;
    byte[] packed = new byte[rows * columns * Short.BYTES];
    ByteBuffer weights = ByteBuffer.wrap(packed).order(ByteOrder.LITTLE_ENDIAN);
    Random random = new Random(0xBF16A110CL);
    for (int i = 0; i < rows * columns; i++)
      weights.putShort((short) (Float.floatToRawIntBits(random.nextFloat() * 0.1f - 0.05f) >>> 16));
    try (Arena arena = Arena.ofShared()) {
      MemorySegment data = arena.allocate(packed.length, Long.BYTES);
      data.copyFrom(MemorySegment.ofArray(packed));
      BFloat16Matrix matrix = BFloat16Matrix.of(data, rows, columns);
      for (int batch : new int[] {2, 13}) {
        float[] input = new float[batch * columns];
        float[] output = new float[batch * rows];
        for (int i = 0; i < input.length; i++) input[i] = random.nextFloat() - 0.5f;
        for (int i = 0; i < 5000; i++) project(matrix, input, output, batch, i);
        long thread = Thread.currentThread().threadId();
        long before = bean.getThreadAllocatedBytes(thread);
        for (int i = 0; i < 1000; i++) project(matrix, input, output, batch, i);
        long allocated = bean.getThreadAllocatedBytes(thread) - before;
        System.out.printf("BF16_ALLOC batch=%d iterations=1000 bytes=%d%n", batch, allocated);
        // Permit fixed per-call bookkeeping, but reject even one boxed vector per output row.
        if (allocated >= 256_000 || !Float.isFinite(sink))
          throw new AssertionError("BF16 batch " + batch + " allocated " + allocated + " bytes");
      }
    }
  }

  private static void project(
      BFloat16Matrix matrix, float[] input, float[] output, int batch, int i) {
    input[0] = (i % 97) * 0.001f;
    matrix.multiplyBatch(input, 0, batch, output, 0);
    sink = output[0];
  }
}
