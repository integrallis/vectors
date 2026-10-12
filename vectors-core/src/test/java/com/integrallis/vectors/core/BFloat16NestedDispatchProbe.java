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
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

/** Child-process regression check: nested work must never republish its active worker pool. */
final class BFloat16NestedDispatchProbe {
  public static void main(String[] args) {
    int rows = 1024, columns = 1025;
    byte[] bytes = new byte[rows * columns * 2];
    var buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
    var random = new Random(0xBF16AABB);
    while (buffer.hasRemaining())
      buffer.putShort((short) (Float.floatToRawIntBits((random.nextInt(17) - 8) * 0.125f) >>> 16));
    float[] input = new float[columns], expected = new float[rows];
    for (int i = 0; i < input.length; i++) input[i] = (random.nextInt(17) - 8) * 0.125f;
    try (var confined = Arena.ofConfined();
        var shared = Arena.ofShared()) {
      var local = confined.allocate(bytes.length, 8);
      local.copyFrom(MemorySegment.ofArray(bytes));
      BFloat16Matrix.of(local, rows, columns).multiply(input, expected);
      var weights = shared.allocate(bytes.length, 8);
      weights.copyFrom(local);
      var matrix = BFloat16Matrix.of(weights, rows, columns);
      var inner =
          GgufStagePlan.of(
              GgufStagePlan.stage(
                  1,
                  (a, b) -> {
                    float[] actual = new float[rows];
                    matrix.multiply(input, actual);
                    if (!Arrays.equals(expected, actual))
                      throw new AssertionError("nested matvec changed output");
                    float[] batchInput = new float[2 * columns], batchOutput = new float[2 * rows];
                    System.arraycopy(input, 0, batchInput, 0, columns);
                    System.arraycopy(input, 0, batchInput, columns, columns);
                    matrix.multiplyBatch(batchInput, 0, 2, batchOutput, 0);
                    for (int i = 0; i < 2 * rows; i++)
                      if (Float.floatToRawIntBits(batchOutput[i])
                          != Float.floatToRawIntBits(expected[i % rows]))
                        throw new AssertionError("nested batch changed output");
                  }));
      System.out.println("ENTER_NESTED_DISPATCH");
      GgufStagePlan.of(
              GgufStagePlan.stage(
                  4,
                  (a, b) -> {
                    for (int i = a; i < b; i++) inner.execute();
                  }))
          .execute();
      matrix.multiply(input, expected);
      crossPoolDispatch();
      System.out.println("NESTED_DISPATCH_COMPLETE");
    }
  }

  private static void crossPoolDispatch() {
    System.out.println("ENTER_CROSS_POOL_DISPATCH");
    int[] actual = new int[8];
    try (var first = new GgufPersistentRowExecutor(2, 1, "bf16-nested-first");
        var second = new GgufPersistentRowExecutor(2, 1, "bf16-nested-second")) {
      first.forEach(
          1,
          ignored -> {
            Thread outer = Thread.currentThread();
            var otherCallback = new CountDownLatch(1);
            second.forEach(
                2,
                row -> {
                  // Give a second-pool worker the opportunity to reenter the first pool. Correct
                  // nested dispatch executes inline, so the bounded wait then simply expires.
                  if (Thread.currentThread() == outer) {
                    try {
                      otherCallback.await(200, TimeUnit.MILLISECONDS);
                    } catch (InterruptedException failure) {
                      Thread.currentThread().interrupt();
                      throw new AssertionError(failure);
                    }
                  } else {
                    otherCallback.countDown();
                  }
                  first.forEach(4, item -> actual[row * 4 + item]++);
                });
          });
      for (int value : actual) if (value != 1) throw new AssertionError("cross-pool row lost");
      first.forEach(8, row -> actual[row]++);
      second.forEach(8, row -> actual[row]++);
      for (int value : actual) if (value != 3) throw new AssertionError("cross-pool reuse failed");
    }
    System.out.println("CROSS_POOL_DISPATCH_COMPLETE");
  }
}
