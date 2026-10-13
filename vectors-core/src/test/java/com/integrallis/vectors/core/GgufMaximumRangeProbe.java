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

import java.nio.ByteBuffer;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HexFormat;
import java.util.List;
import java.util.concurrent.ConcurrentLinkedQueue;

/** Bounded real dispatch checks for integer-limit ranges and ordinary partition equivalence. */
final class GgufMaximumRangeProbe {
  public static void main(String[] args) throws Exception {
    switch (args[0]) {
      case "rows" -> maximumRows();
      case "stages" -> maximumStages();
      case "maximum-chunks" -> maximumChunks();
      case "normal" -> normalRanges();
      default -> throw new IllegalArgumentException(args[0]);
    }
  }

  static void maximumRows() {
    try (var executor = new GgufPersistentRowExecutor(2, 1, "maximum-rows")) {
      try {
        executor.forEach(
            Integer.MAX_VALUE,
            row -> {
              if (row < 0 || row >= Integer.MAX_VALUE)
                throw new AssertionError("invalid row " + row);
              throw new FirstValidRange();
            });
        throw new AssertionError("no row was executed");
      } catch (FirstValidRange expected) {
      }
    }
  }

  static void maximumStages() {
    for (int count : new int[] {Integer.MAX_VALUE, Integer.MAX_VALUE - 1, 1073741825}) {
      for (int chunks : new int[] {1, 2, 3, 1024}) {
        ranges(count, 2, chunks);
      }
    }
  }

  static void maximumChunks() {
    try (var executor = new GgufPersistentRowExecutor(2, Integer.MAX_VALUE, "maximum-chunks")) {
      try {
        executor.execute(
            GgufStagePlan.of(
                GgufStagePlan.stage(
                    Integer.MAX_VALUE,
                    (start, end) -> {
                      if (start < 0 || start >= end || end > Integer.MAX_VALUE)
                        throw new AssertionError("invalid range " + start + ":" + end);
                      throw new FirstValidRange();
                    })));
        throw new AssertionError("no range was executed");
      } catch (FirstValidRange expected) {
      }
    }
  }

  private static List<int[]> ranges(int count, int parallelism, int chunks) {
    var actual = new ConcurrentLinkedQueue<int[]>();
    try (var executor = new GgufPersistentRowExecutor(parallelism, chunks, "range-proof")) {
      executor.execute(
          GgufStagePlan.of(
              GgufStagePlan.stage(count, (start, end) -> actual.add(new int[] {start, end}))));
    }
    var sorted = new ArrayList<>(actual);
    sorted.sort(Comparator.comparingInt(a -> a[0]));
    int next = 0;
    for (int[] range : sorted) {
      if (range[0] != next || range[1] <= range[0] || range[1] > count)
        throw new AssertionError(
            "invalid partition " + range[0] + ":" + range[1] + " expected " + next);
      next = range[1];
    }
    if (next != count) throw new AssertionError("missing rows: ended at " + next + " of " + count);
    return sorted;
  }

  static void normalRanges() throws Exception {
    var digest = MessageDigest.getInstance("SHA-256");
    for (int count : new int[] {1, 2, 3, 17, 257, 4864, 151936}) {
      for (int parallelism : new int[] {2, 4}) {
        for (int chunks : new int[] {1, 2, 3, 17, Integer.MAX_VALUE}) {
          for (int[] range : ranges(count, parallelism, chunks)) {
            digest.update(ByteBuffer.allocate(8).putInt(range[0]).putInt(range[1]).array());
          }
        }
      }
    }
    System.out.println("NORMAL_RANGES_SHA256=" + HexFormat.of().formatHex(digest.digest()));
  }

  private static final class FirstValidRange extends RuntimeException {
    private static final long serialVersionUID = 1L;
  }
}
