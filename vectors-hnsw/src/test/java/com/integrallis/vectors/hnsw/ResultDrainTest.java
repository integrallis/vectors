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
package com.integrallis.vectors.hnsw;

import static org.junit.jupiter.api.Assertions.*;

import java.util.SplittableRandom;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class ResultDrainTest {
  @Test
  void drainMatchesLegacyOrderAndRawScoresIncludingTies() {
    SplittableRandom random = new SplittableRandom(93241);
    for (int size : new int[] {0, 1, 2, 10, 32, 200, 1000}) {
      for (int round = 0; round < 40; round++) {
        NodeQueue original = new NodeQueue(size, true);
        NodeQueue optimized = new NodeQueue(size, true);
        for (int id = 0; id < size; id++) {
          float score =
              switch (round % 4) {
                case 0 -> (float) random.nextDouble();
                case 1 -> random.nextInt(8) / 8f;
                case 2 -> 0f;
                default -> id % 3 == 0 ? Float.POSITIVE_INFINITY : Float.MIN_VALUE;
              };
          original.add(id, score);
          optimized.add(id, score);
        }
        NeighborArray expected = legacy(original);
        NeighborArray actual = NeighborArray.drainResults(optimized);
        assertTrue(optimized.isEmpty());
        assertEquals(expected.size(), actual.size());
        assertEquals(expected.maxSize(), actual.maxSize());
        for (int i = 0; i < size; i++) {
          assertEquals(expected.node(i), actual.node(i), "tie/order at " + i);
          assertEquals(
              Float.floatToRawIntBits(expected.score(i)), Float.floatToRawIntBits(actual.score(i)));
        }
        // The drained heap remains reusable by the next insertion.
        optimized.add(1001, .5f);
        assertEquals(1001, NodeQueue.nodeId(optimized.poll()));
      }
    }
  }

  private static NeighborArray legacy(NodeQueue queue) {
    int size = queue.size();
    NeighborArray result = new NeighborArray(Math.max(1, size));
    int[] nodes = new int[size];
    float[] scores = new float[size];
    for (int i = size - 1; i >= 0; i--) {
      long entry = queue.poll();
      nodes[i] = NodeQueue.nodeId(entry);
      scores[i] = NodeQueue.score(entry);
    }
    for (int i = 0; i < size; i++) result.insert(nodes[i], scores[i]);
    return result;
  }
}
