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
class DistinctNeighborInsertTest {
  @Test
  void knownDistinctInsertionPreservesLegacyOrderAndRawScores() {
    var random = new SplittableRandom(63182);
    for (int capacity : new int[] {1, 2, 16, 32, 200}) {
      for (int mode = 0; mode < 6; mode++) {
        var expected = new NeighborArray(capacity);
        var actual = new NeighborArray(capacity);
        for (int id = 0; id < capacity; id++) {
          float score =
              switch (mode) {
                case 0 -> random.nextFloat();
                case 1 -> random.nextInt(4) / 4f;
                case 2 -> id / (float) capacity;
                case 3 -> 1f - id / (float) capacity;
                case 4 -> id % 2 == 0 ? -0f : 0f;
                default -> id % 2 == 0 ? Float.MIN_VALUE : Float.POSITIVE_INFINITY;
              };
          assertTrue(expected.insert(id, score));
          actual.insertDistinct(id, score);
          assertSameEntries(expected, actual);
        }
        // Public insertion must keep its duplicate protection, even for a better new score.
        assertFalse(actual.insert(actual.node(0), Float.POSITIVE_INFINITY));
        assertSameEntries(expected, actual);
      }
    }
  }

  @Test
  void fullDestinationIsRejectedWithoutMutation() {
    var actual = new NeighborArray(1);
    actual.insertDistinct(3, .5f);
    assertThrows(IllegalStateException.class, () -> actual.insertDistinct(7, 1f));
    assertEquals(1, actual.size());
    assertEquals(3, actual.node(0));
    assertEquals(.5f, actual.score(0));
  }

  private static void assertSameEntries(NeighborArray expected, NeighborArray actual) {
    assertEquals(expected.size(), actual.size());
    for (int i = 0; i < expected.size(); i++) {
      assertEquals(expected.node(i), actual.node(i));
      assertEquals(
          Float.floatToRawIntBits(expected.score(i)), Float.floatToRawIntBits(actual.score(i)));
    }
  }
}
