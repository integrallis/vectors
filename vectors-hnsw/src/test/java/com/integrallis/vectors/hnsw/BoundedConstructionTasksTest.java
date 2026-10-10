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

import com.integrallis.vectors.core.SimilarityFunction;
import java.util.Arrays;
import java.util.SplittableRandom;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class BoundedConstructionTasksTest {
  @Test
  void appendSchedulesAtMostOneTaskPerWorkerAndRetainsCallerExecutor() throws Exception {
    float[][] rows = rows(600);
    HnswGraph prefix =
        ConcurrentHnswGraphBuilder.create(
                8,
                100,
                new InMemoryVectors(Arrays.copyOf(rows, 200)),
                SimilarityFunction.EUCLIDEAN,
                42)
            .build(1);
    HnswGraph owned = new HnswGraph(600, 8);
    for (int i = 0; i < prefix.size(); i++) {
      owned.initNode(i, prefix.nodeLevel(i));
      for (int layer = 0; layer <= prefix.nodeLevel(i); layer++)
        owned.getNeighbors(i, layer).copyFrom(prefix.getNeighbors(i, layer));
    }
    owned.setEntryNode(prefix.entryNode(), prefix.maxLevel());
    ReentrantLock[] locks = new ReentrantLock[600];
    Arrays.setAll(locks, i -> new ReentrantLock());
    AtomicInteger submitted = new AtomicInteger();
    try (var executor =
        new ThreadPoolExecutor(4, 4, 0, TimeUnit.SECONDS, new LinkedBlockingQueue<Runnable>()) {
          @Override
          public void execute(Runnable command) {
            submitted.incrementAndGet();
            super.execute(command);
          }
        }) {
      var builder =
          ConcurrentHnswGraphBuilder.create(
              8, 100, new InMemoryVectors(rows), SimilarityFunction.EUCLIDEAN, 43);
      assertSame(owned, builder.appendInPlace(owned, 200, 4, locks, executor));
      assertEquals(600, owned.size());
      assertFalse(executor.isShutdown());
      assertTrue(submitted.get() <= 4, "submitted " + submitted.get() + " tasks for 4 workers");
      assertEquals(7, executor.submit(() -> 7).get());
      HnswSearcher searcher =
          new HnswSearcher(owned, new InMemoryVectors(rows), SimilarityFunction.EUCLIDEAN);
      for (int i = 200; i < 600; i++) {
        assertTrue(owned.getNeighbors(i, 0).size() > 0);
        assertEquals(i, searcher.search(rows[i], 1, 600).nodeIds()[0]);
      }
      assertSame(owned, builder.appendInPlace(owned, 600, 4, locks, executor));
    }
  }

  @Test
  void rejectsInvalidInPlaceInputsBeforeMutatingGraph() {
    var builder =
        ConcurrentHnswGraphBuilder.create(
            8, 100, new InMemoryVectors(rows(10)), SimilarityFunction.EUCLIDEAN, 42);
    HnswGraph graph = new HnswGraph(10, 8);
    graph.initNode(0, 0);
    graph.setEntryNode(0, 0);
    assertThrows(NullPointerException.class, () -> builder.appendInPlace(null, 0, 1));
    assertThrows(IllegalArgumentException.class, () -> builder.appendInPlace(graph, 0, 1));
    assertThrows(
        IllegalArgumentException.class, () -> builder.appendInPlace(new HnswGraph(1, 8), 0, 1));
    assertThrows(
        IllegalArgumentException.class, () -> builder.appendInPlace(new HnswGraph(10, 4), 0, 1));
    assertThrows(
        IllegalArgumentException.class,
        () -> builder.appendInPlace(graph, 1, 1, new ReentrantLock[1], null));
    assertEquals(1, graph.size());
  }

  private static float[][] rows(int size) {
    SplittableRandom random = new SplittableRandom(9832);
    float[][] rows = new float[size][16];
    for (float[] row : rows)
      for (int d = 0; d < row.length; d++) row[d] = (float) random.nextDouble();
    return rows;
  }
}
