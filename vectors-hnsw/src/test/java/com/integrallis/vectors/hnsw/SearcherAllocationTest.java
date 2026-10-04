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

import static org.junit.jupiter.api.Assertions.assertTrue;

import com.integrallis.vectors.core.SimilarityFunction;
import java.lang.management.ManagementFactory;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class SearcherAllocationTest {
  private static volatile HnswSearcher retained;

  @Test
  void searcherDoesNotAllocateGraphSizedResultScratch() {
    int count = 50000;
    HnswGraph graph = new HnswGraph(count, 16);
    for (int i = 0; i < count; i++) graph.initNode(i, 0);
    graph.setEntryNode(0, 0);
    float[] row = {.5f, .5f, .5f, .5f};
    RandomAccessVectors vectors =
        new RandomAccessVectors() {
          public int size() {
            return count;
          }

          public int dimension() {
            return row.length;
          }

          public float[] getVector(int ordinal) {
            return row;
          }

          public boolean sharesReturnBuffer() {
            return false;
          }
        };
    var index = HnswIndex.ofPrebuilt(graph, vectors, SimilarityFunction.COSINE);
    var bean = (com.sun.management.ThreadMXBean) ManagementFactory.getThreadMXBean();
    assertTrue(bean.isThreadAllocatedMemorySupported());
    bean.setThreadAllocatedMemoryEnabled(true);
    retained = index.searcher(); // initialize classes before measuring
    long thread = Thread.currentThread().threadId();
    long before = bean.getThreadAllocatedBytes(thread);
    retained = index.searcher();
    long allocated = bean.getThreadAllocatedBytes(thread) - before;
    System.out.println("searcher allocation: " + allocated + " bytes for " + count + " nodes");
    // One int per node for visited generations plus bounded queues/scoring scratch.
    // The old extra int[] + float[] consumed another eight bytes per graph node.
    assertTrue(
        allocated < 8L * count, "searcher allocation must stay below 8 bytes/node: " + allocated);
  }
}
