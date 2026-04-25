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
package com.integrallis.vectors.vamana;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Verifies that copied data structures work correctly in the vamana package. */
class DataStructuresTest {

  @Nested
  @Tag("unit")
  class NeighborArrayTests {

    @Test
    void insertMaintainsDescendingOrder() {
      var na = new NeighborArray(5);
      na.insert(0, 0.3f);
      na.insert(1, 0.9f);
      na.insert(2, 0.5f);

      assertThat(na.size()).isEqualTo(3);
      assertThat(na.score(0)).isEqualTo(0.9f);
      assertThat(na.score(1)).isEqualTo(0.5f);
      assertThat(na.score(2)).isEqualTo(0.3f);
    }

    @Test
    void insertRejectsDuplicates() {
      var na = new NeighborArray(5);
      assertThat(na.insert(0, 0.5f)).isTrue();
      assertThat(na.insert(0, 0.8f)).isFalse();
      assertThat(na.size()).isEqualTo(1);
    }

    @Test
    void insertEvictsWorstWhenFull() {
      var na = new NeighborArray(2);
      na.insert(0, 0.9f);
      na.insert(1, 0.5f);
      assertThat(na.insert(2, 0.7f)).isTrue();
      assertThat(na.size()).isEqualTo(2);
      assertThat(na.node(0)).isEqualTo(0);
      assertThat(na.node(1)).isEqualTo(2);
    }

    @Test
    void clearResetsSize() {
      var na = new NeighborArray(5);
      na.insert(0, 0.5f);
      na.insert(1, 0.3f);
      na.clear();
      assertThat(na.size()).isEqualTo(0);
    }

    @Test
    void addUnsortedAndSort() {
      var na = new NeighborArray(5);
      na.addUnsorted(0, 0.3f);
      na.addUnsorted(1, 0.9f);
      na.addUnsorted(2, 0.5f);
      na.sort();

      assertThat(na.size()).isEqualTo(3);
      assertThat(na.score(0)).isEqualTo(0.9f);
      assertThat(na.score(1)).isEqualTo(0.5f);
      assertThat(na.score(2)).isEqualTo(0.3f);
    }

    @Test
    void addUnsortedThrowsWhenFull() {
      var na = new NeighborArray(1);
      na.addUnsorted(0, 0.5f);
      assertThatThrownBy(() -> na.addUnsorted(1, 0.3f)).isInstanceOf(IllegalStateException.class);
    }

    @Test
    void containsFindsInsertedNode() {
      var na = new NeighborArray(5);
      na.insert(42, 0.5f);
      assertThat(na.contains(42)).isTrue();
      assertThat(na.contains(99)).isFalse();
    }

    @Test
    void copyFromReplacesContent() {
      var src = new NeighborArray(5);
      src.insert(0, 0.9f);
      src.insert(1, 0.5f);

      var dst = new NeighborArray(5);
      dst.insert(99, 0.1f);
      dst.copyFrom(src);

      assertThat(dst.size()).isEqualTo(2);
      assertThat(dst.node(0)).isEqualTo(0);
      assertThat(dst.node(1)).isEqualTo(1);
    }
  }

  @Nested
  @Tag("unit")
  class NodeQueueTests {

    @Test
    void minHeapPollsLowestFirst() {
      var q = new NodeQueue(4, true);
      q.add(0, 0.9f);
      q.add(1, 0.3f);
      q.add(2, 0.7f);

      assertThat(NodeQueue.nodeId(q.poll())).isEqualTo(1);
      assertThat(NodeQueue.nodeId(q.poll())).isEqualTo(2);
      assertThat(NodeQueue.nodeId(q.poll())).isEqualTo(0);
    }

    @Test
    void maxHeapPollsHighestFirst() {
      var q = new NodeQueue(4, false);
      q.add(0, 0.3f);
      q.add(1, 0.9f);
      q.add(2, 0.7f);

      assertThat(NodeQueue.nodeId(q.poll())).isEqualTo(1);
      assertThat(NodeQueue.nodeId(q.poll())).isEqualTo(2);
      assertThat(NodeQueue.nodeId(q.poll())).isEqualTo(0);
    }

    @Test
    void encodeDecodeRoundTrip() {
      long entry = NodeQueue.encode(42, 0.75f);
      assertThat(NodeQueue.nodeId(entry)).isEqualTo(42);
      assertThat(NodeQueue.score(entry)).isEqualTo(0.75f);
    }

    @Test
    void encodeRejectsNegativeScores() {
      assertThatThrownBy(() -> NodeQueue.encode(0, -1.0f))
          .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void insertWithOverflowMaintainsBound() {
      var q = new NodeQueue(4, true);
      q.insertWithOverflow(0, 0.5f, 2);
      q.insertWithOverflow(1, 0.3f, 2);
      assertThat(q.size()).isEqualTo(2);

      // Insert better score — should replace worst (0.3f)
      assertThat(q.insertWithOverflow(2, 0.7f, 2)).isTrue();
      assertThat(q.size()).isEqualTo(2);

      // Insert worse score — should be rejected
      assertThat(q.insertWithOverflow(3, 0.2f, 2)).isFalse();
    }

    @Test
    void clearResetsQueue() {
      var q = new NodeQueue(4, true);
      q.add(0, 0.5f);
      q.add(1, 0.3f);
      q.clear();
      assertThat(q.size()).isEqualTo(0);
      assertThat(q.isEmpty()).isTrue();
    }
  }

  @Nested
  @Tag("unit")
  class SearchResultTests {

    @Test
    void accessorsReturnCorrectValues() {
      var result = new SearchResult(new int[] {3, 1, 0}, new float[] {0.9f, 0.7f, 0.5f});
      assertThat(result.size()).isEqualTo(3);
      assertThat(result.nodeId(0)).isEqualTo(3);
      assertThat(result.score(0)).isEqualTo(0.9f);
    }
  }

  @Nested
  @Tag("unit")
  class InMemoryVectorsTests {

    @Test
    void sizeAndDimension() {
      float[][] data = {{1, 2, 3}, {4, 5, 6}};
      var vecs = new InMemoryVectors(data);
      assertThat(vecs.size()).isEqualTo(2);
      assertThat(vecs.dimension()).isEqualTo(3);
    }

    @Test
    void getVectorReturnsCorrectData() {
      float[][] data = {{1, 2, 3}, {4, 5, 6}};
      var vecs = new InMemoryVectors(data);
      assertThat(vecs.getVector(0)).containsExactly(1, 2, 3);
      assertThat(vecs.getVector(1)).containsExactly(4, 5, 6);
    }

    @Test
    void rejectsMismatchedDimensions() {
      float[][] data = {{1, 2, 3}, {4, 5}};
      assertThatThrownBy(() -> new InMemoryVectors(data))
          .isInstanceOf(IllegalArgumentException.class);
    }
  }
}
