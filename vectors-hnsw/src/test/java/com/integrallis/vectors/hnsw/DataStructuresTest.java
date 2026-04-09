package com.integrallis.vectors.hnsw;

import static org.assertj.core.api.Assertions.*;

import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Tests for the core HNSW data structures: NeighborArray, NodeQueue, HnswGraph. */
class DataStructuresTest {

  @Nested
  @Tag("unit")
  class NeighborArrayTests {

    @Test
    void emptyArray_hasZeroSize() {
      var array = new NeighborArray(10);
      assertThat(array.size()).isZero();
      assertThat(array.maxSize()).isEqualTo(10);
    }

    @Test
    void insertSingleElement() {
      var array = new NeighborArray(10);
      assertThat(array.insert(42, 0.9f)).isTrue();
      assertThat(array.size()).isEqualTo(1);
      assertThat(array.node(0)).isEqualTo(42);
      assertThat(array.score(0)).isEqualTo(0.9f);
    }

    @Test
    void insertMaintainsDescendingScoreOrder() {
      var array = new NeighborArray(10);
      array.insert(1, 0.5f);
      array.insert(2, 0.9f);
      array.insert(3, 0.7f);

      // Descending: 0.9, 0.7, 0.5
      assertThat(array.size()).isEqualTo(3);
      assertThat(array.score(0)).isEqualTo(0.9f);
      assertThat(array.node(0)).isEqualTo(2);
      assertThat(array.score(1)).isEqualTo(0.7f);
      assertThat(array.node(1)).isEqualTo(3);
      assertThat(array.score(2)).isEqualTo(0.5f);
      assertThat(array.node(2)).isEqualTo(1);
    }

    @Test
    void insertWhenFull_evictsWorstScore() {
      var array = new NeighborArray(3);
      array.insert(1, 0.5f);
      array.insert(2, 0.7f);
      array.insert(3, 0.9f);

      // Insert better than worst (0.5) → evicts node 1
      assertThat(array.insert(4, 0.6f)).isTrue();
      assertThat(array.size()).isEqualTo(3);
      assertThat(array.contains(1)).isFalse();
      assertThat(array.contains(4)).isTrue();
    }

    @Test
    void insertWhenFull_rejectsWorseScore() {
      var array = new NeighborArray(3);
      array.insert(1, 0.5f);
      array.insert(2, 0.7f);
      array.insert(3, 0.9f);

      // Insert worse than worst (0.5) → rejected
      assertThat(array.insert(4, 0.3f)).isFalse();
      assertThat(array.size()).isEqualTo(3);
      assertThat(array.contains(4)).isFalse();
    }

    @Test
    void contains_returnsTrueForPresentNode() {
      var array = new NeighborArray(10);
      array.insert(42, 0.5f);
      assertThat(array.contains(42)).isTrue();
    }

    @Test
    void contains_returnsFalseForAbsentNode() {
      var array = new NeighborArray(10);
      array.insert(42, 0.5f);
      assertThat(array.contains(99)).isFalse();
    }

    @Test
    void clear_resetsSize() {
      var array = new NeighborArray(10);
      array.insert(1, 0.5f);
      array.insert(2, 0.7f);
      array.clear();
      assertThat(array.size()).isZero();
    }

    @Test
    void insertDuplicateNodeId_doesNotDuplicate() {
      var array = new NeighborArray(10);
      array.insert(42, 0.5f);
      assertThat(array.insert(42, 0.9f)).isFalse();
      assertThat(array.size()).isEqualTo(1);
      // Original score preserved
      assertThat(array.score(0)).isEqualTo(0.5f);
    }

    @Test
    void copyFrom_copiesAllEntries() {
      var src = new NeighborArray(10);
      src.insert(1, 0.9f);
      src.insert(2, 0.7f);
      src.insert(3, 0.5f);

      var dst = new NeighborArray(10);
      dst.copyFrom(src);

      assertThat(dst.size()).isEqualTo(3);
      assertThat(dst.node(0)).isEqualTo(src.node(0));
      assertThat(dst.score(0)).isEqualTo(src.score(0));
      assertThat(dst.node(2)).isEqualTo(src.node(2));
      assertThat(dst.score(2)).isEqualTo(src.score(2));
    }
  }

  @Nested
  @Tag("unit")
  class NodeQueueTests {

    @Test
    void minHeap_pollReturnsLowestScore() {
      var queue = new NodeQueue(4, true);
      queue.add(1, 0.9f);
      queue.add(2, 0.3f);
      queue.add(3, 0.7f);

      long top = queue.poll();
      assertThat(NodeQueue.score(top)).isEqualTo(0.3f);
      assertThat(NodeQueue.nodeId(top)).isEqualTo(2);
    }

    @Test
    void maxHeap_pollReturnsHighestScore() {
      var queue = new NodeQueue(4, false);
      queue.add(1, 0.3f);
      queue.add(2, 0.9f);
      queue.add(3, 0.7f);

      long top = queue.poll();
      assertThat(NodeQueue.score(top)).isEqualTo(0.9f);
      assertThat(NodeQueue.nodeId(top)).isEqualTo(2);
    }

    @Test
    void encodeDecode_roundTrip() {
      long encoded = NodeQueue.encode(12345, 0.75f);
      assertThat(NodeQueue.nodeId(encoded)).isEqualTo(12345);
      assertThat(NodeQueue.score(encoded)).isEqualTo(0.75f);
    }

    @Test
    void insertWithOverflow_maintainsBound() {
      var queue = new NodeQueue(4, true); // min-heap for results
      queue.add(1, 0.5f);
      queue.add(2, 0.7f);
      queue.add(3, 0.9f);

      // Add a better score when at bound=3 → evicts worst (0.9)
      assertThat(queue.insertWithOverflow(4, 0.6f, 3)).isTrue();
      assertThat(queue.size()).isEqualTo(3);
    }

    @Test
    void insertWithOverflow_rejectsWorseScore() {
      var queue = new NodeQueue(4, true); // min-heap for results
      queue.add(1, 0.5f);
      queue.add(2, 0.7f);
      queue.add(3, 0.9f);

      // For min-heap results: worst is the one with highest score (0.9).
      // But insertWithOverflow on min-heap means we keep lowest scores,
      // so adding 0.95 is worse than the worst in results → rejected.
      // Actually: for results min-heap, we want HIGHEST scores (best results).
      // The min-heap peek returns the worst result. If new score > worst: replace.
      // 0.95 > 0.5 (worst in min-heap) → replaces it
      assertThat(queue.insertWithOverflow(4, 0.95f, 3)).isTrue();
      // 0.3 < 0.7 (new worst) → rejected
      assertThat(queue.insertWithOverflow(5, 0.3f, 3)).isFalse();
    }

    @Test
    void growsBeyondInitialCapacity() {
      var queue = new NodeQueue(2, false);
      for (int i = 0; i < 100; i++) {
        queue.add(i, i * 0.01f);
      }
      assertThat(queue.size()).isEqualTo(100);
    }

    @Test
    void scoreOrdering_preservedForNonNegativeFloats() {
      // Verify that long encoding preserves ordering for non-negative floats
      long a = NodeQueue.encode(1, 0.3f);
      long b = NodeQueue.encode(2, 0.7f);
      long c = NodeQueue.encode(3, 0.9f);

      // For non-negative floats, higher float → higher long value
      assertThat(a).isLessThan(b);
      assertThat(b).isLessThan(c);
    }
  }

  @Nested
  @Tag("unit")
  class HnswGraphTests {

    @Test
    void initNode_setsLevel() {
      var graph = new HnswGraph(100, 16);
      graph.initNode(0, 2);
      assertThat(graph.nodeLevel(0)).isEqualTo(2);
    }

    @Test
    void initNode_allocatesNeighborArrays() {
      var graph = new HnswGraph(100, 16);
      graph.initNode(0, 2);

      // Layer 0: capacity = maxConnections0 + 1 = 33 (for overflow during backlink pruning)
      assertThat(graph.getNeighbors(0, 0)).isNotNull();
      assertThat(graph.getNeighbors(0, 0).maxSize()).isEqualTo(33);

      // Upper layers: capacity = maxConnections + 1 = 17
      assertThat(graph.getNeighbors(0, 1)).isNotNull();
      assertThat(graph.getNeighbors(0, 1).maxSize()).isEqualTo(17);
      assertThat(graph.getNeighbors(0, 2)).isNotNull();
      assertThat(graph.getNeighbors(0, 2).maxSize()).isEqualTo(17);
    }

    @Test
    void getNeighbors_layerAboveNodeLevel_returnsNull() {
      var graph = new HnswGraph(100, 16);
      graph.initNode(0, 1);
      assertThat(graph.getNeighbors(0, 2)).isNull();
    }

    @Test
    void getNeighbors_layer0_returnsNeighborArray() {
      var graph = new HnswGraph(100, 16);
      graph.initNode(0, 0);
      assertThat(graph.getNeighbors(0, 0)).isNotNull();
    }

    @Test
    void setEntryNode_updatesEntryAndMaxLevel() {
      var graph = new HnswGraph(100, 16);
      graph.initNode(5, 3);
      graph.setEntryNode(5, 3);
      assertThat(graph.entryNode()).isEqualTo(5);
      assertThat(graph.maxLevel()).isEqualTo(3);
    }

    @Test
    void maxConnections0_isTwiceM() {
      var graph = new HnswGraph(100, 16);
      assertThat(graph.maxConnections0()).isEqualTo(32);
    }

    @Test
    void upperLayers_expandDynamically() {
      var graph = new HnswGraph(100, 16);

      // First node at level 1
      graph.initNode(0, 1);
      assertThat(graph.getNeighbors(0, 1)).isNotNull();

      // Second node at level 3 → should expand upper layers
      graph.initNode(1, 3);
      assertThat(graph.getNeighbors(1, 3)).isNotNull();
      assertThat(graph.getNeighbors(1, 2)).isNotNull();
      assertThat(graph.getNeighbors(1, 1)).isNotNull();
    }
  }
}
