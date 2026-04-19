package com.integrallis.vectors.vamana;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.SimilarityFunction;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

class RobustPrunerTest {

  @Nested
  @Tag("unit")
  class BasicPruning {

    @Test
    void emptyCandidates_returnsEmpty() {
      float[][] data = {{0, 0}, {1, 0}};
      var vectors = new InMemoryVectors(data);
      var candidates = new NeighborArray(5);
      var result = new NeighborArray(5);

      RobustPruner.robustPrune(
          0, candidates, 3, 1.0f, vectors, SimilarityFunction.EUCLIDEAN, result);

      assertThat(result.size()).isEqualTo(0);
    }

    @Test
    void singleCandidate_returnsIt() {
      float[][] data = {{0, 0}, {1, 0}};
      var vectors = new InMemoryVectors(data);
      var candidates = new NeighborArray(5);
      float score = SimilarityFunction.EUCLIDEAN.compare(data[0], data[1]);
      candidates.insert(1, score);
      var result = new NeighborArray(5);

      RobustPruner.robustPrune(
          0, candidates, 3, 1.0f, vectors, SimilarityFunction.EUCLIDEAN, result);

      assertThat(result.size()).isEqualTo(1);
      assertThat(result.node(0)).isEqualTo(1);
    }

    @Test
    void candidatesFitWithinMaxDegree_returnsAll() {
      // 3 candidates, maxDegree=5 → all should be retained
      float[][] data = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
      var vectors = new InMemoryVectors(data);
      var sim = SimilarityFunction.EUCLIDEAN;
      var candidates = new NeighborArray(5);
      candidates.insert(1, sim.compare(data[0], data[1]));
      candidates.insert(2, sim.compare(data[0], data[2]));
      candidates.insert(3, sim.compare(data[0], data[3]));
      var result = new NeighborArray(6);

      RobustPruner.robustPrune(0, candidates, 5, 1.0f, vectors, sim, result);

      assertThat(result.size()).isEqualTo(3);
    }

    @Test
    void alpha1_prunesBlockedCandidates() {
      // Three points on a line: base=0 at origin, 1 at (1,0), 2 at (2,0)
      // Point 2 is "covered" by point 1 since sim(2,1) > sim(0,2) * 1.0
      float[][] data = {{0, 0}, {1, 0}, {2, 0}};
      var vectors = new InMemoryVectors(data);
      var sim = SimilarityFunction.EUCLIDEAN;
      var candidates = new NeighborArray(5);
      candidates.insert(1, sim.compare(data[0], data[1]));
      candidates.insert(2, sim.compare(data[0], data[2]));
      var result = new NeighborArray(5);

      RobustPruner.robustPrune(0, candidates, 1, 1.0f, vectors, sim, result);

      // maxDegree=1 → only the closest (node 1) is selected
      assertThat(result.size()).isEqualTo(1);
      assertThat(result.node(0)).isEqualTo(1);
    }
  }

  @Nested
  @Tag("unit")
  class AlphaEffect {

    @Test
    void higherAlpha_retainsMoreCandidates() {
      // Points in a line: 0=(0,0), 1=(1,0), 2=(3,0), 3=(6,0)
      // At alpha=1.0, pruning is strict. At alpha=2.0, more candidates survive.
      float[][] data = {{0, 0}, {1, 0}, {3, 0}, {6, 0}};
      var vectors = new InMemoryVectors(data);
      var sim = SimilarityFunction.EUCLIDEAN;

      var candidates1 = new NeighborArray(5);
      candidates1.insert(1, sim.compare(data[0], data[1]));
      candidates1.insert(2, sim.compare(data[0], data[2]));
      candidates1.insert(3, sim.compare(data[0], data[3]));
      var result1 = new NeighborArray(5);
      RobustPruner.robustPrune(0, candidates1, 3, 1.0f, vectors, sim, result1);
      int sizeAlpha1 = result1.size();

      var candidates2 = new NeighborArray(5);
      candidates2.insert(1, sim.compare(data[0], data[1]));
      candidates2.insert(2, sim.compare(data[0], data[2]));
      candidates2.insert(3, sim.compare(data[0], data[3]));
      var result2 = new NeighborArray(5);
      RobustPruner.robustPrune(0, candidates2, 3, 2.0f, vectors, sim, result2);
      int sizeAlpha2 = result2.size();

      // Higher alpha should retain at least as many (likely more) candidates
      assertThat(sizeAlpha2).isGreaterThanOrEqualTo(sizeAlpha1);
    }

    @Test
    void alpha_greedy_selectsClosestFirst() {
      // The closest candidate should always be selected first regardless of alpha value
      float[][] data = {{0, 0}, {1, 0}, {0, 2}, {3, 3}};
      var vectors = new InMemoryVectors(data);
      var sim = SimilarityFunction.EUCLIDEAN;

      var candidates = new NeighborArray(5);
      candidates.insert(1, sim.compare(data[0], data[1]));
      candidates.insert(2, sim.compare(data[0], data[2]));
      candidates.insert(3, sim.compare(data[0], data[3]));
      var result = new NeighborArray(5);

      RobustPruner.robustPrune(0, candidates, 3, 1.5f, vectors, sim, result);

      // First selected (best score) should be the closest point
      assertThat(result.node(0)).isEqualTo(1);
    }
  }

  @Nested
  @Tag("unit")
  class EdgeCases {

    @Test
    void selfInCandidates_isRemoved() {
      float[][] data = {{0, 0}, {1, 0}};
      var vectors = new InMemoryVectors(data);
      var sim = SimilarityFunction.EUCLIDEAN;
      var candidates = new NeighborArray(5);
      candidates.insert(0, sim.compare(data[0], data[0])); // self
      candidates.insert(1, sim.compare(data[0], data[1]));
      var result = new NeighborArray(5);

      RobustPruner.robustPrune(0, candidates, 3, 1.0f, vectors, sim, result);

      assertThat(result.contains(0)).isFalse();
      assertThat(result.contains(1)).isTrue();
    }

    @Test
    void maxDegreeOfOne_selectsClosest() {
      float[][] data = {{0, 0}, {1, 0}, {0, 1}};
      var vectors = new InMemoryVectors(data);
      var sim = SimilarityFunction.EUCLIDEAN;
      var candidates = new NeighborArray(5);
      candidates.insert(1, sim.compare(data[0], data[1]));
      candidates.insert(2, sim.compare(data[0], data[2]));
      var result = new NeighborArray(5);

      RobustPruner.robustPrune(0, candidates, 1, 1.0f, vectors, sim, result);

      assertThat(result.size()).isEqualTo(1);
      // Both are equidistant, so first in sorted order is selected
      assertThat(result.node(0)).isIn(1, 2);
    }

    @Test
    void keepPruned_fillsRemainingSlots() {
      // Collinear: base at (0,0), node 1 at (1,0), node 2 at (2,0), node 3 at (3,0).
      // Similarities (EUCLIDEAN = 1/(1+dist)): node1=0.5, node2=0.333, node3=0.25
      //
      // Alpha=1.0 sweep:
      //   - node 1 selected (nothing in result yet)                → result=[1]
      //   - node 2: sim(2,1)=0.5 > score(2)*1.0=0.333 → covered, stays eligible
      //   - node 3: sim(3,1)=0.333 > score(3)*1.0=0.25 → covered, stays eligible
      //
      // Diversity sweep ends with result.size()=1 < maxDegree=2.
      // keepPruned fill: adds node 2 (first remaining eligible, best score).
      //   → result=[1, 2]
      float[][] data = {{0, 0}, {1, 0}, {2, 0}, {3, 0}};
      var vectors = new InMemoryVectors(data);
      var sim = SimilarityFunction.EUCLIDEAN;
      var candidates = new NeighborArray(5);
      candidates.insert(1, sim.compare(data[0], data[1]));
      candidates.insert(2, sim.compare(data[0], data[2]));
      candidates.insert(3, sim.compare(data[0], data[3]));
      var result = new NeighborArray(5);

      RobustPruner.robustPrune(0, candidates, 2, 1.0f, vectors, sim, result);

      assertThat(result.size()).isEqualTo(2);
      assertThat(result.node(0)).isEqualTo(1); // closest always selected by diversity
      assertThat(result.contains(2))
          .as("keepPruned should fill remaining slot with node 2")
          .isTrue();
    }
  }
}
