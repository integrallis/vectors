package com.integrallis.vectors.ivf;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.vectors.core.SimilarityFunction;
import java.util.HashSet;
import java.util.Optional;
import java.util.Random;
import java.util.Set;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** P11: ClusterSplitter — Quake cost model, bisecting K-Means split. */
@Tag("unit")
class ClusterSplitterTest {

  private static final int DIM = 32;
  private static final SimilarityFunction METRIC = SimilarityFunction.EUCLIDEAN;

  private float[][] randomVecs(int n, int dim, long seed) {
    Random rng = new Random(seed);
    float[][] m = new float[n][dim];
    for (float[] row : m) for (int d = 0; d < dim; d++) row[d] = rng.nextFloat() * 2f - 1f;
    return m;
  }

  private ClusterPartition makePartition(int size, float[][] allVectors) {
    int[] ordinals = new int[size];
    for (int i = 0; i < size; i++) ordinals[i] = i;
    float[] centroid = new float[DIM];
    return ClusterPartition.of(0, centroid, ordinals);
  }

  // ─── cost model ──────────────────────────────────────────────────────────

  @Test
  void costIsProportionalToSize() {
    ClusterSplitter s = ClusterSplitter.forRootK(16);
    assertThat(s.cost(100)).isGreaterThan(s.cost(50));
    assertThat(s.cost(50)).isGreaterThan(s.cost(10));
    assertThat(s.cost(10)).isGreaterThan(s.cost(0));
    assertThat(s.cost(0)).isEqualTo(0L);
  }

  @Test
  void costIsAdditive() {
    ClusterSplitter s = ClusterSplitter.forRootK(16);
    long parent = s.cost(100);
    long left = s.cost(60);
    long right = s.cost(40);
    assertThat(left + right).isEqualTo(parent);
  }

  // ─── shouldSplit ──────────────────────────────────────────────────────────

  @Test
  void shouldSplit_aboveThreshold() {
    ClusterSplitter s = ClusterSplitter.forRootK(16);
    float[][] vecs = randomVecs(100, DIM, 1L);
    ClusterPartition big = makePartition(100, vecs);
    assertThat(s.shouldSplit(big)).isTrue();
  }

  @Test
  void shouldSplit_belowThreshold() {
    ClusterSplitter s = ClusterSplitter.forRootK(16);
    float[][] vecs = randomVecs(10, DIM, 2L);
    ClusterPartition small = makePartition(10, vecs);
    assertThat(s.shouldSplit(small)).isFalse();
  }

  @Test
  void shouldSplit_atExactThreshold() {
    int k = 16;
    ClusterSplitter s = ClusterSplitter.forRootK(k);
    float[][] vecs = randomVecs(k, DIM, 3L);
    ClusterPartition boundary = makePartition(k, vecs);
    // Quake: split when size >= minSplitSize = k
    assertThat(s.shouldSplit(boundary)).isTrue();
  }

  @Test
  void shouldSplit_singletonReturnsFalse() {
    ClusterSplitter s = ClusterSplitter.forRootK(4);
    float[][] vecs = randomVecs(1, DIM, 4L);
    ClusterPartition singleton = makePartition(1, vecs);
    assertThat(s.shouldSplit(singleton)).isFalse();
  }

  // ─── split ────────────────────────────────────────────────────────────────

  @Test
  void splitProducesTwoChildren() {
    float[][] allVecs = randomVecs(50, DIM, 5L);
    ClusterPartition parent = makePartition(50, allVecs);
    ClusterSplitter s = ClusterSplitter.forRootK(8);

    Optional<ClusterPartition[]> result = s.split(parent, allVecs, METRIC, 0L);

    assertThat(result).isPresent();
    assertThat(result.get()).hasSize(2);
  }

  @Test
  void splitChildrenCoverAllParentOrdinals() {
    float[][] allVecs = randomVecs(60, DIM, 6L);
    ClusterPartition parent = makePartition(60, allVecs);
    ClusterSplitter s = ClusterSplitter.forRootK(8);

    ClusterPartition[] children = s.split(parent, allVecs, METRIC, 0L).orElseThrow();
    Set<Integer> covered = new HashSet<>();
    for (int ord : children[0].ordinals()) covered.add(ord);
    for (int ord : children[1].ordinals()) covered.add(ord);

    Set<Integer> parentSet = new HashSet<>();
    for (int ord : parent.ordinals()) parentSet.add(ord);
    assertThat(covered).isEqualTo(parentSet);
  }

  @Test
  void splitChildrenAreDisjoint() {
    float[][] allVecs = randomVecs(40, DIM, 7L);
    ClusterPartition parent = makePartition(40, allVecs);
    ClusterSplitter s = ClusterSplitter.forRootK(8);

    ClusterPartition[] children = s.split(parent, allVecs, METRIC, 0L).orElseThrow();
    Set<Integer> left = new HashSet<>();
    for (int ord : children[0].ordinals()) left.add(ord);
    for (int ord : children[1].ordinals()) {
      assertThat(left).doesNotContain(ord);
    }
  }

  @Test
  void splitSingletonReturnsEmpty() {
    float[][] allVecs = randomVecs(1, DIM, 8L);
    ClusterPartition singleton = makePartition(1, allVecs);
    ClusterSplitter s = ClusterSplitter.forRootK(4);

    Optional<ClusterPartition[]> result = s.split(singleton, allVecs, METRIC, 0L);
    assertThat(result).isEmpty();
  }

  @Test
  void forRootKDefaultMinSplitSizeEqualsK() {
    int k = 32;
    ClusterSplitter s = ClusterSplitter.forRootK(k);
    assertThat(s.minSplitSize()).isEqualTo(k);
  }

  @Test
  void rejectsMinSplitSizeLessThanTwo() {
    assertThatThrownBy(() -> new ClusterSplitter(1, 30, 42L))
        .isInstanceOf(IllegalArgumentException.class);
  }
}
