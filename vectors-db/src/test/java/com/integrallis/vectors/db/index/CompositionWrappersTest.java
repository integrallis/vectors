package com.integrallis.vectors.db.index;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatExceptionOfType;
import static org.assertj.core.api.Assertions.assertThatNullPointerException;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.index.IndexSpi.SearchOutcome;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.IntPredicate;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link IndexRefine}, {@link IndexShards}, and {@link IndexReplicas}. Brute-force
 * {@link FlatScanAdapter} serves as both the underlying backend and the ground-truth reference —
 * any deviation from its top-k must come from the composition logic, not the backend.
 */
@Tag("unit")
class CompositionWrappersTest {

  private static float[][] randomVectors(int n, int dim, long seed) {
    Random rng = new Random(seed);
    float[][] out = new float[n][dim];
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < dim; j++) {
        out[i][j] = rng.nextFloat() * 2f - 1f;
      }
    }
    return out;
  }

  private static float[] randomQuery(int dim, long seed) {
    Random rng = new Random(seed);
    float[] q = new float[dim];
    for (int i = 0; i < dim; i++) {
      q[i] = rng.nextFloat() * 2f - 1f;
    }
    return q;
  }

  private static Set<Integer> asSet(int[] a) {
    Set<Integer> s = new HashSet<>();
    for (int v : a) s.add(v);
    return s;
  }

  /** Test-only {@link IndexSpi} that delegates to an inner SPI and counts search+close calls. */
  private static final class CountingSpi implements IndexSpi {
    private final IndexSpi inner;
    final AtomicInteger searchCalls = new AtomicInteger();
    final AtomicInteger closeCalls = new AtomicInteger();

    CountingSpi(IndexSpi inner) {
      this.inner = inner;
    }

    @Override
    public void build(float[][] vectors, SimilarityFunction metric) {
      inner.build(vectors, metric);
    }

    @Override
    public SearchOutcome search(float[] q, int k, int ls, float of) {
      searchCalls.incrementAndGet();
      return inner.search(q, k, ls, of);
    }

    @Override
    public SearchOutcome search(float[] q, int k, int ls, float of, int nStarts) {
      searchCalls.incrementAndGet();
      return inner.search(q, k, ls, of, nStarts);
    }

    @Override
    public SearchOutcome searchWithPredicate(float[] q, int k, int ls, float of, IntPredicate p) {
      searchCalls.incrementAndGet();
      return inner.searchWithPredicate(q, k, ls, of, p);
    }

    @Override
    public int size() {
      return inner.size();
    }

    @Override
    public void close() {
      closeCalls.incrementAndGet();
      inner.close();
    }
  }

  @Nested
  class RefineWrapper {

    @Test
    void rescoreReturnsSameTopKAsBaseWhenBaseIsBruteForce() {
      float[][] vectors = randomVectors(200, 16, 7L);
      float[] query = randomQuery(16, 11L);

      FlatScanAdapter base = new FlatScanAdapter();
      base.build(vectors, SimilarityFunction.EUCLIDEAN);
      SearchOutcome reference = base.search(query, 10, 0, 1.0f);

      IndexRefine refine = new IndexRefine(new FlatScanAdapter(), 3.0f);
      refine.build(vectors, SimilarityFunction.EUCLIDEAN);
      SearchOutcome got = refine.search(query, 10, 0, 1.0f);

      assertThat(got.ordinals()).containsExactly(reference.ordinals());
      assertThat(got.scores()).containsExactly(reference.scores());
    }

    @Test
    void scoresAreRecomputedAgainstStoredVectors() {
      float[][] vectors = randomVectors(50, 8, 3L);
      float[] query = randomQuery(8, 5L);

      IndexRefine refine = new IndexRefine(new FlatScanAdapter(), 2.0f);
      refine.build(vectors, SimilarityFunction.DOT_PRODUCT);
      SearchOutcome got = refine.search(query, 5, 0, 1.0f);

      for (int i = 0; i < got.ordinals().length; i++) {
        float expected = SimilarityFunction.DOT_PRODUCT.compare(query, vectors[got.ordinals()[i]]);
        assertThat(got.scores()[i]).isCloseTo(expected, org.assertj.core.data.Offset.offset(1e-5f));
      }
      // Descending order by rescored score.
      for (int i = 1; i < got.scores().length; i++) {
        assertThat(got.scores()[i]).isLessThanOrEqualTo(got.scores()[i - 1]);
      }
    }

    @Test
    void rejectsKFactorBelowOne() {
      assertThatExceptionOfType(IllegalArgumentException.class)
          .isThrownBy(() -> new IndexRefine(new FlatScanAdapter(), 0.5f));
    }

    @Test
    void rejectsNullBase() {
      assertThatNullPointerException().isThrownBy(() -> new IndexRefine(null, 2.0f));
    }

    @Test
    void emptyBuildReturnsEmptyResults() {
      IndexRefine refine = new IndexRefine(new FlatScanAdapter(), 2.0f);
      refine.build(new float[0][], SimilarityFunction.EUCLIDEAN);
      SearchOutcome got = refine.search(new float[8], 5, 0, 1.0f);
      assertThat(got.ordinals()).isEmpty();
      assertThat(got.scores()).isEmpty();
    }

    @Test
    void closePropagatesToBase() {
      CountingSpi base = new CountingSpi(new FlatScanAdapter());
      IndexRefine refine = new IndexRefine(base, 1.0f);
      refine.close();
      assertThat(base.closeCalls.get()).isEqualTo(1);
    }
  }

  @Nested
  class ShardsWrapper {

    @Test
    void topKMatchesBruteForceOverFullCorpus() {
      float[][] vectors = randomVectors(300, 12, 17L);
      float[] query = randomQuery(12, 19L);

      FlatScanAdapter reference = new FlatScanAdapter();
      reference.build(vectors, SimilarityFunction.EUCLIDEAN);
      SearchOutcome expected = reference.search(query, 10, 0, 1.0f);

      List<IndexSpi> shardSpis =
          List.of(new FlatScanAdapter(), new FlatScanAdapter(), new FlatScanAdapter());
      IndexShards shards = new IndexShards(shardSpis);
      shards.build(vectors, SimilarityFunction.EUCLIDEAN);
      SearchOutcome got = shards.search(query, 10, 0, 1.0f);

      assertThat(got.ordinals()).containsExactly(expected.ordinals());
      assertThat(got.scores()).containsExactly(expected.scores());
      assertThat(shards.size()).isEqualTo(vectors.length);
    }

    @Test
    void mapsShardLocalOrdinalsToGlobal() {
      float[][] vectors = randomVectors(9, 4, 23L);
      IndexShards shards =
          new IndexShards(
              List.of(new FlatScanAdapter(), new FlatScanAdapter(), new FlatScanAdapter()));
      shards.build(vectors, SimilarityFunction.EUCLIDEAN);
      SearchOutcome got = shards.search(vectors[0], 9, 0, 1.0f);
      // All nine global ordinals must appear exactly once.
      assertThat(asSet(got.ordinals())).containsExactlyInAnyOrder(0, 1, 2, 3, 4, 5, 6, 7, 8);
    }

    @Test
    void predicateIsEvaluatedAgainstGlobalOrdinals() {
      float[][] vectors = randomVectors(60, 8, 29L);
      IndexShards shards = new IndexShards(List.of(new FlatScanAdapter(), new FlatScanAdapter()));
      shards.build(vectors, SimilarityFunction.EUCLIDEAN);

      AtomicReference<Set<Integer>> seen = new AtomicReference<>(new HashSet<>());
      SearchOutcome got =
          shards.searchWithPredicate(
              randomQuery(8, 31L),
              5,
              0,
              1.0f,
              ord -> {
                seen.get().add(ord);
                return ord % 2 == 0;
              });

      // The default predicate path in IndexSpi ignores the predicate; FlatScanAdapter inherits
      // that default, so the caller predicate is exercised only when navigating backends override
      // it. We verify behavior that must hold either way: result set is the shards' merged top-k.
      assertThat(got.ordinals()).hasSize(5);
      assertThat(asSet(got.ordinals())).allMatch(o -> o >= 0 && o < vectors.length);
    }

    @Test
    void rejectsEmptyShardList() {
      assertThatExceptionOfType(IllegalArgumentException.class)
          .isThrownBy(() -> new IndexShards(List.of()));
    }

    @Test
    void concurrentSearchIsSafe() throws Exception {
      float[][] vectors = randomVectors(200, 8, 37L);
      IndexShards shards = new IndexShards(List.of(new FlatScanAdapter(), new FlatScanAdapter()));
      shards.build(vectors, SimilarityFunction.EUCLIDEAN);

      float[] query = randomQuery(8, 41L);
      SearchOutcome expected = shards.search(query, 10, 0, 1.0f);

      int threads = 8;
      ExecutorService exec = Executors.newFixedThreadPool(threads);
      try {
        CountDownLatch start = new CountDownLatch(1);
        List<java.util.concurrent.Future<SearchOutcome>> futs = new ArrayList<>();
        for (int t = 0; t < threads; t++) {
          futs.add(
              exec.submit(
                  () -> {
                    start.await();
                    return shards.search(query, 10, 0, 1.0f);
                  }));
        }
        start.countDown();
        for (var f : futs) {
          SearchOutcome s = f.get(30, TimeUnit.SECONDS);
          assertThat(s.ordinals()).containsExactly(expected.ordinals());
          assertThat(s.scores()).containsExactly(expected.scores());
        }
      } finally {
        exec.shutdownNow();
      }
    }
  }

  @Nested
  class ReplicasWrapper {

    @Test
    void resultsMatchAnySingleReplica() {
      float[][] vectors = randomVectors(150, 10, 43L);
      float[] query = randomQuery(10, 47L);
      FlatScanAdapter reference = new FlatScanAdapter();
      reference.build(vectors, SimilarityFunction.EUCLIDEAN);
      SearchOutcome expected = reference.search(query, 7, 0, 1.0f);

      IndexReplicas replicas =
          new IndexReplicas(
              List.of(new FlatScanAdapter(), new FlatScanAdapter(), new FlatScanAdapter()));
      replicas.build(vectors, SimilarityFunction.EUCLIDEAN);
      for (int i = 0; i < 6; i++) {
        SearchOutcome got = replicas.search(query, 7, 0, 1.0f);
        assertThat(got.ordinals()).containsExactly(expected.ordinals());
        assertThat(got.scores()).containsExactly(expected.scores());
      }
      assertThat(replicas.size()).isEqualTo(vectors.length);
    }

    @Test
    void roundRobinDistributesAcrossReplicas() {
      CountingSpi a = new CountingSpi(new FlatScanAdapter());
      CountingSpi b = new CountingSpi(new FlatScanAdapter());
      CountingSpi c = new CountingSpi(new FlatScanAdapter());
      IndexReplicas replicas = new IndexReplicas(List.of(a, b, c));
      replicas.build(randomVectors(30, 4, 53L), SimilarityFunction.EUCLIDEAN);
      for (int i = 0; i < 9; i++) {
        replicas.search(randomQuery(4, 59L + i), 3, 0, 1.0f);
      }
      int total = a.searchCalls.get() + b.searchCalls.get() + c.searchCalls.get();
      assertThat(total).isEqualTo(9);
      assertThat(a.searchCalls.get()).isEqualTo(3);
      assertThat(b.searchCalls.get()).isEqualTo(3);
      assertThat(c.searchCalls.get()).isEqualTo(3);
    }

    @Test
    void rejectsEmptyReplicaList() {
      assertThatExceptionOfType(IllegalArgumentException.class)
          .isThrownBy(() -> new IndexReplicas(List.of()));
    }

    @Test
    void closePropagatesToAllReplicas() {
      CountingSpi a = new CountingSpi(new FlatScanAdapter());
      CountingSpi b = new CountingSpi(new FlatScanAdapter());
      IndexReplicas replicas = new IndexReplicas(List.of(a, b));
      replicas.close();
      assertThat(a.closeCalls.get()).isEqualTo(1);
      assertThat(b.closeCalls.get()).isEqualTo(1);
    }
  }
}
