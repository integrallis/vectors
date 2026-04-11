package com.integrallis.vectors.db;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.SimilarityFunction;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * End-to-end acceptance tests for the Step 4b HNSW backend wired into {@link VectorCollection}.
 *
 * <p>Organized into nested classes that exercise the Step 4b scope, one slice at a time:
 *
 * <ul>
 *   <li>{@link InMemoryHnsw} — HNSW works in in-memory mode (no {@code storagePath}), auto-commit
 *       triggers a rebuild, and {@code add}/{@code addAll} flows identically to FLAT.
 *   <li>Persistent HNSW, crash recovery, and concurrency nested classes are added in Phases 5–6 of
 *       the Step 4b plan.
 * </ul>
 */
class VectorDbHnswPersistenceTest {

  private static final long SEED = 42L;
  private static final int DIM = 16;

  private static float[] randomVector(Random rng) {
    float[] v = new float[DIM];
    for (int i = 0; i < DIM; i++) {
      v[i] = rng.nextFloat();
    }
    return v;
  }

  private static Document docWithMetadata(String id, float[] vector, int i) {
    Map<String, MetadataValue> md = new HashMap<>();
    md.put("idx", MetadataValue.of(Long.valueOf(i)));
    md.put("name", MetadataValue.of("doc-" + i));
    md.put("flag", MetadataValue.of((i % 2) == 0));
    return new Document(id, vector, "text-" + i, md);
  }

  private static List<Document> generateDocs(int count, long seed) {
    Random rng = new Random(seed);
    List<Document> out = new ArrayList<>(count);
    for (int i = 0; i < count; i++) {
      out.add(docWithMetadata("doc-" + i, randomVector(rng), i));
    }
    return out;
  }

  /**
   * Computes the top-{@code k} brute-force neighbors of {@code query} over {@code docs} using the
   * {@link SimilarityFunction#EUCLIDEAN} score {@code 1 / (1 + d²)}. Used as the recall reference
   * for HNSW-backed searches in this module.
   */
  private static Set<String> bruteForceTopKIds(List<Document> docs, float[] query, int k) {
    SimilarityFunction sim = SimilarityFunction.EUCLIDEAN;
    // Parallel arrays of (id, score).
    String[] ids = new String[docs.size()];
    float[] scores = new float[docs.size()];
    for (int i = 0; i < docs.size(); i++) {
      ids[i] = docs.get(i).id();
      scores[i] = sim.compare(query, docs.get(i).vector());
    }
    // Simple selection sort over the first k — sufficient for tests.
    Set<String> top = new HashSet<>();
    boolean[] used = new boolean[docs.size()];
    for (int r = 0; r < Math.min(k, docs.size()); r++) {
      int best = -1;
      for (int i = 0; i < docs.size(); i++) {
        if (used[i]) {
          continue;
        }
        if (best == -1 || scores[i] > scores[best]) {
          best = i;
        }
      }
      used[best] = true;
      top.add(ids[best]);
    }
    return top;
  }

  @Nested
  @Tag("unit")
  class InMemoryHnsw {

    @Test
    void inMemoryHnswSearchReturnsAccurateNeighbors() {
      // BEHAVIOR assertion: the HNSW-backed collection returns results that agree with
      // brute-force ground truth to within HNSW's recall tolerance. A constant-scoring stub
      // would pass a shape-only test (k=10 hits) but fail this recall assertion.
      List<Document> docs = generateDocs(200, SEED);
      float[] query = randomVector(new Random(SEED + 1));

      try (var col =
          VectorCollection.builder()
              .dimension(DIM)
              .metric(SimilarityFunction.EUCLIDEAN)
              .indexType(IndexType.HNSW)
              .hnswM(16)
              .hnswEfConstruction(100)
              .build()) {
        col.addAll(docs);
        col.commit();
        assertThat(col.size()).isEqualTo(200);

        SearchResult result = col.search(SearchRequest.builder(query, 10).build());
        assertThat(result.hits()).hasSize(10);
        // Hits must be sorted by score descending.
        for (int i = 1; i < result.hits().size(); i++) {
          assertThat(result.hits().get(i).score())
              .isLessThanOrEqualTo(result.hits().get(i - 1).score());
        }

        // Recall ≥ 0.8 against brute force (200 × 16-dim HNSW clears this consistently).
        Set<String> expectedTopK = bruteForceTopKIds(docs, query, 10);
        long hits = result.hits().stream().filter(h -> expectedTopK.contains(h.id())).count();
        assertThat((double) hits / 10).isGreaterThanOrEqualTo(0.8);
      }
    }

    @Test
    void inMemoryHnswDefaultBuildParametersWork() {
      // No explicit hnswM / hnswEfConstruction — builder defaults (M=16, efConstruction=200).
      List<Document> docs = generateDocs(100, SEED);
      float[] query = randomVector(new Random(SEED + 1));

      try (var col =
          VectorCollection.builder()
              .dimension(DIM)
              .metric(SimilarityFunction.EUCLIDEAN)
              .indexType(IndexType.HNSW)
              .build()) {
        col.addAll(docs);
        col.commit();
        assertThat(col.config().hnswParams()).isNotNull();
        assertThat(col.config().hnswParams().m()).isEqualTo(VectorCollectionBuilder.DEFAULT_HNSW_M);
        assertThat(col.config().hnswParams().efConstruction())
            .isEqualTo(VectorCollectionBuilder.DEFAULT_HNSW_EF_CONSTRUCTION);

        SearchResult result = col.search(SearchRequest.builder(query, 5).build());
        assertThat(result.hits()).hasSize(5);
      }
    }

    @Test
    void inMemoryHnswCustomBuildParametersFlowThrough() {
      // Verify the builder parameters are captured in the config and can be read back.
      try (var col =
          VectorCollection.builder()
              .dimension(DIM)
              .metric(SimilarityFunction.EUCLIDEAN)
              .indexType(IndexType.HNSW)
              .hnswM(8)
              .hnswEfConstruction(50)
              .build()) {
        assertThat(col.config().hnswParams().m()).isEqualTo(8);
        assertThat(col.config().hnswParams().efConstruction()).isEqualTo(50);
      }
    }

    @Test
    void inMemoryHnswAutoCommitTriggersRebuild() {
      // autoCommitThreshold=50 with single-doc add()s: after the 50th add, maybeAutoCommit fires
      // and rebuilds the HNSW index with the first 50 docs. Docs 51-75 then sit in the staging
      // buffer (size=25, below the threshold) until an explicit commit() flushes them.
      //
      // This verifies TWO things about the auto-commit rebuild path:
      //   1. Mid-stream auto-commit actually rebuilds the HNSW graph — docs visible via search.
      //   2. Staged docs are NOT searchable until commit() flushes them — the HNSW SPI only sees
      //      committed vectors, confirming that the adapter rebuild is what makes data reachable.
      try (var col =
          VectorCollection.builder()
              .dimension(DIM)
              .metric(SimilarityFunction.EUCLIDEAN)
              .indexType(IndexType.HNSW)
              .autoCommitThreshold(50)
              .build()) {
        List<Document> docs = generateDocs(75, SEED);
        // Single-doc adds so auto-commit fires at exactly the 50th doc. addAll(75) would stage
        // all 75 and trigger a single auto-commit for the whole batch, which is a different
        // code path and does not exercise the mid-stream rebuild semantics.
        for (Document d : docs) {
          col.add(d);
        }

        // Mid-stream state: 50 committed, 25 staged. size() reports committed liveCount only.
        assertThat(col.size()).isEqualTo(50);

        // A committed doc (doc-25) queried by its own vector MUST appear in the top-k — this
        // proves the HNSW adapter was rebuilt with the first 50 docs, not left empty.
        Document committedDoc = docs.get(25);
        SearchResult committedResult =
            col.search(SearchRequest.builder(committedDoc.vector(), 5).build());
        assertThat(committedResult.hits())
            .extracting(SearchResult.Hit::id)
            .contains(committedDoc.id());

        // A staged doc (doc-70) queried by its own vector MUST NOT appear — the index only
        // holds the 50 committed docs, so doc-70 is unreachable via search().
        Document stagedDoc = docs.get(70);
        SearchResult preCommitStaged =
            col.search(SearchRequest.builder(stagedDoc.vector(), 10).build());
        assertThat(preCommitStaged.hits())
            .extracting(SearchResult.Hit::id)
            .doesNotContain(stagedDoc.id());
        // get() by id also cannot find staged docs before commit.
        assertThat(col.get(stagedDoc.id())).isNull();

        // After explicit commit: all 75 are committed and reachable. The staged doc now appears
        // in search results when queried by its own vector.
        col.commit();
        assertThat(col.size()).isEqualTo(75);

        SearchResult postCommitStaged =
            col.search(SearchRequest.builder(stagedDoc.vector(), 5).build());
        assertThat(postCommitStaged.hits())
            .extracting(SearchResult.Hit::id)
            .contains(stagedDoc.id());
        assertThat(col.get(stagedDoc.id())).isNotNull();
      }
    }

    @Test
    void inMemoryHnswRebuildsAcrossCommits() {
      // Two sequential commits with different data must not corrupt each other. After the
      // second commit, the first batch's ids should still be present (the adapter rebuilds from
      // the full doc set, not just the new batch).
      try (var col =
          VectorCollection.builder()
              .dimension(DIM)
              .metric(SimilarityFunction.EUCLIDEAN)
              .indexType(IndexType.HNSW)
              .build()) {
        List<Document> batch1 = generateDocs(50, SEED);
        col.addAll(batch1);
        col.commit();
        assertThat(col.size()).isEqualTo(50);

        // Generate a second batch with distinct ids.
        List<Document> batch2 = new ArrayList<>();
        Random rng = new Random(SEED + 100);
        for (int i = 50; i < 100; i++) {
          batch2.add(docWithMetadata("doc-" + i, randomVector(rng), i));
        }
        col.addAll(batch2);
        col.commit();
        assertThat(col.size()).isEqualTo(100);

        // All 100 ids must be retrievable via get().
        for (int i = 0; i < 100; i++) {
          assertThat(col.get("doc-" + i)).as("doc-%d", i).isNotNull();
        }
      }
    }

    @Test
    void inMemoryHnswEmptyCollectionReportsZero() {
      // HnswIndex rejects empty input; HnswIndexAdapter's empty-state branch handles this.
      try (var col =
          VectorCollection.builder()
              .dimension(DIM)
              .metric(SimilarityFunction.EUCLIDEAN)
              .indexType(IndexType.HNSW)
              .build()) {
        assertThat(col.size()).isZero();
        SearchResult result =
            col.search(SearchRequest.builder(randomVector(new Random(SEED)), 10).build());
        assertThat(result.hits()).isEmpty();
      }
    }
  }
}
