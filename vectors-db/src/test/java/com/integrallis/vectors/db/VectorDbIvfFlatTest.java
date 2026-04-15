package com.integrallis.vectors.db;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.filter.Filters;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * P18 gate: VectorCollection with {@link IndexType#IVF_FLAT} — recall, persistence, metadata
 * filters, and deletions.
 */
@Tag("unit")
class VectorDbIvfFlatTest {

  private static final int DIM = 32;

  private float[][] randomVecs(int n, int dim, long seed) {
    Random rng = new Random(seed);
    float[][] m = new float[n][dim];
    for (float[] row : m) for (int d = 0; d < dim; d++) row[d] = rng.nextFloat() * 2f - 1f;
    return m;
  }

  private VectorCollection buildIvf(int k, int nprobe, Path storage) {
    var builder =
        VectorCollection.builder()
            .dimension(DIM)
            .metric(SimilarityFunction.EUCLIDEAN)
            .indexType(IndexType.IVF_FLAT)
            .ivfK(k)
            .ivfNprobe(nprobe);
    if (storage != null) builder.storagePath(storage);
    return builder.build();
  }

  private int[] bruteForceTopK(float[] query, float[][] data, int k) {
    record P(int i, float d) {}
    P[] ps = new P[data.length];
    for (int i = 0; i < data.length; i++) {
      float d = 0f;
      for (int j = 0; j < query.length; j++) {
        float diff = query[j] - data[i][j];
        d += diff * diff;
      }
      ps[i] = new P(i, d);
    }
    Arrays.sort(ps, (a, b) -> Float.compare(a.d(), b.d()));
    int[] r = new int[k];
    for (int i = 0; i < k; i++) r[i] = ps[i].i();
    return r;
  }

  // ─── recall ──────────────────────────────────────────────────────────────

  @Test
  void buildAndSearch_recallExceedsThreshold() {
    float[][] data = randomVecs(500, DIM, 1L);
    try (var col = buildIvf(4, 3, null)) {
      for (int i = 0; i < data.length; i++) col.add(Document.of("doc-" + i, data[i]));
      col.commit();

      int queryCount = 30, k = 10;
      double totalRecall = 0.0;
      for (int q = 0; q < queryCount; q++) {
        float[] query = randomVecs(1, DIM, 1000L + q)[0];
        List<SearchResult.Hit> hits = col.search(SearchRequest.builder(query, k).build()).hits();
        Set<Integer> gtSet = new HashSet<>();
        for (int g : bruteForceTopK(query, data, k)) gtSet.add(g);
        long found =
            hits.stream()
                .filter(h -> gtSet.contains(Integer.parseInt(h.id().substring(4))))
                .count();
        totalRecall += (double) found / k;
      }
      assertThat(totalRecall / queryCount).isGreaterThanOrEqualTo(0.75);
    }
  }

  // ─── size ────────────────────────────────────────────────────────────────

  @Test
  void sizeReflectsAdds() {
    try (var col = buildIvf(4, 2, null)) {
      assertThat(col.size()).isZero();
      for (int i = 0; i < 50; i++) col.add(Document.of("doc-" + i, randomVecs(1, DIM, i)[0]));
      col.commit();
      assertThat(col.size()).isEqualTo(50);
    }
  }

  // ─── filter ──────────────────────────────────────────────────────────────

  @Test
  void search_appliesMetadataFilter() {
    float[][] data = randomVecs(60, DIM, 2L);
    try (var col = buildIvf(4, 4, null)) {
      for (int i = 0; i < data.length; i++) {
        String color = (i % 2 == 0) ? "red" : "blue";
        col.add(new Document("doc-" + i, data[i], null, Map.of("color", MetadataValue.of(color))));
      }
      col.commit();

      float[] query = data[0];
      List<SearchResult.Hit> hits =
          col.search(SearchRequest.builder(query, 10).filter(Filters.eq("color", "red")).build())
              .hits();
      assertThat(hits).isNotEmpty();
      // All returned hits must have color=red
      hits.forEach(
          h -> {
            int idx = Integer.parseInt(h.id().substring(4));
            assertThat(idx % 2).isZero();
          });
    }
  }

  // ─── delete ──────────────────────────────────────────────────────────────

  @Test
  void search_withDelete_excludesDeletedDoc() {
    float[][] data = randomVecs(50, DIM, 3L);
    // Put the needle as a very distinctive vector
    float[] needle = new float[DIM];
    Arrays.fill(needle, 100f);
    try (var col = buildIvf(4, 4, null)) {
      for (int i = 0; i < data.length; i++) col.add(Document.of("doc-" + i, data[i]));
      col.add(Document.of("needle", needle));
      col.commit();

      // Before delete: needle is top hit
      List<SearchResult.Hit> before = col.search(SearchRequest.builder(needle, 1).build()).hits();
      assertThat(before).isNotEmpty();
      assertThat(before.get(0).id()).isEqualTo("needle");

      // After delete + commit: needle must not appear
      assertThat(col.delete("needle")).isTrue();
      col.commit(); // tombstones are applied on commit, matching FLAT/HNSW/VAMANA semantics
      List<SearchResult.Hit> after = col.search(SearchRequest.builder(needle, 5).build()).hits();
      assertThat(after.stream().map(SearchResult.Hit::id)).doesNotContain("needle");
    }
  }

  // ─── persistence ─────────────────────────────────────────────────────────

  @Test
  void commit_thenReopen_preservesVectors(@TempDir Path tmp) {
    float[][] data = randomVecs(200, DIM, 4L);
    float[] needle = new float[DIM];
    Arrays.fill(needle, 50f);

    // Session 1: build, add, commit, close
    try (var col = buildIvf(4, 3, tmp)) {
      for (int i = 0; i < data.length; i++) col.add(Document.of("doc-" + i, data[i]));
      col.add(Document.of("needle", needle));
      col.commit();
    }

    // Session 2: reopen, search
    try (var col = buildIvf(4, 3, tmp)) {
      List<SearchResult.Hit> hits = col.search(SearchRequest.builder(needle, 1).build()).hits();
      assertThat(hits).isNotEmpty();
      assertThat(hits.get(0).id()).isEqualTo("needle");
    }
  }
}
