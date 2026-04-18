package com.integrallis.vectors.db;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.cache.QvCache;
import java.util.Arrays;
import java.util.Map;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Gate tests for QVCache (OM5): semantic query-result cache with LRU eviction and commit-triggered
 * invalidation.
 */
class VectorDbQvCacheTest {

  private static final int DIM = 4;

  private static VectorCollection newCachedCollection(int cacheSize) {
    return VectorCollection.builder()
        .dimension(DIM)
        .metric(SimilarityFunction.EUCLIDEAN)
        .indexType(IndexType.FLAT)
        .cacheSize(cacheSize)
        .build();
  }

  private static void populate(VectorCollection col, int n) {
    for (int i = 0; i < n; i++) {
      float v = i;
      col.add(new Document("d" + i, new float[] {v, v, v, v}, null, Map.of()));
    }
    col.commit();
  }

  @Nested
  @Tag("unit")
  class QvCacheIntegrationTests {

    @Test
    void cacheHit_returnsSameResultObject() {
      try (VectorCollection col = newCachedCollection(256)) {
        populate(col, 20);
        float[] q = {1f, 1f, 1f, 1f};
        // includeVector(false) makes the request cache-eligible
        SearchRequest req = SearchRequest.builder(q, 5).includeVector(false).build();

        SearchResult r1 = col.search(req);
        SearchResult r2 = col.search(req);

        // Second call must return the same cached object
        assertThat(r2).isSameAs(r1);
      }
    }

    @Test
    void cacheMiss_afterDifferentQuery_executesSearch() {
      try (VectorCollection col = newCachedCollection(256)) {
        populate(col, 20);

        // Use queries with different SHAPES (not just different magnitudes) so the int8
        // max-abs normalisation produces distinct cache keys.
        SearchResult r1 =
            col.search(
                SearchRequest.builder(new float[] {1f, 0f, 0f, 0f}, 5)
                    .includeVector(false)
                    .build());
        SearchResult r2 =
            col.search(
                SearchRequest.builder(new float[] {0f, 0f, 0f, 1f}, 5)
                    .includeVector(false)
                    .build());

        // Different direction → different cache entries → different result objects
        assertThat(r2).isNotSameAs(r1);
      }
    }

    @Test
    void cacheInvalidation_onCommit_causesSearchToReExecute() {
      try (VectorCollection col = newCachedCollection(256)) {
        populate(col, 10);
        float[] q = {99f, 99f, 99f, 99f};
        SearchRequest req = SearchRequest.builder(q, 3).includeVector(false).build();

        SearchResult before = col.search(req);

        // Add a document near the query point and commit — should invalidate the cache
        col.add(new Document("near", new float[] {99f, 99f, 99f, 99f}, null, Map.of()));
        col.commit();

        SearchResult after = col.search(req);
        // After commit the new nearest doc should appear
        assertThat(after.hits().stream().anyMatch(h -> h.id().equals("near"))).isTrue();
        // The result object must differ (cache was invalidated)
        assertThat(after).isNotSameAs(before);
      }
    }

    @Test
    void disabledCache_isTransparent_noCaching() {
      try (VectorCollection col = newCachedCollection(0)) {
        populate(col, 10);
        float[] q = {1f, 1f, 1f, 1f};
        SearchRequest req = SearchRequest.builder(q, 3).includeVector(false).build();

        SearchResult r1 = col.search(req);
        SearchResult r2 = col.search(req);

        // Without a cache two independent search calls return equal content but different objects
        assertThat(r2).isNotSameAs(r1);
        assertThat(r2.hits()).hasSize(r1.hits().size());
      }
    }

    @Test
    void cacheStats_hitAndMissCounts() {
      try (VectorCollectionImpl col = (VectorCollectionImpl) newCachedCollection(256)) {
        populate(col, 10);
        float[] q = {1f, 1f, 1f, 1f};
        SearchRequest req = SearchRequest.builder(q, 3).includeVector(false).build();

        col.search(req); // miss
        col.search(req); // hit
        col.search(req); // hit

        QvCache.CacheStats stats = col.queryCache().stats();
        assertThat(stats.misses()).isEqualTo(1);
        assertThat(stats.hits()).isEqualTo(2);
        assertThat(stats.hitRatio())
            .isCloseTo(2.0 / 3.0, org.assertj.core.api.Assertions.offset(0.01));
      }
    }

    @Test
    void lruEviction_whenCapacityExceeded() {
      int capacity = 3;
      try (VectorCollectionImpl col = (VectorCollectionImpl) newCachedCollection(capacity)) {
        populate(col, 20);
        // Fill the cache with 3 distinct-direction queries (different int8 keys).
        float[][] queries = {{1f, 0f, 0f, 0f}, {0f, 1f, 0f, 0f}, {0f, 0f, 1f, 0f}};
        for (float[] q : queries) {
          col.search(SearchRequest.builder(q, 3).includeVector(false).build());
        }
        assertThat(col.queryCache().stats().size()).isEqualTo(capacity);

        // One more unique query should evict the oldest
        col.search(
            SearchRequest.builder(new float[] {99f, 99f, 99f, 99f}, 3)
                .includeVector(false)
                .build());
        assertThat(col.queryCache().stats().size()).isEqualTo(capacity);
        assertThat(col.queryCache().stats().evictions()).isGreaterThanOrEqualTo(1);
      }
    }

    @Test
    void includeVector_requestsAreNotCached() {
      try (VectorCollectionImpl col = (VectorCollectionImpl) newCachedCollection(256)) {
        populate(col, 10);
        float[] q = {1f, 1f, 1f, 1f};
        SearchRequest req = SearchRequest.builder(q, 3).includeVector(true).build();

        col.search(req); // should NOT cache
        col.search(req); // should NOT be served from cache

        // Hits should be 0 — includeVector=true is never cached
        assertThat(col.queryCache().stats().hits()).isEqualTo(0);
        assertThat(col.queryCache().stats().size()).isEqualTo(0);
      }
    }
  }

  @Nested
  @Tag("unit")
  class QvCacheUnitTests {

    @Test
    void quantize_similarVectors_produceSameKey() {
      float[] v1 = {1.001f, 2.001f, -3.001f, 4.001f};
      float[] v2 = {1.002f, 2.002f, -3.002f, 4.002f}; // within int8 quantization tolerance
      byte[] q1 = QvCache.quantize(v1);
      byte[] q2 = QvCache.quantize(v2);
      assertThat(Arrays.equals(q1, q2)).isTrue();
    }

    @Test
    void quantize_oppositeVectors_produceDifferentKeys() {
      float[] v1 = {1f, 0f, 0f, 0f};
      float[] v2 = {-1f, 0f, 0f, 0f};
      assertThat(Arrays.equals(QvCache.quantize(v1), QvCache.quantize(v2))).isFalse();
    }
  }
}
