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
package com.integrallis.vectors.db;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatIllegalArgumentException;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.core.filter.Filters;
import java.nio.file.Path;
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
import org.junit.jupiter.api.io.TempDir;

/**
 * End-to-end tests for the {@link SearchRequest#searchMultiStart()} pass-through through {@link
 * VectorCollection}: builder defaults, validation, recall parity on in-memory and persistent HNSW
 * collections, concurrency with {@link VectorCollection#searchBatch}, and graceful no-op fallback
 * on non-HNSW backends and the ACORN pre-filter path.
 */
class VectorDbMultiStartTest {

  private static final int DIM = 16;
  private static final long SEED = 42L;

  private static float[] randomVector(Random rng) {
    float[] v = new float[DIM];
    for (int i = 0; i < DIM; i++) v[i] = rng.nextFloat();
    return v;
  }

  private static List<Document> generateDocs(int count, long seed) {
    Random rng = new Random(seed);
    List<Document> out = new ArrayList<>(count);
    for (int i = 0; i < count; i++) {
      Map<String, MetadataValue> md = new HashMap<>();
      md.put("color", MetadataValue.of((i % 2 == 0) ? "red" : "blue"));
      out.add(new Document("doc-" + i, randomVector(rng), "text-" + i, md));
    }
    return out;
  }

  @Nested
  @Tag("unit")
  class RequestDefaults {

    @Test
    void searchRequestSearchMultiStartDefaultsToOne() {
      SearchRequest req = SearchRequest.builder(new float[DIM], 10).build();
      assertThat(req.searchMultiStart()).isEqualTo(1);
    }

    @Test
    void searchMultiStartNegativeRejected() {
      // Builder-level validation.
      assertThatIllegalArgumentException()
          .isThrownBy(() -> SearchRequest.builder(new float[DIM], 10).searchMultiStart(0));
      assertThatIllegalArgumentException()
          .isThrownBy(() -> SearchRequest.builder(new float[DIM], 10).searchMultiStart(-3));
      // Compact-constructor validation (builder bypass).
      assertThatIllegalArgumentException()
          .isThrownBy(
              () ->
                  new SearchRequest(
                      new float[DIM],
                      10,
                      100,
                      4.0f,
                      4.0f,
                      -Float.MAX_VALUE,
                      null,
                      true,
                      true,
                      true,
                      0));
    }
  }

  @Nested
  @Tag("unit")
  class InMemoryHnswMultiStart {

    @Test
    void inMemoryHnswMultiStartReturnsTopK() {
      List<Document> docs = generateDocs(500, SEED);
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

        SearchResult r = col.search(SearchRequest.builder(query, 10).searchMultiStart(4).build());
        assertThat(r.hits()).hasSize(10);
      }
    }

    @Test
    void inMemoryHnswMultiStartRecallParity() {
      List<Document> docs = generateDocs(500, SEED);
      float[] query = randomVector(new Random(SEED + 3));

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

        SearchResult single = col.search(SearchRequest.builder(query, 10).build());
        SearchResult multi =
            col.search(SearchRequest.builder(query, 10).searchMultiStart(4).build());

        Set<String> singleIds = new HashSet<>();
        for (SearchResult.Hit h : single.hits()) singleIds.add(h.id());
        long overlap = multi.hits().stream().filter(h -> singleIds.contains(h.id())).count();
        assertThat(overlap / 10.0)
            .as("multi-start top-10 recall vs single-start top-10 (overlap=%d)", overlap)
            .isGreaterThanOrEqualTo(0.90);
      }
    }
  }

  @Nested
  @Tag("unit")
  class MappedHnswMultiStart {

    @Test
    void mappedHnswMultiStartSmoke(@TempDir Path tempDir) {
      Path storageRoot = tempDir.resolve("col");
      List<Document> docs = generateDocs(300, SEED);
      float[] query = randomVector(new Random(SEED + 5));

      try (var col =
          VectorCollection.builder()
              .dimension(DIM)
              .metric(SimilarityFunction.EUCLIDEAN)
              .indexType(IndexType.HNSW)
              .hnswM(16)
              .hnswEfConstruction(100)
              .storagePath(storageRoot)
              .build()) {
        col.addAll(docs);
        col.commit();
      }
      // Reopen and issue a multi-start request.
      try (var col =
          VectorCollection.builder()
              .dimension(DIM)
              .metric(SimilarityFunction.EUCLIDEAN)
              .indexType(IndexType.HNSW)
              .hnswM(16)
              .hnswEfConstruction(100)
              .storagePath(storageRoot)
              .build()) {
        SearchResult r = col.search(SearchRequest.builder(query, 10).searchMultiStart(4).build());
        assertThat(r.hits()).hasSize(10);
      }
    }
  }

  @Nested
  @Tag("unit")
  class ConcurrencySmoke {

    @Test
    void concurrencySmokeSearchBatchPlusMultiStart() {
      List<Document> docs = generateDocs(1000, SEED);
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

        Random rng = new Random(SEED + 11);
        List<SearchRequest> reqs = new ArrayList<>(16);
        for (int i = 0; i < 16; i++) {
          reqs.add(SearchRequest.builder(randomVector(rng), 10).searchMultiStart(4).build());
        }
        List<SearchResult> results = col.searchBatch(reqs);
        assertThat(results).hasSize(16);
        for (SearchResult r : results) {
          assertThat(r.hits()).hasSize(10);
        }
      }
    }
  }

  @Nested
  @Tag("unit")
  class NonHnswBackends {

    @Test
    void flatAdapterIgnoresMultiStart() {
      List<Document> docs = generateDocs(200, SEED);
      float[] query = randomVector(new Random(SEED + 13));
      try (var col =
          VectorCollection.builder()
              .dimension(DIM)
              .metric(SimilarityFunction.EUCLIDEAN)
              .indexType(IndexType.FLAT)
              .build()) {
        col.addAll(docs);
        col.commit();

        SearchResult single = col.search(SearchRequest.builder(query, 10).build());
        SearchResult multi =
            col.search(SearchRequest.builder(query, 10).searchMultiStart(4).build());
        // FLAT is deterministic brute force — searchMultiStart must be a no-op.
        List<String> singleIds = new ArrayList<>();
        for (SearchResult.Hit h : single.hits()) singleIds.add(h.id());
        List<String> multiIds = new ArrayList<>();
        for (SearchResult.Hit h : multi.hits()) multiIds.add(h.id());
        assertThat(multiIds).isEqualTo(singleIds);
      }
    }

    @Test
    void preFilterIgnoresMultiStart() {
      List<Document> docs = generateDocs(400, SEED);
      float[] query = randomVector(new Random(SEED + 17));
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

        // ACORN pre-filter active when a filter is supplied on an HNSW backend — the
        // searchMultiStart parameter is silently ignored (logged at FINE).
        SearchResult r =
            col.search(
                SearchRequest.builder(query, 10)
                    .filter(Filters.eq("color", "red"))
                    .searchMultiStart(4)
                    .build());
        assertThat(r.hits().size()).isBetween(1, 10);
        for (SearchResult.Hit h : r.hits()) {
          MetadataValue color = h.document().metadata().get("color");
          assertThat(((MetadataValue.Str) color).value()).isEqualTo("red");
        }
      }
    }
  }
}
