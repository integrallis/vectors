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

import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.core.SimilarityFunction;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * An ingest committed in batches must retrieve the same documents as one committed once. Before
 * append-on-commit the batched path rebuilt the whole graph per commit, which was quadratic in the
 * number of commits; the risk in fixing that is a cheaper graph that retrieves worse, so these
 * tests compare results rather than timings.
 */
class CommitAppendsHnswGraphTest {

  private static final int DIM = 32;
  private static final int COUNT = 3_000;
  private static final int K = 10;

  @Test
  void batchedCommitsRetrieveWhatASingleCommitRetrieves(@TempDir Path dir) throws IOException {
    float[][] vectors = randomUnitVectors(COUNT, DIM, 11L);
    float[][] queries = randomUnitVectors(50, DIM, 12L);

    try (VectorCollection batched = open(dir.resolve("batched"));
        VectorCollection once = open(dir.resolve("once"))) {

      for (int from = 0; from < COUNT; from += 500) {
        batched.addAll(documents(vectors, from, Math.min(from + 500, COUNT)));
        batched.commit();
      }
      once.addAll(documents(vectors, 0, COUNT));
      once.commit();

      assertThat(batched.size()).isEqualTo(COUNT);
      assertThat(batched.generationNumber()).isGreaterThan(once.generationNumber());

      double overlap = meanOverlap(batched, once, queries);
      assertThat(overlap)
          .as("top-%d overlap between a batched and a single-commit collection", K)
          .isGreaterThan(0.90);

      double recall = meanRecall(batched, vectors, queries);
      assertThat(recall).as("recall@%d of the batched collection", K).isGreaterThan(0.90);
    }
  }

  @Test
  void inMemoryBatchedCommitsRetrieveWhatASingleCommitRetrieves() {
    float[][] vectors = randomUnitVectors(COUNT, DIM, 21L);
    float[][] queries = randomUnitVectors(50, DIM, 22L);

    try (VectorCollection batched = open(null);
        VectorCollection once = open(null)) {

      for (int from = 0; from < COUNT; from += 500) {
        batched.addAll(documents(vectors, from, Math.min(from + 500, COUNT)));
        batched.commit();
      }
      once.addAll(documents(vectors, 0, COUNT));
      once.commit();

      assertThat(batched.size()).isEqualTo(COUNT);
      assertThat(meanOverlap(batched, once, queries)).isGreaterThan(0.90);
      assertThat(meanRecall(batched, vectors, queries)).isGreaterThan(0.90);
    }
  }

  @Test
  void deletesStillApplyAcrossAppendedGenerations(@TempDir Path dir) {
    float[][] vectors = randomUnitVectors(1_000, DIM, 31L);
    try (VectorCollection collection = open(dir.resolve("deletes"))) {
      collection.addAll(documents(vectors, 0, 500));
      collection.commit();
      collection.delete("v1");
      collection.delete("v2");
      collection.commit();
      collection.addAll(documents(vectors, 500, 1_000));
      collection.commit();

      assertThat(collection.size()).isEqualTo(998);
      SearchResult result = collection.search(SearchRequest.builder(vectors[1], 5).build());
      assertThat(result.hits()).extracting(SearchResult.Hit::id).doesNotContain("v1", "v2");
    }
  }

  @Test
  void upsertsAcrossGenerationsReturnTheNewVector(@TempDir Path dir) {
    float[][] vectors = randomUnitVectors(400, DIM, 41L);
    try (VectorCollection collection = open(dir.resolve("upserts"))) {
      collection.addAll(documents(vectors, 0, 300));
      collection.commit();

      float[] replacement = vectors[299].clone();
      replacement[0] = -replacement[0];
      collection.upsert(Document.of("v0", replacement, "replaced"));
      collection.commit();

      assertThat(collection.size()).isEqualTo(300);
      SearchResult result = collection.search(SearchRequest.builder(replacement, 1).build());
      assertThat(result.hits().getFirst().id()).isEqualTo("v0");
    }
  }

  // -------------------------------------------------------------------------

  private static VectorCollection open(Path storage) {
    VectorCollectionBuilder builder =
        VectorCollection.builder()
            .dimension(DIM)
            .metric(SimilarityFunction.COSINE)
            .indexType(IndexType.HNSW);
    if (storage != null) {
      builder.storagePath(storage.toAbsolutePath());
    }
    return builder.build();
  }

  private static List<Document> documents(float[][] vectors, int from, int to) {
    List<Document> documents = new ArrayList<>(to - from);
    for (int i = from; i < to; i++) {
      documents.add(Document.of("v" + i, vectors[i]));
    }
    return documents;
  }

  private static double meanOverlap(VectorCollection a, VectorCollection b, float[][] queries) {
    double total = 0;
    for (float[] query : queries) {
      Set<String> left = topIds(a, query);
      Set<String> right = topIds(b, query);
      Set<String> both = new LinkedHashSet<>(left);
      both.retainAll(right);
      total += both.size() / (double) K;
    }
    return total / queries.length;
  }

  private static Set<String> topIds(VectorCollection collection, float[] query) {
    Set<String> ids = new LinkedHashSet<>();
    for (SearchResult.Hit hit : collection.search(SearchRequest.builder(query, K).build()).hits()) {
      ids.add(hit.id());
    }
    return ids;
  }

  private static double meanRecall(
      VectorCollection collection, float[][] vectors, float[][] queries) {
    double total = 0;
    for (float[] query : queries) {
      Set<String> truth = bruteForceTop(query, vectors);
      Set<String> found = topIds(collection, query);
      found.retainAll(truth);
      total += found.size() / (double) K;
    }
    return total / queries.length;
  }

  private static Set<String> bruteForceTop(float[] query, float[][] vectors) {
    Integer[] order = new Integer[vectors.length];
    for (int i = 0; i < vectors.length; i++) {
      order[i] = i;
    }
    Arrays.sort(
        order,
        (x, y) ->
            Float.compare(
                SimilarityFunction.COSINE.compare(query, vectors[y]),
                SimilarityFunction.COSINE.compare(query, vectors[x])));
    Set<String> ids = new LinkedHashSet<>();
    for (int i = 0; i < K; i++) {
      ids.add("v" + order[i]);
    }
    return ids;
  }

  private static float[][] randomUnitVectors(int count, int dim, long seed) {
    SplittableRandom random = new SplittableRandom(seed);
    float[][] out = new float[count][dim];
    for (int i = 0; i < count; i++) {
      double sum = 0;
      for (int d = 0; d < dim; d++) {
        out[i][d] = (float) random.nextGaussian();
        sum += (double) out[i][d] * out[i][d];
      }
      float norm = (float) Math.sqrt(sum);
      for (int d = 0; d < dim; d++) {
        out[i][d] /= norm;
      }
    }
    return out;
  }
}
