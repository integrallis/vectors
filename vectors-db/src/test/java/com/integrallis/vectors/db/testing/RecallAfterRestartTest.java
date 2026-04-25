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
package com.integrallis.vectors.db.testing;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.QuantizerKind;
import com.integrallis.vectors.db.SearchRequest;
import com.integrallis.vectors.db.SearchResult;
import com.integrallis.vectors.db.VectorCollection;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.TestFactory;
import org.junit.jupiter.api.io.TempDir;

/**
 * Verifies that search recall is identical before and after a close/reopen cycle.
 *
 * <p>Inspired by Weaviate's chaos-engineering {@code compare_recall_after_restart.sh}: build an
 * index, measure recall, close (simulating a crash or restart), reopen from disk, and assert that
 * recall is unchanged.
 *
 * <p>Tests are tagged {@code @Tag("slow")} because each variation builds an index, searches,
 * closes, and reopens.
 */
class RecallAfterRestartTest {

  private static final int DIM = 64;
  private static final int DOC_COUNT = 500;
  private static final int K = 10;
  private static final long SEED = 42L;
  private static final long QUERY_SEED = 999L;

  @TempDir Path tempDir;

  @TestFactory
  @Tag("slow")
  Stream<DynamicTest> recallSurvivesRestart() {
    return VectorConfigVariations.builder()
        .indexTypes(IndexType.HNSW, IndexType.VAMANA)
        .quantizers(QuantizerKind.NONE, QuantizerKind.SQ8)
        .metrics(SimilarityFunction.EUCLIDEAN, SimilarityFunction.COSINE)
        .dimensions(DIM)
        .buildTests(this::assertRecallSurvivesRestart);
  }

  private void assertRecallSurvivesRestart(VectorConfigVariations.Config config) {
    Path storageRoot = tempDir.resolve(config.toString().replace('/', '_'));
    List<Document> docs = VectorSearchTestSupport.generateDocs(DOC_COUNT, config.dimension(), SEED);
    float[] query =
        VectorSearchTestSupport.randomVector(config.dimension(), new java.util.Random(QUERY_SEED));

    float[] scoresBefore;
    String[] idsBefore;

    // Phase 1: build, search, record results
    try (VectorCollection col = config.newBuilder().storagePath(storageRoot).build()) {
      col.addAll(docs);
      col.commit();

      SearchResult before =
          col.search(
              SearchRequest.builder(query, K).searchListSize(200).overQueryFactor(2.0f).build());
      assertThat(before.hits()).isNotEmpty();

      scoresBefore = new float[before.hits().size()];
      idsBefore = new String[before.hits().size()];
      for (int i = 0; i < before.hits().size(); i++) {
        scoresBefore[i] = before.hits().get(i).score();
        idsBefore[i] = before.hits().get(i).id();
      }
    }

    // Phase 2: reopen from disk, search again
    try (VectorCollection col = config.newBuilder().storagePath(storageRoot).build()) {
      assertThat(col.size()).isEqualTo(DOC_COUNT);

      SearchResult after =
          col.search(
              SearchRequest.builder(query, K).searchListSize(200).overQueryFactor(2.0f).build());
      assertThat(after.hits()).hasSameSizeAs(List.of(idsBefore));

      // Assert same ids in same order
      String[] idsAfter = new String[after.hits().size()];
      float[] scoresAfter = new float[after.hits().size()];
      for (int i = 0; i < after.hits().size(); i++) {
        idsAfter[i] = after.hits().get(i).id();
        scoresAfter[i] = after.hits().get(i).score();
      }
      assertThat(idsAfter).isEqualTo(idsBefore);

      // Assert scores match within float epsilon
      for (int i = 0; i < scoresBefore.length; i++) {
        assertThat(scoresAfter[i])
            .as("score[%d] for id=%s", i, idsBefore[i])
            .isCloseTo(scoresBefore[i], org.assertj.core.data.Offset.offset(1e-4f));
      }
    }
  }
}
