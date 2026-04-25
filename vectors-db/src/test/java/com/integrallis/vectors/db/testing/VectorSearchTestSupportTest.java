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
import com.integrallis.vectors.db.VectorCollection;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

class VectorSearchTestSupportTest {

  private static final int DIM = 32;
  private static final long SEED = 42L;

  @Nested
  @Tag("unit")
  class DocumentGeneration {

    @Test
    void generateDocsProducesDeterministicOutput() {
      List<Document> a = VectorSearchTestSupport.generateDocs(10, DIM, SEED);
      List<Document> b = VectorSearchTestSupport.generateDocs(10, DIM, SEED);
      assertThat(a).hasSize(10);
      for (int i = 0; i < 10; i++) {
        assertThat(a.get(i).id()).isEqualTo(b.get(i).id());
        assertThat(a.get(i).vector()).isEqualTo(b.get(i).vector());
      }
    }

    @Test
    void generateDocsWithMetadataIncludesAllFields() {
      List<Document> docs = VectorSearchTestSupport.generateDocsWithMetadata(5, DIM, SEED);
      for (Document doc : docs) {
        assertThat(doc.metadata()).containsKeys("idx", "name", "flag", "group");
      }
    }

    @Test
    void randomUnitVectorIsNormalized() {
      Random rng = new Random(SEED);
      float[] v = VectorSearchTestSupport.randomUnitVector(DIM, rng);
      double norm = 0;
      for (float x : v) norm += (double) x * x;
      assertThat(Math.sqrt(norm)).isCloseTo(1.0, org.assertj.core.data.Offset.offset(1e-5));
    }
  }

  @Nested
  @Tag("unit")
  class RecallComputation {

    @Test
    void bruteForceTopKIdsReturnsCorrectIds() {
      List<Document> docs = VectorSearchTestSupport.generateDocs(100, DIM, SEED);
      float[] query = docs.get(0).vector();
      Set<String> top1 =
          VectorSearchTestSupport.bruteForceTopKIds(docs, query, 1, SimilarityFunction.EUCLIDEAN);
      assertThat(top1).containsExactly("doc-0");
    }

    @Test
    void measureRecallAgainstSelfIsOne() {
      List<Document> docs = VectorSearchTestSupport.generateDocs(50, DIM, SEED);
      float[] query = docs.get(0).vector();

      try (VectorCollection col =
          VectorCollection.builder().dimension(DIM).metric(SimilarityFunction.EUCLIDEAN).build()) {
        col.addAll(docs);
        col.commit();

        double recall =
            VectorSearchTestSupport.searchAndMeasureRecall(
                docs, query, 10, SimilarityFunction.EUCLIDEAN, col);
        assertThat(recall).isEqualTo(1.0);
      }
    }
  }

  @Nested
  @Tag("unit")
  class RandomizedConfig {

    @Test
    void randomBuilderProducesValidConfiguration() {
      Random rng = new Random(SEED);
      var builder =
          VectorSearchTestSupport.randomBuilder(rng)
              .dimension(DIM)
              .metric(SimilarityFunction.COSINE);
      try (VectorCollection col = builder.build()) {
        assertThat(col.config().dimension()).isEqualTo(DIM);
        assertThat(col.config().metric()).isEqualTo(SimilarityFunction.COSINE);
        assertThat(col.config().indexType()).isIn(IndexType.HNSW, IndexType.VAMANA);
      }
    }

    @Test
    void randomBuilderCompleteProducesValidConfiguration() {
      Random rng = new Random(SEED);
      try (VectorCollection col = VectorSearchTestSupport.randomBuilderComplete(rng).build()) {
        assertThat(col.config().dimension()).isGreaterThan(0);
        assertThat(col.config().metric()).isNotNull();
        assertThat(col.config().indexType()).isIn(IndexType.HNSW, IndexType.VAMANA);
      }
    }

    @Test
    void sameSeedProducesSameConfig() {
      Random rng1 = new Random(SEED);
      Random rng2 = new Random(SEED);

      try (VectorCollection col1 = VectorSearchTestSupport.randomBuilderComplete(rng1).build();
          VectorCollection col2 = VectorSearchTestSupport.randomBuilderComplete(rng2).build()) {
        assertThat(col1.config().dimension()).isEqualTo(col2.config().dimension());
        assertThat(col1.config().metric()).isEqualTo(col2.config().metric());
        assertThat(col1.config().indexType()).isEqualTo(col2.config().indexType());
      }
    }

    @Test
    void differentSeedsProduceDifferentConfigs() {
      Set<String> configs = new HashSet<>();
      for (long s = 0; s < 20; s++) {
        Random rng = new Random(s);
        IndexType idx = VectorSearchTestSupport.randomGraphIndexType(rng);
        QuantizerKind q = VectorSearchTestSupport.randomQuantizer(rng);
        SimilarityFunction m = VectorSearchTestSupport.randomMetric(rng);
        configs.add(idx + "/" + q + "/" + m);
      }
      assertThat(configs.size())
          .as("20 seeds should produce multiple distinct configs")
          .isGreaterThan(1);
    }
  }
}
