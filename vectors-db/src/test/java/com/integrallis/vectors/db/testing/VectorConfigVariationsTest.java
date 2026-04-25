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

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.QuantizerKind;
import com.integrallis.vectors.db.SearchRequest;
import com.integrallis.vectors.db.VectorCollection;
import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestFactory;

class VectorConfigVariationsTest {

  @Nested
  @Tag("unit")
  class BuilderDefaults {

    @Test
    void defaultConfigsAreReasonable() {
      List<VectorConfigVariations.Config> configs = VectorConfigVariations.builder().build();
      // Default: 2 index types × 2 quantizers × 2 metrics × 2 dimensions = 16
      assertThat(configs).hasSize(16);
    }

    @Test
    void customConfigsProduceCartesianProduct() {
      List<VectorConfigVariations.Config> configs =
          VectorConfigVariations.builder()
              .indexTypes(IndexType.HNSW)
              .quantizers(QuantizerKind.NONE, QuantizerKind.SQ8, QuantizerKind.PQ)
              .metrics(SimilarityFunction.COSINE)
              .dimensions(64)
              .build();
      // 1 × 3 × 1 × 1 = 3
      assertThat(configs).hasSize(3);
    }

    @Test
    void configToStringIsDescriptive() {
      var config =
          new VectorConfigVariations.Config(
              IndexType.HNSW, QuantizerKind.SQ8, SimilarityFunction.COSINE, 128);
      assertThat(config.toString()).isEqualTo("HNSW/SQ8/COSINE/dim=128");
    }

    @Test
    void newBuilderCreatesValidCollection() {
      var config =
          new VectorConfigVariations.Config(
              IndexType.HNSW, QuantizerKind.NONE, SimilarityFunction.EUCLIDEAN, 32);
      try (VectorCollection col = config.newBuilder().build()) {
        assertThat(col.config().dimension()).isEqualTo(32);
        assertThat(col.config().indexType()).isEqualTo(IndexType.HNSW);
      }
    }
  }

  @Nested
  @Tag("unit")
  class DynamicTestGeneration {

    @TestFactory
    Stream<DynamicTest> searchReturnsResultsAcrossConfigs() {
      return VectorConfigVariations.builder()
          .indexTypes(IndexType.HNSW, IndexType.VAMANA)
          .quantizers(QuantizerKind.NONE)
          .metrics(SimilarityFunction.EUCLIDEAN)
          .dimensions(32)
          .buildTests(
              config -> {
                try (VectorCollection col = config.newBuilder().build()) {
                  var docs = VectorSearchTestSupport.generateDocs(50, config.dimension(), 42L);
                  col.addAll(docs);
                  col.commit();

                  var result = col.search(SearchRequest.builder(docs.get(0).vector(), 5).build());
                  assertThat(result.hits()).isNotEmpty();
                  assertThat(result.hits().size()).isLessThanOrEqualTo(5);
                }
              });
    }
  }
}
