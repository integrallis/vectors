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
package com.integrallis.vectors.studio.core.connection;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.core.MetadataValue;
import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.VectorCollection;
import com.integrallis.vectors.studio.core.search.DocumentView;
import com.integrallis.vectors.studio.core.search.SearchHit;
import com.integrallis.vectors.studio.core.search.SearchSpec;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("integration")
class StudioBackendEmbeddedIT {

  private static final int DIM = 8;
  private VectorCollection collection;
  private EmbeddedStudioBackend backend;

  @BeforeEach
  void setUp() {
    collection =
        VectorCollection.builder()
            .dimension(DIM)
            .metric(SimilarityFunction.COSINE)
            .indexType(IndexType.FLAT)
            .build();
    for (int i = 0; i < 100; i++) {
      float[] v = new float[DIM];
      v[i % DIM] = 1.0f;
      collection.add(
          new Document("doc-" + i, v, "text " + i, Map.of("idx", MetadataValue.of((double) i))));
    }
    collection.commit();
    backend = EmbeddedStudioBackend.withCollections(Map.of("docs", collection));
  }

  @AfterEach
  void tearDown() {
    if (backend != null) backend.close();
  }

  @Test
  void listAndDescribe() {
    List<CollectionSummary> all = backend.listCollections();
    assertThat(all).hasSize(1);
    CollectionSummary sum = backend.describe("docs");
    assertThat(sum.name()).isEqualTo("docs");
    assertThat(sum.dimension()).isEqualTo(DIM);
    assertThat(sum.size()).isEqualTo(100);
  }

  @Test
  void searchReturnsTopKHits() {
    float[] query = new float[DIM];
    query[0] = 1.0f;
    SearchSpec spec = new SearchSpec(query, null, 5, null, false, false, true);
    List<SearchHit> hits = backend.search("docs", spec);
    assertThat(hits).hasSize(5);
    assertThat(hits.get(0).score()).isGreaterThan(0.99);
  }

  @Test
  void previewDocumentsPaginates() {
    List<DocumentView> page = backend.previewDocuments("docs", 10, 5);
    assertThat(page).hasSize(5);
    assertThat(page.get(0).id()).isEqualTo("doc-10");
    assertThat(page.get(0).vector()).hasSize(DIM);
  }

  @Test
  void vectorBatchReturnsRequestedRows() {
    float[][] batch = backend.vectorBatch("docs", List.of("doc-0", "doc-1", "missing"));
    assertThat(batch).hasNumberOfRows(2);
    assertThat(batch[0]).hasSize(DIM);
  }

  @Test
  void streamAllVectorsTouchesEveryDocument() {
    AtomicInteger counter = new AtomicInteger();
    AtomicInteger lastProgress = new AtomicInteger();
    List<String> ids = new ArrayList<>();
    backend.streamAllVectors(
        "docs",
        (id, v) -> {
          ids.add(id);
          counter.incrementAndGet();
        },
        lastProgress::set);
    assertThat(counter.get()).isEqualTo(100);
    assertThat(ids).hasSize(100);
    assertThat(lastProgress.get()).isEqualTo(100);
  }

  @Test
  void getDocumentReturnsHydratedView() {
    DocumentView v = backend.getDocument("docs", "doc-7");
    assertThat(v).isNotNull();
    assertThat(v.id()).isEqualTo("doc-7");
    assertThat(v.metadata()).containsKey("idx");
  }
}
