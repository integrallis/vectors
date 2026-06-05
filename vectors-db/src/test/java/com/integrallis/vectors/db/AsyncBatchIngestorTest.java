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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.core.SimilarityFunction;
import java.nio.file.Path;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.SubmissionPublisher;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Async batch ingestion (I.8): direct submit + reactive Flow paths, batching, and completion. */
@Tag("unit")
class AsyncBatchIngestorTest {

  private static final int DIM = 8;

  private static Document doc(int i) {
    float[] v = new float[DIM];
    for (int d = 0; d < DIM; d++) {
      v[d] = i + d * 0.01f;
    }
    return Document.of("doc-" + i, v);
  }

  private static VectorCollection inMemory() {
    return VectorCollection.builder()
        .dimension(DIM)
        .metric(SimilarityFunction.EUCLIDEAN)
        .indexType(IndexType.FLAT)
        .build();
  }

  @Test
  void directSubmitBatchesAndCommitsAll() throws Exception {
    try (VectorCollection col = inMemory()) {
      AsyncBatchIngestor ingestor = new AsyncBatchIngestor(col, 10);
      for (int i = 0; i < 25; i++) {
        ingestor.submit(doc(i));
      }
      ingestor.close(); // flush final partial batch (5) + finish

      CommitToken token = ingestor.completion().toCompletableFuture().get(10, TimeUnit.SECONDS);
      assertThat(token.documentsCommitted()).isEqualTo(25);
      assertThat(col.size()).isEqualTo(25);
      assertThat(col.contains("doc-24")).isTrue();
    }
  }

  @Test
  void reactiveFlowPublisherPath() throws Exception {
    try (VectorCollection col = inMemory()) {
      SubmissionPublisher<Document> publisher = new SubmissionPublisher<>();
      AsyncBatchIngestor ingestor = new AsyncBatchIngestor(col, 8);
      publisher.subscribe(ingestor);
      for (int i = 0; i < 20; i++) {
        publisher.submit(doc(i));
      }
      publisher.close(); // -> onComplete -> final commit

      CommitToken token = ingestor.completion().toCompletableFuture().get(10, TimeUnit.SECONDS);
      assertThat(token.documentsCommitted()).isEqualTo(20);
      assertThat(col.size()).isEqualTo(20);
    }
  }

  @Test
  void failedAddCompletesExceptionally() {
    try (VectorCollection col = inMemory()) {
      AsyncBatchIngestor ingestor = new AsyncBatchIngestor(col, 100);
      ingestor.submit(doc(1));
      ingestor.submit(doc(1)); // duplicate id within the same uncommitted batch -> add throws
      ingestor.close();

      CompletableFuture<CommitToken> future = ingestor.completion().toCompletableFuture();
      assertThatThrownBy(() -> future.get(10, TimeUnit.SECONDS))
          .isInstanceOf(ExecutionException.class)
          .hasCauseInstanceOf(IllegalArgumentException.class);
      // Further submits are rejected once terminated.
      assertThatThrownBy(() -> ingestor.submit(doc(2))).isInstanceOf(IllegalStateException.class);
    }
  }

  @Test
  void persistentTokenCarriesGenerationNumber(@TempDir Path tmp) throws Exception {
    try (VectorCollection col =
        VectorCollection.builder()
            .dimension(DIM)
            .metric(SimilarityFunction.EUCLIDEAN)
            .indexType(IndexType.FLAT)
            .storagePath(tmp)
            .build()) {
      AsyncBatchIngestor ingestor = new AsyncBatchIngestor(col, 5);
      for (int i = 0; i < 15; i++) {
        ingestor.submit(doc(i));
      }
      ingestor.close();

      CommitToken token = ingestor.completion().toCompletableFuture().get(10, TimeUnit.SECONDS);
      assertThat(token.documentsCommitted()).isEqualTo(15);
      // Persistent collections advance the generation number on each commit.
      assertThat(token.generationNumber()).isGreaterThan(0L);
      assertThat(token.generationNumber()).isEqualTo(col.generationNumber());
      assertThat(col.size()).isEqualTo(15);
    }
  }
}
