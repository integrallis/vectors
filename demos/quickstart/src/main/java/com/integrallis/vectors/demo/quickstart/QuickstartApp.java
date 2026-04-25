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
package com.integrallis.vectors.demo.quickstart;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.Document;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.SearchRequest;
import com.integrallis.vectors.db.SearchResult;
import com.integrallis.vectors.db.VectorCollection;

/**
 * Minimal java-vectors quickstart: build a collection, add a handful of documents, search, and
 * print the results.
 *
 * <p>Run from the repo root:
 *
 * <pre>
 *   ./gradlew :demos:quickstart:run
 * </pre>
 */
public final class QuickstartApp {

  private QuickstartApp() {}

  public static void main(String[] args) {
    try (VectorCollection collection =
        VectorCollection.builder()
            .dimension(4)
            .metric(SimilarityFunction.COSINE)
            .indexType(IndexType.HNSW)
            .hnswM(8)
            .hnswEfConstruction(64)
            .build()) {

      collection.add(Document.of("red", new float[] {1.0f, 0.1f, 0.1f, 0.0f}, "a red apple"));
      collection.add(Document.of("fire", new float[] {0.9f, 0.2f, 0.0f, 0.1f}, "a warm fire"));
      collection.add(Document.of("sky", new float[] {0.0f, 0.1f, 0.9f, 0.2f}, "a blue sky"));
      collection.add(Document.of("ocean", new float[] {0.1f, 0.0f, 0.8f, 0.3f}, "a deep ocean"));
      collection.add(Document.of("leaf", new float[] {0.0f, 0.9f, 0.1f, 0.1f}, "a green leaf"));
      collection.commit();

      float[] query = {0.8f, 0.15f, 0.05f, 0.05f};
      SearchResult result =
          collection.search(SearchRequest.builder(query, 3).includeText(true).build());

      System.out.printf(
          "collection size: %d, search took %.2f ms%n",
          collection.size(), result.searchTimeNanos() / 1_000_000.0);
      for (SearchResult.Hit hit : result.hits()) {
        System.out.printf("  %.4f  %-6s  %s%n", hit.score(), hit.id(), hit.document().text());
      }
    }
  }
}
