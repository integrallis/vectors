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
package com.integrallis.vectors.demo.langchain4j;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.VectorCollection;
import com.integrallis.vectors.langchain4j.JavaVectorsEmbeddingStore;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import java.util.List;

/**
 * LangChain4j RAG demo using {@link JavaVectorsEmbeddingStore} as a drop-in replacement for
 * LangChain4j's {@code InMemoryEmbeddingStore}.
 *
 * <p>Same {@link EmbeddingStore} interface, same LangChain4j application code — now backed by
 * HNSW-indexed, SIMD-accelerated java-vectors storage.
 *
 * <p>Uses a zero-dependency {@link DeterministicEmbeddingModel} so the demo runs without model
 * downloads or API keys. In a real application, substitute e.g. {@code AllMiniLmL6V2EmbeddingModel}
 * or {@code OpenAiEmbeddingModel}.
 *
 * <p>Run:
 *
 * <pre>
 *   ./gradlew :demos:langchain4j-rag:run
 * </pre>
 */
public final class LangChain4jRagApp {

  private LangChain4jRagApp() {}

  public static void main(String[] args) {
    EmbeddingModel embeddingModel = new DeterministicEmbeddingModel(128);

    try (VectorCollection collection =
            VectorCollection.builder()
                .dimension(embeddingModel.dimension())
                .metric(SimilarityFunction.COSINE)
                .indexType(IndexType.HNSW)
                .hnswM(16)
                .hnswEfConstruction(100)
                .autoCommitThreshold(32)
                .build();
        JavaVectorsEmbeddingStore store =
            JavaVectorsEmbeddingStore.builder(collection).commitAfterAdd(true).build()) {

      List<TextSegment> docs =
          List.of(
              TextSegment.from(
                  "Java 25 ships the Vector API and mature foreign-function interop (FFM).",
                  Metadata.from("topic", "java")),
              TextSegment.from(
                  "HNSW is a multi-layer navigable small-world graph for approximate nearest"
                      + " neighbor search.",
                  Metadata.from("topic", "indexing")),
              TextSegment.from(
                  "Vamana is DiskANN's single-layer pruned graph with graduated alpha pruning.",
                  Metadata.from("topic", "indexing")),
              TextSegment.from(
                  "Product quantization splits vectors into subspaces and encodes each"
                      + " independently with k-means codebooks.",
                  Metadata.from("topic", "quantization")),
              TextSegment.from(
                  "Scalar quantization reduces float32 vectors to int8 with per-vector correction"
                      + " factors, about 4x compression.",
                  Metadata.from("topic", "quantization")),
              TextSegment.from(
                  "mmap-based persistence lets a vector database share one directory layout"
                      + " across hosts.",
                  Metadata.from("topic", "storage")));

      List<Embedding> embeddings =
          docs.stream().map(s -> embeddingModel.embed(s).content()).toList();
      store.addAll(embeddings, docs);

      String question = "How does HNSW make graph search fast?";
      Embedding qv = embeddingModel.embed(question).content();
      EmbeddingSearchRequest request =
          EmbeddingSearchRequest.builder().queryEmbedding(qv).maxResults(3).build();

      System.out.println();
      System.out.println("Q: " + question);
      System.out.println("Top-3 retrieved context:");
      int i = 1;
      for (EmbeddingMatch<TextSegment> match : store.search(request).matches()) {
        System.out.printf("  [%d] score=%.4f  %s%n", i++, match.score(), match.embedded().text());
      }
    }
  }
}
