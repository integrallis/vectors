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
package com.integrallis.vectors.demo.springai;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.VectorCollection;
import com.integrallis.vectors.spring.ai.JavaVectorsVectorStore;
import java.util.List;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;

/**
 * Spring Boot RAG demo using {@link JavaVectorsVectorStore} as a drop-in replacement for Spring
 * AI's {@code SimpleVectorStore}.
 *
 * <p>This demo wires the same {@link VectorStore} interface Spring AI applications already code
 * against, but backed by an HNSW-indexed, SIMD-accelerated java-vectors collection.
 *
 * <p>The demo uses a {@link DeterministicEmbeddingModel} so it runs without API keys or network
 * access. In a real application, substitute your preferred Spring AI {@link EmbeddingModel}
 * (OpenAI, Ollama, Transformers, etc.) — no other changes are required.
 *
 * <p>Run:
 *
 * <pre>
 *   ./gradlew :demos:spring-ai-rag:run
 * </pre>
 */
@SpringBootApplication
public class SpringAiRagApp {

  public static void main(String[] args) {
    SpringApplication.run(SpringAiRagApp.class, args);
  }

  @Bean
  EmbeddingModel embeddingModel() {
    return new DeterministicEmbeddingModel(128);
  }

  @Bean(destroyMethod = "close")
  VectorCollection collection(EmbeddingModel embeddingModel) {
    return VectorCollection.builder()
        .dimension(embeddingModel.dimensions())
        .metric(SimilarityFunction.COSINE)
        .indexType(IndexType.HNSW)
        .hnswM(16)
        .hnswEfConstruction(100)
        .autoCommitThreshold(32)
        .build();
  }

  @Bean(destroyMethod = "close")
  VectorStore vectorStore(EmbeddingModel embeddingModel, VectorCollection collection) {
    return JavaVectorsVectorStore.builder(embeddingModel, collection)
        .collectionName("spring-ai-rag-demo")
        .commitAfterAdd(true)
        .build();
  }

  @Bean
  CommandLineRunner run(VectorStore store) {
    return args -> {
      store.add(
          List.of(
              new Document(
                  "Java 25 ships the Vector API and mature foreign-function interop (FFM)."),
              new Document(
                  "HNSW is a multi-layer navigable small-world graph for approximate nearest"
                      + " neighbor search."),
              new Document(
                  "Vamana is DiskANN's single-layer pruned graph with graduated alpha pruning."),
              new Document(
                  "Product quantization splits vectors into subspaces and encodes each independently"
                      + " with k-means codebooks."),
              new Document(
                  "Scalar quantization reduces float32 vectors to int8 with per-vector correction"
                      + " factors, about 4x compression."),
              new Document(
                  "mmap-based persistence lets a vector database share one directory layout across"
                      + " hosts and survives process restarts."),
              new Document(
                  "Spring AI's VectorStore and LangChain4j's EmbeddingStore are the two canonical"
                      + " Java AI vector-search interfaces.")));

      String question = "How does java-vectors make graph search fast?";
      List<Document> hits =
          store.similaritySearch(SearchRequest.builder().query(question).topK(3).build());

      System.out.println();
      System.out.println("Q: " + question);
      System.out.println("Top-3 retrieved context:");
      int i = 1;
      for (Document d : hits) {
        System.out.printf("  [%d] %s%n", i++, d.getText());
      }
    };
  }
}
