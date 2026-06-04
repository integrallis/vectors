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
package com.integrallis.vectors.demo.rag.service;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.integrallis.vectors.demo.rag.model.CacheType;
import com.integrallis.vectors.demo.rag.model.ChatMessage;
import com.integrallis.vectors.demo.rag.model.LLMConfig;
import com.integrallis.vectors.server.ServerConfig;
import com.integrallis.vectors.server.VectorsServer;
import com.integrallis.vectors.server.client.DocumentPayload;
import com.integrallis.vectors.server.client.VectorsServerClient;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;
import java.util.Base64;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Captor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

@ExtendWith(MockitoExtension.class)
@Tag("integration")
@DisplayName("RAGService integration — embedded vectors-server")
class RAGServiceIntegrationTest {

  private static final String COLLECTION = "rag-test";
  private static final int DIM = 4;

  private VectorsServer.ServerHandle serverHandle;
  private VectorsServerClient client;

  @Mock EmbeddingModel embeddingModel;
  @Mock ChatModel chatModel;
  @Mock CostTracker costTracker;

  @Captor ArgumentCaptor<List<dev.langchain4j.data.message.ChatMessage>> messagesCaptor;

  @BeforeEach
  void setUp() {
    serverHandle = VectorsServer.start(ServerConfig.forTesting());
    client = new VectorsServerClient("http://localhost:" + serverHandle.port());

    // Create collection
    client.createCollection(COLLECTION, DIM, "COSINE", "FLAT", null);

    // Upsert 3 text chunks + 2 image chunks
    float[] v1 = {1.0f, 0.0f, 0.0f, 0.0f};
    float[] v2 = {0.0f, 1.0f, 0.0f, 0.0f};
    float[] v3 = {0.0f, 0.0f, 1.0f, 0.0f};
    float[] v4 = {0.7f, 0.7f, 0.0f, 0.0f};
    float[] v5 = {0.0f, 0.7f, 0.7f, 0.0f};

    String fakeBlob1 = Base64.getEncoder().encodeToString("fake-png-data-1".getBytes());
    String fakeBlob2 = Base64.getEncoder().encodeToString("fake-png-data-2".getBytes());

    client.upsertDocuments(
        COLLECTION,
        List.of(
            DocumentPayload.of(
                "txt-1", v1, "Introduction to vector search", Map.of("type", "TEXT", "page", 1)),
            DocumentPayload.of(
                "txt-2", v2, "HNSW algorithm details", Map.of("type", "TEXT", "page", 3)),
            DocumentPayload.of(
                "txt-3", v3, "Performance benchmarks", Map.of("type", "TEXT", "page", 7)),
            new DocumentPayload(
                "img-1",
                v4,
                "Image 1 from page 2 of doc.pdf (1024x768 PNG)",
                Map.of("type", "IMAGE", "page", 2),
                fakeBlob1),
            new DocumentPayload(
                "img-2",
                v5,
                "Image 2 from page 5 of doc.pdf (800x600 JPEG)",
                Map.of("type", "IMAGE", "page", 5),
                fakeBlob2)));

    client.commit(COLLECTION);
  }

  @AfterEach
  void tearDown() {
    if (serverHandle != null) {
      serverHandle.close();
    }
  }

  @Test
  @DisplayName("listing query — LLM calls list_images tool, deterministic response returned")
  void listingQueryUsesToolCall() {
    // Arrange: embedding model returns a vector
    float[] queryVector = {0.5f, 0.5f, 0.5f, 0.0f};
    when(embeddingModel.embed(anyString())).thenReturn(new Response<>(Embedding.from(queryVector)));
    when(costTracker.countTokens(anyString())).thenReturn(20);

    // LLM responds with a list_images tool call
    dev.langchain4j.agent.tool.ToolExecutionRequest toolCall =
        dev.langchain4j.agent.tool.ToolExecutionRequest.builder()
            .name("list_images")
            .arguments("{}")
            .build();
    dev.langchain4j.data.message.AiMessage aiMessage =
        dev.langchain4j.data.message.AiMessage.from(toolCall);
    ChatResponse chatResponse = ChatResponse.builder().aiMessage(aiMessage).build();
    when(chatModel.chat(any(dev.langchain4j.model.chat.request.ChatRequest.class)))
        .thenReturn(chatResponse);

    LLMConfig config = LLMConfig.defaultConfig(LLMConfig.Provider.OPENAI, "test-key");
    RAGService ragService =
        new RAGService(client, COLLECTION, embeddingModel, chatModel, costTracker, config);

    // Act
    com.integrallis.vectors.demo.rag.model.ChatMessage result =
        ragService.query("list the images in the document", CacheType.NONE);

    // Assert: LLM was called (to decide which tool to use)
    verify(chatModel).chat(any(dev.langchain4j.model.chat.request.ChatRequest.class));

    // Assert: deterministic response from list_images tool execution
    assertThat(result.content()).contains("2 image(s)");
    assertThat(result.content()).contains("Image 1 from page 2");
    assertThat(result.content()).contains("Image 2 from page 5");
    assertThat(result.content()).contains("1024x768");
    assertThat(result.content()).contains("800x600");
    assertThat(result.costUsd()).isEqualTo(0.0);
  }

  @Test
  @DisplayName("listing query for document with no images returns 'no images' message")
  void listingQueryWithNoImagesReturnsNoImagesMessage() {
    // Create a collection with only text chunks (no IMAGE type)
    client.deleteCollection(COLLECTION);
    client.createCollection(COLLECTION, DIM, "COSINE", "FLAT", null);
    float[] v1 = {1.0f, 0.0f, 0.0f, 0.0f};
    client.upsertDocuments(
        COLLECTION,
        List.of(
            DocumentPayload.of(
                "txt-only", v1, "Just text content", Map.of("type", "TEXT", "page", 1))));
    client.commit(COLLECTION);

    float[] queryVector = {0.5f, 0.5f, 0.5f, 0.0f};
    when(embeddingModel.embed(anyString())).thenReturn(new Response<>(Embedding.from(queryVector)));

    // LLM responds with list_images tool call
    dev.langchain4j.agent.tool.ToolExecutionRequest toolCall =
        dev.langchain4j.agent.tool.ToolExecutionRequest.builder()
            .name("list_images")
            .arguments("{}")
            .build();
    dev.langchain4j.data.message.AiMessage aiMessage =
        dev.langchain4j.data.message.AiMessage.from(toolCall);
    ChatResponse chatResponse = ChatResponse.builder().aiMessage(aiMessage).build();
    when(chatModel.chat(any(dev.langchain4j.model.chat.request.ChatRequest.class)))
        .thenReturn(chatResponse);

    LLMConfig config = LLMConfig.defaultConfig(LLMConfig.Provider.OPENAI, "test-key");
    RAGService ragService =
        new RAGService(client, COLLECTION, embeddingModel, chatModel, costTracker, config);

    // Act
    com.integrallis.vectors.demo.rag.model.ChatMessage result =
        ragService.query("list the images in the document", CacheType.NONE);

    // Assert: list_images tool execution returns "no images" message
    assertThat(result.content()).contains("No images were found");
  }

  @Test
  @DisplayName("non-listing query follows standard multimodal path")
  void nonListingQueryFollowsStandardPath() {
    // Arrange: vector close to txt-1
    float[] queryVector = {0.9f, 0.1f, 0.0f, 0.0f};
    when(embeddingModel.embed(anyString())).thenReturn(new Response<>(Embedding.from(queryVector)));

    dev.langchain4j.data.message.AiMessage aiMessage =
        dev.langchain4j.data.message.AiMessage.from("Vector search is a technique...");
    ChatResponse chatResponse = ChatResponse.builder().aiMessage(aiMessage).build();
    when(chatModel.chat(any(dev.langchain4j.model.chat.request.ChatRequest.class)))
        .thenReturn(chatResponse);

    when(costTracker.countTokens(anyString())).thenReturn(15);
    when(costTracker.calculateCost(any(), anyString(), anyInt())).thenReturn(0.001);

    LLMConfig config = LLMConfig.defaultConfig(LLMConfig.Provider.OPENAI, "test-key");
    RAGService ragService =
        new RAGService(client, COLLECTION, embeddingModel, chatModel, costTracker, config);

    // Act
    ragService.query("what is vector search?", CacheType.NONE);

    // Assert: chat model was called with tool specifications (standard path)
    verify(chatModel).chat(any(dev.langchain4j.model.chat.request.ChatRequest.class));
  }

  @Nested
  @Tag("integration")
  @DisplayName("Blob storage and retrieval")
  class BlobRoundTrip {

    @Test
    @DisplayName("batch-upserted blob is retrievable via getBlob")
    void batchUpsertedBlobIsRetrievable() {
      // Blobs were batch-upserted in setUp(). Verify retrieval.
      Optional<byte[]> blob1 = client.getBlob(COLLECTION, "img-1");
      assertThat(blob1).isPresent();
      assertThat(new String(blob1.get())).isEqualTo("fake-png-data-1");

      Optional<byte[]> blob2 = client.getBlob(COLLECTION, "img-2");
      assertThat(blob2).isPresent();
      assertThat(new String(blob2.get())).isEqualTo("fake-png-data-2");
    }

    @Test
    @DisplayName("individually upserted blob is retrievable (demo ingestion pattern)")
    void individuallyUpsertedBlobIsRetrievable() {
      byte[] imageBytes = "individual-image-data".getBytes();
      String blob = Base64.getEncoder().encodeToString(imageBytes);
      float[] v = {0.0f, 0.0f, 0.0f, 1.0f};
      client.upsertDocuments(
          COLLECTION,
          List.of(
              new DocumentPayload(
                  "img-individual",
                  v,
                  "Individually upserted image",
                  Map.of("type", "IMAGE", "page", 9),
                  blob)));
      client.commit(COLLECTION);

      Optional<byte[]> retrieved = client.getBlob(COLLECTION, "img-individual");
      assertThat(retrieved).isPresent();
      assertThat(new String(retrieved.get())).isEqualTo("individual-image-data");
    }

    @Test
    @DisplayName("text-only document returns empty blob")
    void textDocumentHasNoBlob() {
      Optional<byte[]> blob = client.getBlob(COLLECTION, "txt-1");
      // Text documents may exist in the text index but with null blob
      assertThat(blob)
          .satisfiesAnyOf(b -> assertThat(b).isEmpty(), b -> assertThat(b.get()).isEmpty());
    }

    @Test
    @DisplayName("blobs survive collection reset and re-ingestion")
    void blobsSurviveCollectionReset() {
      // Delete and recreate collection (same pattern as ServiceFactory.resetCollection)
      client.deleteCollection(COLLECTION);
      client.createCollection(COLLECTION, DIM, "COSINE", "FLAT", null);

      // Re-upsert a single image
      byte[] newData = "after-reset-image".getBytes();
      String blob = Base64.getEncoder().encodeToString(newData);
      float[] v = {1.0f, 0.0f, 0.0f, 0.0f};
      client.upsertDocuments(
          COLLECTION,
          List.of(
              new DocumentPayload(
                  "img-reset", v, "Image after reset", Map.of("type", "IMAGE"), blob)));
      client.commit(COLLECTION);

      Optional<byte[]> retrieved = client.getBlob(COLLECTION, "img-reset");
      assertThat(retrieved).isPresent();
      assertThat(new String(retrieved.get())).isEqualTo("after-reset-image");
    }
  }

  @Nested
  @Tag("integration")
  @DisplayName("Cache admission policy integration")
  class AdmissionPolicyIntegration {

    @Test
    @DisplayName("RAGService does not cache LLM refusal responses")
    void ragServiceDoesNotCacheRefusals() {
      float[] queryVector = {0.9f, 0.1f, 0.0f, 0.0f};
      when(embeddingModel.embed(anyString()))
          .thenReturn(new Response<>(Embedding.from(queryVector)));
      when(chatModel.chat(any(dev.langchain4j.model.chat.request.ChatRequest.class)))
          .thenReturn(
              ChatResponse.builder()
                  .aiMessage(
                      dev.langchain4j.data.message.AiMessage.from(
                          "I'm sorry, but I can't answer that question."))
                  .build());
      when(costTracker.countTokens(anyString())).thenReturn(12);
      when(costTracker.calculateCost(any(), anyString(), anyInt())).thenReturn(0.001);

      LLMConfig config = LLMConfig.defaultConfig(LLMConfig.Provider.OPENAI, "test-key");
      RAGService ragService =
          new RAGService(client, COLLECTION, embeddingModel, chatModel, costTracker, config);

      ChatMessage first = ragService.query("what is the answer?", CacheType.LOCAL);
      ChatMessage second = ragService.query("what is the answer?", CacheType.LOCAL);

      assertThat(first.fromCache()).isFalse();
      assertThat(second.fromCache()).isFalse();
      verify(chatModel, times(2)).chat(any(dev.langchain4j.model.chat.request.ChatRequest.class));
    }
  }
}
