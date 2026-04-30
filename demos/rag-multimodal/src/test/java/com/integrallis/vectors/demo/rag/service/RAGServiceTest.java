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
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.integrallis.vectors.demo.rag.model.CacheType;
import com.integrallis.vectors.demo.rag.model.ChatMessage;
import com.integrallis.vectors.demo.rag.model.LLMConfig;
import com.integrallis.vectors.server.client.SearchHit;
import com.integrallis.vectors.server.client.VectorsServerClient;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

@ExtendWith(MockitoExtension.class)
@Tag("unit")
class RAGServiceTest {

  @Nested
  @DisplayName("query — tool-based architecture")
  class ToolBasedQueryTest {

    @Mock VectorsServerClient client;
    @Mock EmbeddingModel embeddingModel;
    @Mock ChatModel chatModel;
    @Mock CostTracker costTracker;

    private RAGService ragService;

    @BeforeEach
    void setUp() {
      LLMConfig config = LLMConfig.defaultConfig(LLMConfig.Provider.OPENAI, "test-key");
      ragService =
          new RAGService(client, "test-collection", embeddingModel, chatModel, costTracker, config);
    }

    @Test
    @DisplayName("all queries use hybridSearch — no hardcoded image shortcut")
    void allQueriesUseHybridSearch() {
      // Arrange
      float[] dummyVector = new float[] {0.1f, 0.2f, 0.3f};
      when(embeddingModel.embed(anyString()))
          .thenReturn(new Response<>(Embedding.from(dummyVector)));

      SearchHit textHit =
          new SearchHit("txt-1", 0.85f, null, "Some text content", Map.of("type", "TEXT", "page", 1));
      when(client.hybridSearch(
              eq("test-collection"), any(float[].class), anyString(), eq(40), eq("RRF")))
          .thenReturn(List.of(textHit));

      dev.langchain4j.data.message.AiMessage aiMessage =
          dev.langchain4j.data.message.AiMessage.from("The answer is...");
      ChatResponse chatResponse = ChatResponse.builder().aiMessage(aiMessage).build();
      when(chatModel.chat(any(ChatRequest.class))).thenReturn(chatResponse);

      when(costTracker.countTokens(anyString())).thenReturn(10);
      when(costTracker.calculateCost(any(), anyString(), anyInt())).thenReturn(0.001);

      // Act — even a "list images" query now goes through hybridSearch
      ragService.query("list all images", CacheType.NONE);

      // Assert: hybridSearch is always used, no blob fetches for inline images
      verify(client)
          .hybridSearch(eq("test-collection"), any(float[].class), anyString(), eq(40), eq("RRF"));
      verify(client, never()).getBlob(anyString(), anyString());
    }

    @Test
    @DisplayName("text query — no blobs fetched, LLM gets text-only prompt with tools")
    void textQuerySendsTextOnlyWithTools() {
      // Arrange
      float[] dummyVector = new float[] {0.1f, 0.2f, 0.3f};
      when(embeddingModel.embed(anyString()))
          .thenReturn(new Response<>(Embedding.from(dummyVector)));

      SearchHit textHit =
          new SearchHit("txt-1", 0.85f, null, "Some text content", Map.of("type", "TEXT", "page", 1));
      when(client.hybridSearch(
              eq("test-collection"), any(float[].class), anyString(), eq(40), eq("RRF")))
          .thenReturn(List.of(textHit));

      dev.langchain4j.data.message.AiMessage aiMessage =
          dev.langchain4j.data.message.AiMessage.from("The document discusses...");
      ChatResponse chatResponse = ChatResponse.builder().aiMessage(aiMessage).build();
      when(chatModel.chat(any(ChatRequest.class))).thenReturn(chatResponse);

      when(costTracker.countTokens(anyString())).thenReturn(10);
      when(costTracker.calculateCost(any(), anyString(), anyInt())).thenReturn(0.001);

      // Act
      ChatMessage result = ragService.query("what is the main topic?", CacheType.NONE);

      // Assert: LLM was called (with tools available)
      verify(chatModel).chat(any(ChatRequest.class));
      // Assert: no blobs fetched (text-only prompt)
      verify(client, never()).getBlob(anyString(), anyString());
      // Assert: response comes from LLM
      assertThat(result.content()).isEqualTo("The document discusses...");
    }

    @Test
    @DisplayName("LLM tool call for list_images — executes list_images tool")
    void llmListImageToolCallExecutesListImages() {
      // Arrange
      float[] dummyVector = new float[] {0.1f, 0.2f, 0.3f};
      when(embeddingModel.embed(anyString()))
          .thenReturn(new Response<>(Embedding.from(dummyVector)));

      when(client.hybridSearch(
              eq("test-collection"), any(float[].class), anyString(), eq(40), eq("RRF")))
          .thenReturn(List.of());

      // LLM responds with a tool call for list_images
      dev.langchain4j.agent.tool.ToolExecutionRequest toolCall =
          dev.langchain4j.agent.tool.ToolExecutionRequest.builder()
              .name("list_images")
              .arguments("{}")
              .build();
      dev.langchain4j.data.message.AiMessage aiMessage =
          dev.langchain4j.data.message.AiMessage.from(toolCall);
      ChatResponse chatResponse = ChatResponse.builder().aiMessage(aiMessage).build();
      when(chatModel.chat(any(ChatRequest.class))).thenReturn(chatResponse);

      // list_images tool searches for IMAGE chunks
      SearchHit img1 =
          new SearchHit(
              "img-1",
              0.95f,
              null,
              "Revenue pie chart — page 1 (800x600 PNG)",
              Map.of("type", "IMAGE", "page", 1));
      SearchHit img2 =
          new SearchHit(
              "img-2",
              0.90f,
              null,
              "Stock price chart — page 2 (900x500 PNG)",
              Map.of("type", "IMAGE", "page", 2));
      when(client.search(
              eq("test-collection"),
              any(float[].class),
              eq(200),
              isNull(),
              eq(Map.of("field", "type", "eq", "IMAGE"))))
          .thenReturn(List.of(img1, img2));

      // Blob availability check — both images have blobs
      when(client.getBlob("test-collection", "img-1"))
          .thenReturn(java.util.Optional.of("fake-blob-1".getBytes()));
      when(client.getBlob("test-collection", "img-2"))
          .thenReturn(java.util.Optional.of("fake-blob-2".getBytes()));

      when(costTracker.countTokens(anyString())).thenReturn(10);

      // Act
      ChatMessage result = ragService.query("list the images", CacheType.NONE);

      // Assert: deterministic listing returned (no LLM formatting)
      assertThat(result.content()).contains("2 image(s)");
      assertThat(result.content()).contains("Revenue pie chart");
      assertThat(result.content()).contains("Stock price chart");
      assertThat(result.costUsd()).isEqualTo(0.0);
    }
  }

  @Nested
  @DisplayName("query — semantic cache behavior")
  class SemanticCacheTest {

    @Mock VectorsServerClient client;
    @Mock EmbeddingModel embeddingModel;
    @Mock ChatModel chatModel;
    @Mock CostTracker costTracker;

    private RAGService ragService;

    @BeforeEach
    void setUp() {
      LLMConfig config = LLMConfig.defaultConfig(LLMConfig.Provider.OPENAI, "test-key");
      ragService =
          new RAGService(client, "test-collection", embeddingModel, chatModel, costTracker, config);
    }

    /** Stubs the full LLM round-trip with a default embedding vector. */
    private void stubLlmRoundTrip() {
      float[] dummyVector = new float[] {0.1f, 0.2f, 0.3f};
      when(embeddingModel.embed(anyString()))
          .thenReturn(new Response<>(Embedding.from(dummyVector)));
      stubSearchAndLlm();
    }

    /** Stubs only the search + LLM parts (caller provides embed stubs separately). */
    private void stubSearchAndLlm() {
      SearchHit textHit =
          new SearchHit("txt-1", 0.85f, null, "Some text content", Map.of("type", "TEXT", "page", 1));
      when(client.hybridSearch(
              eq("test-collection"), any(float[].class), anyString(), eq(40), eq("RRF")))
          .thenReturn(List.of(textHit));

      dev.langchain4j.data.message.AiMessage aiMessage =
          dev.langchain4j.data.message.AiMessage.from("The LLM answer is 42.");
      ChatResponse chatResponse = ChatResponse.builder().aiMessage(aiMessage).build();
      when(chatModel.chat(any(ChatRequest.class))).thenReturn(chatResponse);

      when(costTracker.countTokens(anyString())).thenReturn(15);
      when(costTracker.calculateCost(any(), anyString(), anyInt())).thenReturn(0.005);
    }

    @Test
    @DisplayName("first query with LOCAL cache: fromCache=false, cost > 0")
    void firstQueryIsNotFromCache() {
      stubLlmRoundTrip();

      ChatMessage result = ragService.query("what is the answer?", CacheType.LOCAL);

      assertThat(result.fromCache()).isFalse();
      assertThat(result.costUsd()).isGreaterThan(0.0);
      assertThat(result.content()).isEqualTo("The LLM answer is 42.");
    }

    @Test
    @DisplayName("second identical query with LOCAL cache: fromCache=true, cost=$0")
    void cachedResponseHasZeroCostAndFromCacheTrue() {
      stubLlmRoundTrip();

      // First call — populates cache
      ChatMessage first = ragService.query("what is the answer?", CacheType.LOCAL);
      assertThat(first.fromCache()).isFalse();

      // Second call — same embedding, cosine=1.0 ≥ 0.9 → cache hit
      ChatMessage cached = ragService.query("what is the answer?", CacheType.LOCAL);

      assertThat(cached.fromCache()).as("cached response must be flagged fromCache").isTrue();
      assertThat(cached.costUsd()).as("cached response must have zero cost").isEqualTo(0.0);
      assertThat(cached.content())
          .as("cached content must match original LLM response")
          .isEqualTo("The LLM answer is 42.");
    }

    @Test
    @DisplayName("cache hit still embeds (for cosine comparison) but skips LLM")
    void cacheHitStillEmbedsButSkipsLlm() {
      stubLlmRoundTrip();

      ragService.query("what is the answer?", CacheType.LOCAL);
      ragService.query("what is the answer?", CacheType.LOCAL);

      // Embedding called TWICE (semantic cache needs vector for each query)
      verify(embeddingModel, times(2)).embed(anyString());
      // LLM called only ONCE (second query hits cache)
      verify(chatModel, times(1)).chat(any(ChatRequest.class));
    }

    @Test
    @DisplayName("CacheType.NONE never caches, always calls LLM")
    void noCacheModeAlwaysCallsLlm() {
      stubLlmRoundTrip();

      ChatMessage first = ragService.query("what is the answer?", CacheType.NONE);
      ChatMessage second = ragService.query("what is the answer?", CacheType.NONE);

      assertThat(first.fromCache()).isFalse();
      assertThat(second.fromCache()).isFalse();
      assertThat(second.costUsd()).isGreaterThan(0.0);

      // LLM called twice
      verify(chatModel, times(2)).chat(any(ChatRequest.class));
    }

    @Test
    @DisplayName("orthogonal embeddings are cached independently")
    void orthogonalEmbeddingsCachedIndependently() {
      // Different queries → orthogonal embeddings → cosine=0 → no cache match
      when(embeddingModel.embed("question one"))
          .thenReturn(new Response<>(Embedding.from(new float[] {1f, 0f, 0f})));
      when(embeddingModel.embed("question two"))
          .thenReturn(new Response<>(Embedding.from(new float[] {0f, 1f, 0f})));
      stubSearchAndLlm();

      ragService.query("question one", CacheType.LOCAL);
      ragService.query("question two", CacheType.LOCAL);

      // Both are cache misses — LLM called twice
      verify(chatModel, times(2)).chat(any(ChatRequest.class));

      // Re-ask question one — cosine(same embedding, cached) = 1.0 → hit
      ChatMessage cachedOne = ragService.query("question one", CacheType.LOCAL);
      assertThat(cachedOne.fromCache()).isTrue();
      assertThat(cachedOne.costUsd()).isEqualTo(0.0);

      // LLM still only called twice total
      verify(chatModel, times(2)).chat(any(ChatRequest.class));
    }

    @Test
    @DisplayName("semantically similar queries hit the cache (cosine > threshold)")
    void semanticallySimilarQueriesHitCache() {
      // Two vectors with cosine ≈ 0.80 (above default threshold 0.75)
      // cos([1,0,0], [0.8,0.6,0]) = 0.8/1.0 = 0.80
      when(embeddingModel.embed("what is this document about?"))
          .thenReturn(new Response<>(Embedding.from(new float[] {1.0f, 0.0f, 0.0f})));
      when(embeddingModel.embed("what is this PDF about?"))
          .thenReturn(new Response<>(Embedding.from(new float[] {0.8f, 0.6f, 0.0f})));
      stubSearchAndLlm();

      ChatMessage first = ragService.query("what is this document about?", CacheType.LOCAL);
      assertThat(first.fromCache()).isFalse();

      // Second query — similar embedding → cosine ≈ 0.80 ≥ 0.75 → cache hit
      ChatMessage second = ragService.query("what is this PDF about?", CacheType.LOCAL);
      assertThat(second.fromCache()).as("similar query should hit semantic cache").isTrue();
      assertThat(second.costUsd()).isEqualTo(0.0);

      // LLM called only once
      verify(chatModel, times(1)).chat(any(ChatRequest.class));
    }

    @Test
    @DisplayName("dissimilar queries do not hit cache (cosine < threshold)")
    void dissimilarQueriesMissCache() {
      // Two orthogonal vectors — cosine = 0 (well below threshold)
      when(embeddingModel.embed("question about images"))
          .thenReturn(new Response<>(Embedding.from(new float[] {1.0f, 0.0f, 0.0f})));
      when(embeddingModel.embed("completely different topic"))
          .thenReturn(new Response<>(Embedding.from(new float[] {0.0f, 1.0f, 0.0f})));
      stubSearchAndLlm();

      ragService.query("question about images", CacheType.LOCAL);
      ChatMessage second = ragService.query("completely different topic", CacheType.LOCAL);

      assertThat(second.fromCache()).as("dissimilar query should not hit cache").isFalse();
      verify(chatModel, times(2)).chat(any(ChatRequest.class));
    }

    @Test
    @DisplayName("setSimilarityThreshold adjusts cache matching sensitivity")
    void thresholdAdjustsMatching() {
      // Vectors with cosine ≈ 0.80
      // cos([1,0,0], [0.8,0.6,0]) = 0.8/1.0 = 0.80
      when(embeddingModel.embed("query A"))
          .thenReturn(new Response<>(Embedding.from(new float[] {1.0f, 0.0f, 0.0f})));
      when(embeddingModel.embed("query B"))
          .thenReturn(new Response<>(Embedding.from(new float[] {0.8f, 0.6f, 0.0f})));
      stubSearchAndLlm();

      // Set threshold above the cosine similarity → no match
      ragService.setSimilarityThreshold(0.90);
      ragService.query("query A", CacheType.LOCAL);
      ChatMessage miss = ragService.query("query B", CacheType.LOCAL);
      assertThat(miss.fromCache()).as("high threshold should prevent match").isFalse();

      // Lower threshold → match
      ragService.setSimilarityThreshold(0.75);
      ChatMessage hit = ragService.query("query B", CacheType.LOCAL);
      assertThat(hit.fromCache()).as("lowered threshold should allow match").isTrue();
    }

    @Test
    @DisplayName("cached response preserves model name from config")
    void cachedResponsePreservesModelName() {
      stubLlmRoundTrip();

      ChatMessage first = ragService.query("what is the answer?", CacheType.LOCAL);
      ChatMessage cached = ragService.query("what is the answer?", CacheType.LOCAL);

      assertThat(cached.model())
          .as("cached response must carry the same model name")
          .isEqualTo(first.model());
    }

    @Test
    @DisplayName("cached response has token count from the cached text")
    void cachedResponseHasTokenCount() {
      stubLlmRoundTrip();

      ragService.query("what is the answer?", CacheType.LOCAL);

      // On cache hit, costTracker.countTokens() is called with the cached text
      ChatMessage cached = ragService.query("what is the answer?", CacheType.LOCAL);

      assertThat(cached.tokenCount())
          .as("cached response must report token count")
          .isGreaterThan(0);
    }
  }

  @Nested
  @DisplayName("cosineSimilarity")
  class CosineSimilarityTest {

    @Test
    @DisplayName("identical vectors have similarity 1.0")
    void identicalVectors() {
      float[] v = {1f, 2f, 3f};
      assertThat(RAGService.cosineSimilarity(v, v)).isEqualTo(1.0f);
    }

    @Test
    @DisplayName("orthogonal vectors have similarity 0.0")
    void orthogonalVectors() {
      assertThat(RAGService.cosineSimilarity(new float[] {1f, 0f}, new float[] {0f, 1f}))
          .isEqualTo(0.0f);
    }

    @Test
    @DisplayName("opposite vectors have similarity -1.0")
    void oppositeVectors() {
      assertThat(RAGService.cosineSimilarity(new float[] {1f, 0f}, new float[] {-1f, 0f}))
          .isEqualTo(-1.0f);
    }

    @Test
    @DisplayName("zero vector returns 0.0")
    void zeroVector() {
      assertThat(RAGService.cosineSimilarity(new float[] {0f, 0f}, new float[] {1f, 0f}))
          .isEqualTo(0.0f);
    }
  }
}
