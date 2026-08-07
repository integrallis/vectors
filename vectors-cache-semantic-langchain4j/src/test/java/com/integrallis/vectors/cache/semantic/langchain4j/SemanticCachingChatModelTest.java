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
package com.integrallis.vectors.cache.semantic.langchain4j;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.vectors.cache.CacheStats;
import com.integrallis.vectors.cache.SemanticCache;
import com.integrallis.vectors.cache.langchain4j.CachingChatModel;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

class SemanticCachingChatModelTest {

  /** In-memory similarity cache: cosine over stored embeddings, honouring a threshold. */
  private static class InMemorySemanticCache implements SemanticCache<String> {
    private final Map<String, float[]> embeddings = new LinkedHashMap<>();
    private final Map<String, String> values = new LinkedHashMap<>();
    private final Map<String, Map<String, String>> attributes = new LinkedHashMap<>();
    private final double threshold;
    private int hits;
    private int misses;

    InMemorySemanticCache(double threshold) {
      this.threshold = threshold;
    }

    @Override
    public Optional<String> get(String key) {
      return Optional.ofNullable(values.get(key));
    }

    @Override
    public boolean supportsAttributes() {
      return true;
    }

    @Override
    public void put(String key, float[] embedding, String value) {
      put(key, embedding, value, Map.of());
    }

    @Override
    public void put(
        String key, float[] embedding, String value, Map<String, String> entryAttributes) {
      embeddings.put(key, embedding.clone());
      values.put(key, value);
      attributes.put(key, Map.copyOf(entryAttributes));
    }

    @Override
    public void putAll(Collection<Entry<String>> entries) {
      entries.forEach(e -> put(e.key(), e.embedding(), e.value(), e.attributes()));
    }

    @Override
    public Optional<Hit<String>> lookup(float[] query) {
      String bestKey = null;
      double best = -1;
      for (Map.Entry<String, float[]> candidate : embeddings.entrySet()) {
        double score = cosine(query, candidate.getValue());
        if (score > best) {
          best = score;
          bestKey = candidate.getKey();
        }
      }
      if (bestKey == null || best < threshold) {
        misses++;
        return Optional.empty();
      }
      hits++;
      return Optional.of(new Hit<>(bestKey, values.get(bestKey), best, attributes.get(bestKey)));
    }

    @Override
    public List<Hit<String>> lookupTopK(float[] query, int k) {
      List<Hit<String>> ranked = new ArrayList<>();
      for (Map.Entry<String, float[]> candidate : embeddings.entrySet()) {
        double score = cosine(query, candidate.getValue());
        if (score >= threshold) {
          String key = candidate.getKey();
          ranked.add(new Hit<>(key, values.get(key), score, attributes.get(key)));
        }
      }
      ranked.sort(Comparator.comparingDouble(Hit<String>::score).reversed());
      List<Hit<String>> top = ranked.subList(0, Math.min(k, ranked.size()));
      if (top.isEmpty()) {
        misses++;
      } else {
        hits++;
      }
      return List.copyOf(top);
    }

    @Override
    public void invalidate(String key) {
      embeddings.remove(key);
      values.remove(key);
      attributes.remove(key);
    }

    @Override
    public void invalidateAll() {
      embeddings.clear();
      values.clear();
      attributes.clear();
    }

    @Override
    public CacheStats stats() {
      return new CacheStats(hits, misses, 0, 0, values.size());
    }

    @Override
    public double threshold() {
      return threshold;
    }

    @Override
    public com.integrallis.vectors.cache.CacheAdmissionPolicy<String> admissionPolicy() {
      return value -> true;
    }

    int size() {
      return values.size();
    }

    private static double cosine(float[] a, float[] b) {
      double dot = 0;
      double na = 0;
      double nb = 0;
      for (int i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
      }
      return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-12);
    }
  }

  /** Embeds by topic-word overlap, so paraphrases land close together and topics do not. */
  private static final class TopicEmbeddingModel implements EmbeddingModel {
    private static final List<String> AXES =
        List.of("password", "reset", "login", "weather", "forecast", "rain");

    @Override
    public Response<List<Embedding>> embedAll(List<TextSegment> segments) {
      return Response.from(segments.stream().map(s -> vector(s.text())).toList());
    }

    private static Embedding vector(String text) {
      String lower = text.toLowerCase(java.util.Locale.ROOT);
      float[] v = new float[AXES.size()];
      for (int i = 0; i < AXES.size(); i++) {
        v[i] = lower.contains(AXES.get(i)) ? 1.0f : 0.0f;
      }
      return Embedding.from(v);
    }
  }

  /** Counts calls and echoes. */
  private static final class CountingChatModel implements ChatModel {
    final AtomicInteger calls = new AtomicInteger();

    @Override
    public ChatResponse doChat(ChatRequest request) {
      calls.incrementAndGet();
      return ChatResponse.builder().aiMessage(AiMessage.from("answer")).build();
    }
  }

  private static ChatRequest ask(String text) {
    return ChatRequest.builder().messages(UserMessage.from(text)).build();
  }

  private static ChatRequest at(String text, double temperature) {
    return ChatRequest.builder().messages(UserMessage.from(text)).temperature(temperature).build();
  }

  // "reset my password" embeds to {password, reset} and "I forgot my password" to {password},
  // which are 0.707 apart. A threshold either side of that separates hit from miss.
  private static final String QUERY = "how do I reset my password";
  private static final String PARAPHRASE = "I forgot my password";

  @Test
  void servesANearDuplicateRequestFromCache() {
    CountingChatModel fake = new CountingChatModel();
    SemanticCachingChatModel model =
        new SemanticCachingChatModel(
            fake, new TopicEmbeddingModel(), new InMemorySemanticCache(0.60));

    model.chat(ask(QUERY));
    model.chat(ask(PARAPHRASE));

    assertThat(fake.calls.get()).isEqualTo(1);
  }

  @Test
  void anExactKeyCacheWouldHaveMissedThatParaphrase() {
    assertThat(CachingChatModel.defaultRequestKey(ask(QUERY)))
        .isNotEqualTo(CachingChatModel.defaultRequestKey(ask(PARAPHRASE)));
  }

  @Test
  void callsTheDelegateForAnUnrelatedRequest() {
    CountingChatModel fake = new CountingChatModel();
    SemanticCachingChatModel model =
        new SemanticCachingChatModel(
            fake, new TopicEmbeddingModel(), new InMemorySemanticCache(0.60));

    model.chat(ask(QUERY));
    model.chat(ask("what is the weather forecast"));

    assertThat(fake.calls.get()).isEqualTo(2);
  }

  @Test
  void respectsTheCachesOwnSimilarityThreshold() {
    CountingChatModel fake = new CountingChatModel();
    SemanticCachingChatModel model =
        new SemanticCachingChatModel(
            fake, new TopicEmbeddingModel(), new InMemorySemanticCache(0.80));

    model.chat(ask(QUERY));
    model.chat(ask(PARAPHRASE));

    assertThat(fake.calls.get()).isEqualTo(2);
  }

  @Test
  void neverServesAResponseGeneratedUnderDifferentParameters() {
    // An embedding captures the text alone, so identical text at a different temperature must
    // miss rather than hand back output the caller did not ask for.
    CountingChatModel fake = new CountingChatModel();
    SemanticCachingChatModel model =
        new SemanticCachingChatModel(
            fake, new TopicEmbeddingModel(), new InMemorySemanticCache(0.60));

    model.chat(at(QUERY, 0.0));
    model.chat(at(QUERY, 0.9));

    assertThat(fake.calls.get()).isEqualTo(2);
  }

  @Test
  void servesAFartherEntryWhenTheNearestOneCarriesTheWrongParameters() {
    // The nearest entry is a verbatim repeat at the wrong temperature; a usable paraphrase sits
    // farther out. Post-filtering the single nearest result would report a miss here.
    CountingChatModel fake = new CountingChatModel();
    SemanticCachingChatModel model =
        new SemanticCachingChatModel(
            fake, new TopicEmbeddingModel(), new InMemorySemanticCache(0.60));

    model.chat(at(QUERY, 0.9));
    model.chat(at(PARAPHRASE, 0.0));
    int afterWarmup = fake.calls.get();

    model.chat(at(QUERY, 0.0));

    assertThat(fake.calls.get()).isEqualTo(afterWarmup);
  }

  @Test
  void refusesACacheThatCannotStoreAttributes() {
    // Without attributes the decorator cannot tell which parameters produced an entry, so it
    // would silently serve completions across temperatures. Fail at wiring time.
    SemanticCache<String> noAttributes =
        new InMemorySemanticCache(0.60) {
          @Override
          public boolean supportsAttributes() {
            return false;
          }
        };

    assertThatThrownBy(
            () ->
                new SemanticCachingChatModel(
                    new CountingChatModel(), new TopicEmbeddingModel(), noAttributes))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("does not store entry attributes");
  }

  @Test
  void doesNotCacheAResponseCarryingToolExecutionRequests() {
    ChatModel tooling =
        new ChatModel() {
          @Override
          public ChatResponse doChat(ChatRequest request) {
            return ChatResponse.builder()
                .aiMessage(
                    AiMessage.from(
                        dev.langchain4j.agent.tool.ToolExecutionRequest.builder()
                            .id("1")
                            .name("resetPassword")
                            .arguments("{}")
                            .build()))
                .build();
          }
        };
    InMemorySemanticCache cache = new InMemorySemanticCache(0.60);

    new SemanticCachingChatModel(tooling, new TopicEmbeddingModel(), cache).chat(ask(QUERY));

    assertThat(cache.size()).isZero();
  }

  @Test
  void exposesTheDelegateAndCacheForInspection() {
    CountingChatModel fake = new CountingChatModel();
    InMemorySemanticCache cache = new InMemorySemanticCache(0.60);
    SemanticCachingChatModel model =
        new SemanticCachingChatModel(fake, new TopicEmbeddingModel(), cache);

    assertThat(model.delegate()).isSameAs(fake);
    assertThat(model.cache()).isSameAs(cache);
  }
}
