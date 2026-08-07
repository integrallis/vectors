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
package com.integrallis.vectors.cache.semantic.springai;

import com.integrallis.vectors.cache.CacheStats;
import com.integrallis.vectors.cache.SemanticCache;
import com.integrallis.vectors.cache.springai.CachingChatModel;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.chat.prompt.Prompt;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

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
    public void put(String key, float[] embedding, String value, Map<String, String> entryAttributes) {
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
  private static final class TopicEmbeddingModel
      implements org.springframework.ai.embedding.EmbeddingModel {
    private static final List<String> AXES =
        List.of("password", "reset", "login", "weather", "forecast", "rain");

    @Override
    public float[] embed(String text) {
      String lower = text.toLowerCase(java.util.Locale.ROOT);
      float[] vector = new float[AXES.size()];
      for (int i = 0; i < AXES.size(); i++) {
        vector[i] = lower.contains(AXES.get(i)) ? 1.0f : 0.0f;
      }
      return vector;
    }

    @Override
    public float[] embed(org.springframework.ai.document.Document document) {
      return embed(document.getText());
    }

    @Override
    public org.springframework.ai.embedding.EmbeddingResponse call(
        org.springframework.ai.embedding.EmbeddingRequest request) {
      List<org.springframework.ai.embedding.Embedding> out = new ArrayList<>();
      List<String> inputs = request.getInstructions();
      for (int i = 0; i < inputs.size(); i++) {
        out.add(new org.springframework.ai.embedding.Embedding(embed(inputs.get(i)), i));
      }
      return new org.springframework.ai.embedding.EmbeddingResponse(out);
    }
  }

  // "reset my password" embeds to {password, reset} and "I forgot my password" to {password},
  // which are 0.707 apart. A threshold either side of that separates hit from miss.
  private static final String QUERY = "how do I reset my password";
  private static final String PARAPHRASE = "I forgot my password";

  @Test
  void servesANearDuplicateQueryFromCache() {
    // The entire point: a paraphrase must hit, where an exact-key cache would miss.
    FakeChatModel fake = new FakeChatModel();
    SemanticCachingChatModel model =
        new SemanticCachingChatModel(
            fake, new TopicEmbeddingModel(), new InMemorySemanticCache(0.60));

    model.call(new Prompt(QUERY));
    model.call(new Prompt(PARAPHRASE));

    assertThat(fake.calls.get()).isEqualTo(1);
  }

  @Test
  void anExactKeyCacheWouldHaveMissedThatParaphrase() {
    // Guards the distinction from CachingChatModel: these prompts share no normalized key.
    assertThat(CachingChatModel.defaultPromptKey(new Prompt(QUERY)))
        .isNotEqualTo(CachingChatModel.defaultPromptKey(new Prompt(PARAPHRASE)));
  }

  @Test
  void callsTheDelegateForAnUnrelatedQuery() {
    FakeChatModel fake = new FakeChatModel();
    SemanticCachingChatModel model =
        new SemanticCachingChatModel(
            fake, new TopicEmbeddingModel(), new InMemorySemanticCache(0.80));

    model.call(new Prompt("how do I reset my password"));
    model.call(new Prompt("what is the weather forecast"));

    assertThat(fake.calls.get()).isEqualTo(2);
  }

  @Test
  void neverServesAResponseGeneratedUnderDifferentOptions() {
    // lookup() searches by embedding alone, so without a guard a hit could return text produced
    // at a different temperature or token limit. Identical text, different options, must miss.
    FakeChatModel fake = new FakeChatModel();
    SemanticCachingChatModel model =
        new SemanticCachingChatModel(
            fake, new TopicEmbeddingModel(), new InMemorySemanticCache(0.80));

    model.call(new Prompt("reset password", ChatOptions.builder().temperature(0.0).build()));
    model.call(new Prompt("reset password", ChatOptions.builder().temperature(0.9).build()));

    assertThat(fake.calls.get()).isEqualTo(2);
  }

  @Test
  void servesAFartherEntryWhenTheNearestOneCarriesTheWrongOptions() {
    // The entry nearest the query is a verbatim repeat at the wrong temperature; a usable
    // paraphrase sits farther out. Filtering the single nearest result after the search would
    // report a miss here, so the filter has to run inside the search.
    FakeChatModel fake = new FakeChatModel();
    InMemorySemanticCache cache = new InMemorySemanticCache(0.80);
    SemanticCachingChatModel model =
        new SemanticCachingChatModel(fake, new TopicEmbeddingModel(), cache);

    // {password, reset}, cosine 1.0 against the query, but temperature 0.9.
    model.call(new Prompt("reset password", ChatOptions.builder().temperature(0.9).build()));
    // {password, reset, login}, cosine 0.816 against the query, at the temperature we want.
    model.call(
        new Prompt("reset password login", ChatOptions.builder().temperature(0.0).build()));
    int afterWarmup = fake.calls.get();

    ChatResponse response =
        model.call(new Prompt("reset password", ChatOptions.builder().temperature(0.0).build()));

    assertThat(fake.calls.get()).isEqualTo(afterWarmup);
    assertThat(response.getResult().getOutput().getText()).isEqualTo("echo: reset password login");
  }

  @Test
  void identicalOptionsBuiltSeparatelyProduceTheSameSignature() {
    // DefaultChatOptions inherits the identity toString(), so a signature taken from it differs on
    // every call and no prompt carrying options ever hits the cache. The signature reads values.
    FakeChatModel fake = new FakeChatModel();
    InMemorySemanticCache cache = new InMemorySemanticCache(0.80);
    SemanticCachingChatModel model =
        new SemanticCachingChatModel(fake, new TopicEmbeddingModel(), cache);

    model.call(new Prompt("reset password", ChatOptions.builder().temperature(0.0).build()));
    model.call(new Prompt("reset password", ChatOptions.builder().temperature(0.0).build()));

    assertThat(fake.calls.get()).isEqualTo(1);
  }

  @Test
  void refusesACacheThatCannotStoreAttributes() {
    // Without attributes the decorator cannot tell which options produced an entry, so it would
    // silently serve completions across temperatures. Fail at wiring time, not mid-conversation.
    SemanticCache<String> noAttributes =
        new InMemorySemanticCache(0.80) {
          @Override
          public boolean supportsAttributes() {
            return false;
          }
        };

    assertThatThrownBy(
            () ->
                new SemanticCachingChatModel(
                    new FakeChatModel(), new TopicEmbeddingModel(), noAttributes))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("does not store entry attributes");
  }

  @Test
  void doesNotCacheAResponseCarryingToolCalls() {
    // A tool call is an instruction to go and do something, not an answer. Replaying one from
    // cache would skip the tool and hand back a stale result.
    ToolCallingChatModel fake = new ToolCallingChatModel();
    InMemorySemanticCache cache = new InMemorySemanticCache(0.80);
    SemanticCachingChatModel model =
        new SemanticCachingChatModel(fake, new TopicEmbeddingModel(), cache);

    model.call(new Prompt("reset password"));

    assertThat(cache.size()).isZero();
  }

  @Test
  void streamingDelegatesAndIsNotCached() {
    FakeChatModel fake = new FakeChatModel();
    InMemorySemanticCache cache = new InMemorySemanticCache(0.80);
    SemanticCachingChatModel model =
        new SemanticCachingChatModel(fake, new TopicEmbeddingModel(), cache);

    model.stream(new Prompt("reset password")).blockLast();

    assertThat(cache.size()).isZero();
  }

  @Test
  void respectsTheCachesOwnSimilarityThreshold() {
    // The same pair that hits at 0.60 must miss at 0.80: the cache's threshold decides, not us.
    FakeChatModel fake = new FakeChatModel();
    SemanticCachingChatModel model =
        new SemanticCachingChatModel(
            fake, new TopicEmbeddingModel(), new InMemorySemanticCache(0.80));

    model.call(new Prompt(QUERY));
    model.call(new Prompt(PARAPHRASE));

    assertThat(fake.calls.get()).isEqualTo(2);
  }

  @Test
  void exposesTheDelegateAndCacheForInspection() {
    FakeChatModel fake = new FakeChatModel();
    InMemorySemanticCache cache = new InMemorySemanticCache(0.80);
    SemanticCachingChatModel model =
        new SemanticCachingChatModel(fake, new TopicEmbeddingModel(), cache);

    assertThat(model.delegate()).isSameAs(fake);
    assertThat(model.cache()).isSameAs(cache);
  }

  /** A delegate that answers with a tool call rather than text. */
  private static final class ToolCallingChatModel
      implements org.springframework.ai.chat.model.ChatModel {
    @Override
    public org.springframework.ai.chat.model.ChatResponse call(Prompt prompt) {
      org.springframework.ai.chat.messages.AssistantMessage message =
          org.springframework.ai.chat.messages.AssistantMessage.builder()
              .content("")
              .toolCalls(
                  List.of(
                      new org.springframework.ai.chat.messages.AssistantMessage.ToolCall(
                          "1", "function", "resetPassword", "{}")))
              .build();
      return new org.springframework.ai.chat.model.ChatResponse(
          List.of(new org.springframework.ai.chat.model.Generation(message)));
    }
  }
}
