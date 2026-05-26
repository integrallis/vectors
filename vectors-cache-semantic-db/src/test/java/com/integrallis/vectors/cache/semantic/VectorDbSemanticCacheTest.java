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
package com.integrallis.vectors.cache.semantic;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.cache.CacheAdmissionPolicy;
import com.integrallis.vectors.cache.LLMResponseFilters;
import com.integrallis.vectors.cache.SemanticCache;
import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.VectorCollection;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Proxy;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

class VectorDbSemanticCacheTest {

  private VectorCollection collection;
  private VectorDbSemanticCache<String> cache;

  @BeforeEach
  void setUp() {
    collection =
        VectorCollection.builder()
            .dimension(4)
            .metric(SimilarityFunction.COSINE)
            .indexType(IndexType.FLAT)
            .build();
    cache =
        VectorDbSemanticCache.builder(collection, PayloadCodec.identity())
            .threshold(0.92)
            .closeCollectionOnClose(true)
            .build();
  }

  @AfterEach
  void tearDown() {
    cache.close();
  }

  @Test
  void putAndExactGetRoundTrip() {
    cache.put("k1", new float[] {1f, 0f, 0f, 0f}, "assistant reply #1");
    assertThat(cache.get("k1")).hasValue("assistant reply #1");
    assertThat(cache.get("missing")).isEmpty();
  }

  @Test
  void putAllCommitsAcceptedEntriesOnce() {
    AtomicInteger commits = new AtomicInteger();
    VectorDbSemanticCache<String> countedCache =
        VectorDbSemanticCache.builder(countingCommits(collection, commits), PayloadCodec.identity())
            .threshold(0.92)
            .build();

    countedCache.putAll(
        List.of(
            new SemanticCache.Entry<>("k1", new float[] {1f, 0f, 0f, 0f}, "r1"),
            new SemanticCache.Entry<>("k2", new float[] {0f, 1f, 0f, 0f}, "r2")));

    assertThat(commits.get()).isEqualTo(1);
    assertThat(countedCache.get("k1")).hasValue("r1");
    assertThat(countedCache.get("k2")).hasValue("r2");
  }

  @Test
  void putAllDoesNotCommitWhenAdmissionRejectsEveryEntry() {
    AtomicInteger commits = new AtomicInteger();
    VectorDbSemanticCache<String> countedCache =
        VectorDbSemanticCache.builder(countingCommits(collection, commits), PayloadCodec.identity())
            .threshold(0.92)
            .admissionPolicy(value -> false)
            .build();

    countedCache.putAll(
        List.of(
            new SemanticCache.Entry<>("k1", new float[] {1f, 0f, 0f, 0f}, "r1"),
            new SemanticCache.Entry<>("k2", new float[] {0f, 1f, 0f, 0f}, "r2")));

    assertThat(commits.get()).isZero();
    assertThat(countedCache.stats().rejections()).isEqualTo(2);
    assertThat(countedCache.stats().size()).isZero();
  }

  @Test
  void nearDuplicateLookupReturnsMatchAboveThreshold() {
    cache.put("k1", new float[] {1f, 0f, 0f, 0f}, "reply");
    Optional<SemanticCache.Hit<String>> hit = cache.lookup(new float[] {0.999f, 0.01f, 0f, 0f});
    assertThat(hit).isPresent();
    assertThat(hit.get().value()).isEqualTo("reply");
    assertThat(hit.get().score()).isGreaterThan(0.92);
  }

  @Test
  void belowThresholdReturnsMiss() {
    cache.put("k1", new float[] {1f, 0f, 0f, 0f}, "reply");
    // Orthogonal to k1 -> cosine 0 -> below 0.92 threshold.
    Optional<SemanticCache.Hit<String>> hit = cache.lookup(new float[] {0f, 1f, 0f, 0f});
    assertThat(hit).isEmpty();
  }

  @Test
  void statsTrackHitsAndMisses() {
    cache.put("k1", new float[] {1f, 0f, 0f, 0f}, "reply");
    cache.lookup(new float[] {0.99f, 0.01f, 0f, 0f}); // hit
    cache.lookup(new float[] {0f, 1f, 0f, 0f}); // miss
    assertThat(cache.stats().hits()).isEqualTo(1);
    assertThat(cache.stats().misses()).isEqualTo(1);
  }

  @Test
  void invalidateRemovesEntry() {
    cache.put("k1", new float[] {1f, 0f, 0f, 0f}, "reply");
    cache.invalidate("k1");
    assertThat(cache.get("k1")).isEmpty();
    assertThat(cache.lookup(new float[] {1f, 0f, 0f, 0f})).isEmpty();
  }

  @Test
  void invalidateAllClearsEverything() {
    cache.put("k1", new float[] {1f, 0f, 0f, 0f}, "r1");
    cache.put("k2", new float[] {0f, 1f, 0f, 0f}, "r2");
    cache.invalidateAll();
    assertThat(cache.stats().size()).isZero();
  }

  @Test
  void lookupOnEmptyCacheReturnsMiss() {
    assertThat(cache.lookup(new float[] {1f, 0f, 0f, 0f})).isEmpty();
  }

  @Test
  void admissionPolicyDefaultsToAllowAll() {
    assertThat(cache.admissionPolicy()).isSameAs(CacheAdmissionPolicy.allowAll());
  }

  @Nested
  @Tag("unit")
  class AdmissionPolicyTests {

    private VectorCollection filteredCollection;
    private VectorDbSemanticCache<String> filteredCache;

    @BeforeEach
    void setUp() {
      filteredCollection =
          VectorCollection.builder()
              .dimension(4)
              .metric(SimilarityFunction.COSINE)
              .indexType(IndexType.FLAT)
              .build();
      filteredCache =
          VectorDbSemanticCache.builder(filteredCollection, PayloadCodec.identity())
              .threshold(0.92)
              .admissionPolicy(LLMResponseFilters.rejectRefusals())
              .closeCollectionOnClose(true)
              .build();
    }

    @AfterEach
    void tearDown() {
      filteredCache.close();
    }

    @Test
    void putWithAdmissionPolicyRejectsRefusal() {
      filteredCache.put("q1", new float[] {1f, 0f, 0f, 0f}, "I can't answer that.");
      assertThat(filteredCache.get("q1")).isEmpty();
      assertThat(filteredCache.stats().size()).isZero();
    }

    @Test
    void putWithAdmissionPolicyAcceptsNormalResponse() {
      filteredCache.put("q1", new float[] {1f, 0f, 0f, 0f}, "The capital of France is Paris.");
      assertThat(filteredCache.get("q1")).hasValue("The capital of France is Paris.");
    }

    @Test
    void rejectionCountTrackedInStats() {
      filteredCache.put("q1", new float[] {1f, 0f, 0f, 0f}, "I can't answer that.");
      filteredCache.put("q2", new float[] {0f, 1f, 0f, 0f}, "I cannot do that.");
      assertThat(filteredCache.stats().rejections()).isEqualTo(2);
    }
  }

  private static VectorCollection countingCommits(
      VectorCollection delegate, AtomicInteger commits) {
    return (VectorCollection)
        Proxy.newProxyInstance(
            VectorCollection.class.getClassLoader(),
            new Class<?>[] {VectorCollection.class},
            (proxy, method, args) -> {
              if (method.getName().equals("commit") && method.getParameterCount() == 0) {
                commits.incrementAndGet();
              }
              try {
                return method.invoke(delegate, args);
              } catch (InvocationTargetException e) {
                throw e.getCause();
              }
            });
  }
}
