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

import com.integrallis.vectors.cache.SemanticCache;
import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.VectorCollection;
import java.util.Optional;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
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
}
