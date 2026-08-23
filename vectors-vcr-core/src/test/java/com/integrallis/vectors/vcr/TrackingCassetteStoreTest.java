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
package com.integrallis.vectors.vcr;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class TrackingCassetteStoreTest {

  @Test
  void tracksOnlySuccessfulWrites() {
    StubCassetteStore delegate = new StubCassetteStore();
    var tracked = new ArrayList<CassetteKey>();
    CassetteStore store = TrackingCassetteStore.wrap(delegate, tracked::add);
    CassetteKey key = new CassetteKey("embedding", "Suite:test", 1);
    CassetteRecord record =
        new CassetteRecord.Embedding(
            "Suite:test", "test-model", System.currentTimeMillis(), new float[] {1f});

    store.store(key, record);

    assertThat(tracked).containsExactly(key);

    CassetteKey failedKey = new CassetteKey("embedding", "Suite:test", 2);
    delegate.failOnStore = failedKey;

    assertThatThrownBy(() -> store.store(failedKey, record))
        .isInstanceOf(IllegalStateException.class)
        .hasMessage("write failed");
    assertThat(tracked).containsExactly(key);
  }

  @Test
  void preservesSimilarityLookupCapability() {
    float[] query = {1f, 2f};
    CassetteRecord expected =
        new CassetteRecord.Embedding("Suite:test", "test-model", System.currentTimeMillis(), query);
    SimilarityStubCassetteStore delegate = new SimilarityStubCassetteStore(expected);

    CassetteStore store = TrackingCassetteStore.wrap(delegate, ignored -> {});

    assertThat(store).isInstanceOf(SimilarityCassetteStore.class);
    assertThat(((SimilarityCassetteStore) store).retrieveSimilar(query)).containsSame(expected);
  }

  private static class StubCassetteStore implements CassetteStore {
    private final Map<CassetteKey, CassetteRecord> records = new LinkedHashMap<>();
    private CassetteKey failOnStore;

    @Override
    public void store(CassetteKey key, CassetteRecord record) {
      if (key.equals(failOnStore)) {
        throw new IllegalStateException("write failed");
      }
      records.put(key, record);
    }

    @Override
    public Optional<CassetteRecord> retrieve(CassetteKey key) {
      return Optional.ofNullable(records.get(key));
    }

    @Override
    public boolean exists(CassetteKey key) {
      return records.containsKey(key);
    }

    @Override
    public void delete(CassetteKey key) {
      records.remove(key);
    }

    @Override
    public List<CassetteKey> listByTestId(String testId) {
      return records.keySet().stream().filter(key -> key.testId().equals(testId)).toList();
    }
  }

  private static final class SimilarityStubCassetteStore extends StubCassetteStore
      implements SimilarityCassetteStore {
    private final CassetteRecord similar;

    private SimilarityStubCassetteStore(CassetteRecord similar) {
      this.similar = similar;
    }

    @Override
    public Optional<CassetteRecord> retrieveSimilar(float[] queryEmbedding) {
      return Optional.of(similar);
    }
  }
}
