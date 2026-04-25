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

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.integrallis.vectors.storage.backend.HeapStorageBackend;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class ExactCassetteStoreTest {

  private final ExactCassetteStore store =
      new ExactCassetteStore(new HeapStorageBackend(), new TestSerializer());

  @Test
  void storeAndRetrieveEmbedding() {
    CassetteKey key = new CassetteKey("embedding", "T:one", 1);
    float[] vec = {0.1f, 0.2f, 0.3f};
    CassetteRecord.Embedding rec = new CassetteRecord.Embedding("T:one", "m", 123L, vec);

    store.store(key, rec);

    assertTrue(store.exists(key));
    Optional<CassetteRecord> got = store.retrieve(key);
    assertTrue(got.isPresent());
    CassetteRecord.Embedding back = (CassetteRecord.Embedding) got.get();
    assertEquals("T:one", back.testId());
    assertEquals("m", back.model());
    assertEquals(123L, back.timestamp());
    assertArrayEquals(vec, back.embedding());
  }

  @Test
  void retrieveReturnsEmptyWhenAbsent() {
    CassetteKey key = new CassetteKey("embedding", "T:missing", 1);
    assertFalse(store.exists(key));
    assertTrue(store.retrieve(key).isEmpty());
  }

  @Test
  void storeAndRetrieveBatchEmbedding() {
    CassetteKey key = new CassetteKey("batch_embedding", "T:batch", 1);
    float[][] batch = {{1f, 2f}, {3f, 4f}};
    CassetteRecord.BatchEmbedding rec =
        new CassetteRecord.BatchEmbedding("T:batch", "m", 0L, batch);
    store.store(key, rec);

    CassetteRecord.BatchEmbedding back =
        (CassetteRecord.BatchEmbedding) store.retrieve(key).orElseThrow();
    assertEquals(2, back.embeddings().length);
    assertArrayEquals(batch[0], back.embeddings()[0]);
    assertArrayEquals(batch[1], back.embeddings()[1]);
  }

  @Test
  void storeAndRetrieveChat() {
    CassetteKey key = new CassetteKey("chat", "T:chat", 1);
    CassetteRecord.Chat rec =
        new CassetteRecord.Chat(
            "T:chat", "gpt", 99L, "hello?", "world!", Map.of("role", "assistant"));
    store.store(key, rec);

    CassetteRecord.Chat back = (CassetteRecord.Chat) store.retrieve(key).orElseThrow();
    assertEquals("hello?", back.prompt());
    assertEquals("world!", back.response());
    assertEquals("assistant", back.metadata().get("role"));
  }

  @Test
  void listByTestIdFiltersByTest() {
    store.store(
        new CassetteKey("embedding", "TA:one", 1),
        new CassetteRecord.Embedding("TA:one", "m", 0L, new float[] {1f}));
    store.store(
        new CassetteKey("embedding", "TA:one", 2),
        new CassetteRecord.Embedding("TA:one", "m", 0L, new float[] {2f}));
    store.store(
        new CassetteKey("embedding", "TB:other", 1),
        new CassetteRecord.Embedding("TB:other", "m", 0L, new float[] {3f}));

    List<CassetteKey> a = store.listByTestId("TA:one");
    assertEquals(2, a.size());
    assertTrue(a.stream().allMatch(k -> k.testId().equals("TA:one")));

    List<CassetteKey> b = store.listByTestId("TB:other");
    assertEquals(1, b.size());
  }

  @Test
  void deleteRemovesCassette() {
    CassetteKey key = new CassetteKey("embedding", "T:del", 1);
    store.store(key, new CassetteRecord.Embedding("T:del", "m", 0L, new float[] {0f}));
    assertTrue(store.exists(key));
    store.delete(key);
    assertFalse(store.exists(key));
  }
}
