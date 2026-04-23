package com.integrallis.vectors.vcr.semantic;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.integrallis.vectors.storage.backend.HeapStorageBackend;
import com.integrallis.vectors.vcr.CassetteKey;
import com.integrallis.vectors.vcr.CassetteRecord;
import java.io.IOException;
import java.util.Optional;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class SemanticCassetteStoreTest {

  private static final int DIM = 4;

  private SemanticCassetteStore newStore() {
    return new SemanticCassetteStore(new HeapStorageBackend(), new TestSerializer(), DIM);
  }

  private SemanticCassetteStore newStore(float threshold) {
    return new SemanticCassetteStore(
        new HeapStorageBackend(), new TestSerializer(), DIM, threshold);
  }

  private static float[] unit(float... v) {
    double n = 0.0;
    for (float x : v) n += x * x;
    float inv = (float) (1.0 / Math.sqrt(n));
    float[] out = new float[v.length];
    for (int i = 0; i < v.length; i++) out[i] = v[i] * inv;
    return out;
  }

  @Test
  void exactRetrievalBehavesLikeExactStore() throws IOException {
    try (SemanticCassetteStore store = newStore()) {
      CassetteKey key = new CassetteKey("embedding", "T:x", 1);
      float[] v = unit(1f, 2f, 3f, 4f);
      store.store(key, new CassetteRecord.Embedding("T:x", "m", 1L, v));

      assertTrue(store.exists(key));
      CassetteRecord.Embedding got = (CassetteRecord.Embedding) store.retrieve(key).orElseThrow();
      assertArrayEquals(v, got.embedding());
    }
  }

  @Test
  void retrieveSimilarReturnsMatchAboveThreshold() throws IOException {
    try (SemanticCassetteStore store = newStore(0.9f)) {
      float[] stored = unit(1f, 1f, 0f, 0f);
      store.store(
          new CassetteKey("embedding", "T:a", 1),
          new CassetteRecord.Embedding("T:a", "m", 0L, stored));

      float[] close = unit(1.01f, 0.99f, 0f, 0f);
      Optional<CassetteRecord> hit = store.retrieveSimilar(close);
      assertTrue(hit.isPresent());
      CassetteRecord.Embedding back = (CassetteRecord.Embedding) hit.get();
      assertArrayEquals(stored, back.embedding());
    }
  }

  @Test
  void retrieveSimilarReturnsEmptyBelowThreshold() throws IOException {
    try (SemanticCassetteStore store = newStore(0.99f)) {
      float[] stored = unit(1f, 0f, 0f, 0f);
      store.store(
          new CassetteKey("embedding", "T:a", 1),
          new CassetteRecord.Embedding("T:a", "m", 0L, stored));

      float[] far = unit(0f, 1f, 0f, 0f);
      assertTrue(store.retrieveSimilar(far).isEmpty());
    }
  }

  @Test
  void retrieveSimilarReturnsEmptyWhenIndexIsEmpty() throws IOException {
    try (SemanticCassetteStore store = newStore()) {
      assertTrue(store.retrieveSimilar(unit(1f, 0f, 0f, 0f)).isEmpty());
    }
  }

  @Test
  void batchEmbeddingIndexesFirstVector() throws IOException {
    try (SemanticCassetteStore store = newStore(0.9f)) {
      float[] v0 = unit(1f, 0f, 0f, 0f);
      float[] v1 = unit(0f, 1f, 0f, 0f);
      CassetteKey key = new CassetteKey("batch_embedding", "T:b", 1);
      store.store(key, new CassetteRecord.BatchEmbedding("T:b", "m", 0L, new float[][] {v0, v1}));

      Optional<CassetteRecord> hit = store.retrieveSimilar(v0);
      assertTrue(hit.isPresent());
    }
  }

  @Test
  void chatRecordHasNoSemanticHook() throws IOException {
    try (SemanticCassetteStore store = newStore()) {
      CassetteKey key = new CassetteKey("chat", "T:c", 1);
      store.store(
          key, new CassetteRecord.Chat("T:c", "gpt", 0L, "hi", "hello", java.util.Map.of()));

      assertTrue(store.exists(key));
      assertTrue(store.retrieve(key).isPresent());
      assertTrue(store.retrieveSimilar(unit(1f, 0f, 0f, 0f)).isEmpty());
    }
  }

  @Test
  void deleteRemovesFromBothStores() throws IOException {
    try (SemanticCassetteStore store = newStore(0.9f)) {
      CassetteKey key = new CassetteKey("embedding", "T:d", 1);
      float[] v = unit(1f, 1f, 0f, 0f);
      store.store(key, new CassetteRecord.Embedding("T:d", "m", 0L, v));
      assertTrue(store.retrieveSimilar(v).isPresent());

      store.delete(key);
      assertFalse(store.exists(key));
      assertTrue(store.retrieveSimilar(v).isEmpty());
    }
  }

  @Test
  void upsertReplacesExistingVector() throws IOException {
    try (SemanticCassetteStore store = newStore(0.9f)) {
      CassetteKey key = new CassetteKey("embedding", "T:u", 1);
      float[] v1 = unit(1f, 0f, 0f, 0f);
      float[] v2 = unit(0f, 1f, 0f, 0f);

      store.store(key, new CassetteRecord.Embedding("T:u", "m", 0L, v1));
      store.store(key, new CassetteRecord.Embedding("T:u", "m", 1L, v2));

      CassetteRecord.Embedding got = (CassetteRecord.Embedding) store.retrieve(key).orElseThrow();
      assertArrayEquals(v2, got.embedding());
      assertEquals(1L, got.timestamp());
    }
  }

  @Test
  void listByTestIdFiltersByTest() throws IOException {
    try (SemanticCassetteStore store = newStore(0.9f)) {
      store.store(
          new CassetteKey("embedding", "TA:x", 1),
          new CassetteRecord.Embedding("TA:x", "m", 0L, unit(1f, 0f, 0f, 0f)));
      store.store(
          new CassetteKey("embedding", "TB:y", 1),
          new CassetteRecord.Embedding("TB:y", "m", 0L, unit(0f, 1f, 0f, 0f)));
      assertEquals(1, store.listByTestId("TA:x").size());
      assertEquals(1, store.listByTestId("TB:y").size());
    }
  }
}
