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
package com.integrallis.vectors.vcr.langchain4j;

import com.integrallis.vectors.vcr.CassetteKey;
import com.integrallis.vectors.vcr.CassetteRecord;
import com.integrallis.vectors.vcr.CassetteStore;
import com.integrallis.vectors.vcr.VCRCassetteMissingException;
import com.integrallis.vectors.vcr.VCRMode;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Abstract base class that adds VCR record/replay to any embedding provider that does not fit a
 * pre-existing framework interface.
 *
 * <p>Subclasses implement {@link #callRealEmbedding(String)} and {@link
 * #callRealBatchEmbedding(List)} to plug in the underlying embedding backend (HTTP client, custom
 * model, etc). The base class handles cassette key generation, mode-based dispatch, statistics, and
 * persistence through a {@link CassetteStore}.
 *
 * <p>Single-text calls produce {@link CassetteRecord.Embedding} cassettes under the {@code
 * "embedding"} type; batch calls produce {@link CassetteRecord.BatchEmbedding} cassettes under the
 * {@code "batch_embedding"} type, each with its own monotonic call counter per test.
 *
 * <p>This class is part of {@code vectors-vcr-langchain4j} for packaging convenience; it does not
 * depend on any LangChain4j types and can be used standalone.
 */
public abstract class VCREmbeddingInterceptor {

  private static final String TYPE_SINGLE = "embedding";
  private static final String TYPE_BATCH = "batch_embedding";

  private final CassetteStore store;
  private final AtomicInteger singleCounter = new AtomicInteger(0);
  private final AtomicInteger batchCounter = new AtomicInteger(0);
  private final List<String> recordedKeys = new ArrayList<>();
  private final AtomicLong cacheHits = new AtomicLong(0);
  private final AtomicLong cacheMisses = new AtomicLong(0);

  private volatile VCRMode mode = VCRMode.OFF;
  private volatile String testId = "";
  private volatile String modelName = "default";

  /**
   * @param store cassette store backing this interceptor
   */
  protected VCREmbeddingInterceptor(CassetteStore store) {
    this.store = Objects.requireNonNull(store, "store");
  }

  /**
   * @param mode the new VCR mode
   */
  public final void setMode(VCRMode mode) {
    this.mode = Objects.requireNonNull(mode, "mode");
  }

  /**
   * @return the current VCR mode
   */
  public final VCRMode getMode() {
    return mode;
  }

  /**
   * Sets the active test identifier and resets the per-test call counters.
   *
   * @param testId the test identifier
   */
  public final void setTestId(String testId) {
    this.testId = Objects.requireNonNull(testId, "testId");
    resetCallCounter();
  }

  /**
   * @return the current test identifier
   */
  public final String getTestId() {
    return testId;
  }

  /**
   * @param modelName label stored in cassette metadata
   */
  public final void setModelName(String modelName) {
    this.modelName = Objects.requireNonNull(modelName, "modelName");
  }

  /**
   * @return the current model name
   */
  public final String getModelName() {
    return modelName;
  }

  /**
   * Embeds a single text, routing through the cassette store according to {@link #getMode()}.
   *
   * @param text the text to embed
   * @return the embedding vector
   */
  public final float[] embed(String text) {
    if (mode == VCRMode.OFF) {
      return callRealEmbedding(text);
    }
    CassetteKey key = new CassetteKey(TYPE_SINGLE, testId, singleCounter.incrementAndGet());
    if (mode.isPlaybackMode()) {
      Optional<CassetteRecord> hit = store.retrieve(key);
      if (hit.isPresent()) {
        cacheHits.incrementAndGet();
        return ((CassetteRecord.Embedding) hit.get()).embedding();
      }
      if (mode == VCRMode.PLAYBACK) {
        throw new VCRCassetteMissingException(key.serializedKey(), testId);
      }
    }
    cacheMisses.incrementAndGet();
    float[] vector = callRealEmbedding(text);
    if (mode.isRecordMode()) {
      store.store(
          key, new CassetteRecord.Embedding(testId, modelName, System.currentTimeMillis(), vector));
      recordedKeys.add(key.serializedKey());
    }
    return vector;
  }

  /**
   * Embeds a batch of texts, routing through the cassette store according to {@link #getMode()}.
   *
   * @param texts the texts to embed
   * @return one embedding vector per input text
   */
  public final List<float[]> embedBatch(List<String> texts) {
    if (mode == VCRMode.OFF) {
      return callRealBatchEmbedding(texts);
    }
    CassetteKey key = new CassetteKey(TYPE_BATCH, testId, batchCounter.incrementAndGet());
    if (mode.isPlaybackMode()) {
      Optional<CassetteRecord> hit = store.retrieve(key);
      if (hit.isPresent()) {
        cacheHits.incrementAndGet();
        float[][] arr = ((CassetteRecord.BatchEmbedding) hit.get()).embeddings();
        List<float[]> out = new ArrayList<>(arr.length);
        for (float[] v : arr) {
          out.add(v);
        }
        return out;
      }
      if (mode == VCRMode.PLAYBACK) {
        throw new VCRCassetteMissingException(key.serializedKey(), testId);
      }
    }
    cacheMisses.incrementAndGet();
    List<float[]> vectors = callRealBatchEmbedding(texts);
    if (mode.isRecordMode()) {
      store.store(
          key,
          new CassetteRecord.BatchEmbedding(
              testId, modelName, System.currentTimeMillis(), vectors.toArray(new float[0][])));
      recordedKeys.add(key.serializedKey());
    }
    return vectors;
  }

  /**
   * Preloads a single-embedding cassette under the given serialized key. The key must be in {@link
   * CassetteKey#serializedKey()} format.
   *
   * @param serializedKey full cassette key, e.g. {@code vcr:embedding:MyTest:0001}
   * @param embedding the vector to record
   */
  public final void preloadCassette(String serializedKey, float[] embedding) {
    CassetteKey parsed = CassetteKey.parse(serializedKey);
    if (parsed == null) {
      throw new IllegalArgumentException("invalid cassette key: " + serializedKey);
    }
    store.store(
        parsed,
        new CassetteRecord.Embedding(testId, modelName, System.currentTimeMillis(), embedding));
  }

  /**
   * Preloads a batch-embedding cassette under the given serialized key.
   *
   * @param serializedKey full cassette key
   * @param embeddings the vectors to record
   */
  public final void preloadBatchCassette(String serializedKey, float[][] embeddings) {
    CassetteKey parsed = CassetteKey.parse(serializedKey);
    if (parsed == null) {
      throw new IllegalArgumentException("invalid cassette key: " + serializedKey);
    }
    store.store(
        parsed,
        new CassetteRecord.BatchEmbedding(
            testId, modelName, System.currentTimeMillis(), embeddings));
  }

  /**
   * @return number of cassettes recorded by this interceptor since construction
   */
  public final int getRecordedCount() {
    return recordedKeys.size();
  }

  /**
   * @return a snapshot of the keys recorded by this interceptor
   */
  public final List<String> getRecordedKeys() {
    return new ArrayList<>(recordedKeys);
  }

  /**
   * @return number of playback cache hits
   */
  public final long getCacheHits() {
    return cacheHits.get();
  }

  /**
   * @return number of playback cache misses (including record-mode calls)
   */
  public final long getCacheMisses() {
    return cacheMisses.get();
  }

  /** Resets cache-hit and cache-miss counters. */
  public final void resetStatistics() {
    cacheHits.set(0);
    cacheMisses.set(0);
  }

  /** Resets per-test call counters. Invoked automatically by {@link #setTestId(String)}. */
  public final void resetCallCounter() {
    singleCounter.set(0);
    batchCounter.set(0);
  }

  /**
   * Performs the actual single-text embedding call.
   *
   * @param text the text to embed
   * @return the embedding vector
   */
  protected abstract float[] callRealEmbedding(String text);

  /**
   * Performs the actual batch embedding call.
   *
   * @param texts the texts to embed
   * @return one embedding vector per input text
   */
  protected abstract List<float[]> callRealBatchEmbedding(List<String> texts);
}
