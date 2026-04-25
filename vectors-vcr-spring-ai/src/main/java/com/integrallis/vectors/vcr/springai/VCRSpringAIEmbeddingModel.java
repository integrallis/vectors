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
package com.integrallis.vectors.vcr.springai;

import com.integrallis.vectors.vcr.CassetteKey;
import com.integrallis.vectors.vcr.CassetteRecord;
import com.integrallis.vectors.vcr.CassetteStore;
import com.integrallis.vectors.vcr.VCRCassetteMissingException;
import com.integrallis.vectors.vcr.VCRMode;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.Embedding;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.embedding.EmbeddingRequest;
import org.springframework.ai.embedding.EmbeddingResponse;

/**
 * Spring AI {@link EmbeddingModel} wrapper that records and replays calls through a {@link
 * CassetteStore}.
 */
public final class VCRSpringAIEmbeddingModel implements EmbeddingModel {

  private static final String TYPE_SINGLE = "embedding";
  private static final String TYPE_BATCH = "batch_embedding";

  private final EmbeddingModel delegate;
  private final CassetteStore store;
  private final String testId;
  private final String modelName;
  private final VCRMode mode;
  private final AtomicInteger singleCalls = new AtomicInteger();
  private final AtomicInteger batchCalls = new AtomicInteger();
  private final int dimensionSize;

  /**
   * @param delegate real Spring AI embedding model
   * @param testId test identifier
   * @param mode VCR mode
   * @param modelName model name for cassette metadata
   * @param store cassette store
   */
  public VCRSpringAIEmbeddingModel(
      EmbeddingModel delegate, String testId, VCRMode mode, String modelName, CassetteStore store) {
    this.delegate = delegate;
    this.testId = testId;
    this.mode = mode;
    this.modelName = modelName;
    this.store = store;
    this.dimensionSize = detectDimensions(delegate);
  }

  @Override
  public EmbeddingResponse call(EmbeddingRequest request) {
    List<float[]> embeddings = embed(request.getInstructions());
    List<Embedding> results =
        IntStream.range(0, embeddings.size())
            .mapToObj(i -> new Embedding(embeddings.get(i), i))
            .toList();
    return new EmbeddingResponse(results);
  }

  @Override
  public float[] embed(String text) {
    if (mode == VCRMode.OFF) {
      return delegate.embed(text);
    }
    CassetteKey key = new CassetteKey(TYPE_SINGLE, testId, singleCalls.incrementAndGet());
    if (mode.isPlaybackMode()) {
      Optional<CassetteRecord> cached = store.retrieve(key);
      if (cached.isPresent()) {
        if (cached.get() instanceof CassetteRecord.Embedding e) {
          return e.embedding();
        }
        throw new IllegalStateException(
            "Expected Embedding cassette for key "
                + key.serializedKey()
                + " but got "
                + cached.get().getClass().getSimpleName());
      }
      if (mode == VCRMode.PLAYBACK) {
        throw new VCRCassetteMissingException(key.serializedKey(), testId);
      }
    }
    float[] embedding = delegate.embed(text);
    store.store(
        key,
        new CassetteRecord.Embedding(testId, modelName, System.currentTimeMillis(), embedding));
    return embedding;
  }

  @Override
  public float[] embed(Document document) {
    String text = document.getText();
    return embed(text == null ? "" : text);
  }

  @Override
  public List<float[]> embed(List<String> texts) {
    if (mode == VCRMode.OFF) {
      return delegate.embed(texts);
    }
    CassetteKey key = new CassetteKey(TYPE_BATCH, testId, batchCalls.incrementAndGet());
    if (mode.isPlaybackMode()) {
      Optional<CassetteRecord> cached = store.retrieve(key);
      if (cached.isPresent()) {
        if (!(cached.get() instanceof CassetteRecord.BatchEmbedding b)) {
          throw new IllegalStateException(
              "Expected BatchEmbedding cassette for key "
                  + key.serializedKey()
                  + " but got "
                  + cached.get().getClass().getSimpleName());
        }
        float[][] arr = b.embeddings();
        List<float[]> result = new ArrayList<>(arr.length);
        for (float[] v : arr) {
          result.add(v);
        }
        return result;
      }
      if (mode == VCRMode.PLAYBACK) {
        throw new VCRCassetteMissingException(key.serializedKey(), testId);
      }
    }
    List<float[]> embeddings = delegate.embed(texts);
    store.store(
        key,
        new CassetteRecord.BatchEmbedding(
            testId, modelName, System.currentTimeMillis(), embeddings.toArray(new float[0][])));
    return embeddings;
  }

  @Override
  public EmbeddingResponse embedForResponse(List<String> texts) {
    return call(new EmbeddingRequest(texts, null));
  }

  @Override
  public int dimensions() {
    return dimensionSize;
  }

  /**
   * @return the underlying delegate (for diagnostics)
   */
  public EmbeddingModel getDelegate() {
    return delegate;
  }

  private static int detectDimensions(EmbeddingModel model) {
    try {
      return model.dimensions();
    } catch (Exception e) {
      return -1;
    }
  }
}
