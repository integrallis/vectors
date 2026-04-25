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
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * LangChain4j {@link EmbeddingModel} wrapper that records/replays calls through a {@link
 * CassetteStore}.
 */
public final class VCREmbeddingModel implements EmbeddingModel {

  private static final String TYPE_SINGLE = "embedding";
  private static final String TYPE_BATCH = "batch_embedding";

  private final EmbeddingModel delegate;
  private final CassetteStore store;
  private final String testId;
  private final String modelName;
  private final VCRMode mode;
  private final AtomicInteger singleCalls = new AtomicInteger();
  private final AtomicInteger batchCalls = new AtomicInteger();
  private final int dims;

  /**
   * @param delegate underlying LangChain4j embedding model
   * @param testId test identifier
   * @param mode VCR mode
   * @param modelName model name for cassette metadata
   * @param store cassette store
   */
  public VCREmbeddingModel(
      EmbeddingModel delegate, String testId, VCRMode mode, String modelName, CassetteStore store) {
    this.delegate = delegate;
    this.testId = testId;
    this.mode = mode;
    this.modelName = modelName;
    this.store = store;
    this.dims = detectDimensions(delegate);
  }

  @Override
  public Response<Embedding> embed(String text) {
    return Response.from(Embedding.from(embedSingle(text)));
  }

  @Override
  public Response<Embedding> embed(TextSegment segment) {
    return embed(segment.text());
  }

  @Override
  public Response<List<Embedding>> embedAll(List<TextSegment> segments) {
    List<String> texts = segments.stream().map(TextSegment::text).toList();
    List<float[]> vectors = embedBatch(texts);
    List<Embedding> out = new ArrayList<>(vectors.size());
    for (float[] v : vectors) {
      out.add(Embedding.from(v));
    }
    return Response.from(out);
  }

  @Override
  public int dimension() {
    return dims;
  }

  private float[] embedSingle(String text) {
    if (mode == VCRMode.OFF) {
      return delegate.embed(text).content().vector();
    }
    CassetteKey key = new CassetteKey(TYPE_SINGLE, testId, singleCalls.incrementAndGet());
    if (mode.isPlaybackMode()) {
      Optional<CassetteRecord> hit = store.retrieve(key);
      if (hit.isPresent()) {
        if (hit.get() instanceof CassetteRecord.Embedding e) {
          return e.embedding();
        }
        throw new IllegalStateException(
            "Expected Embedding cassette for key "
                + key.serializedKey()
                + " but got "
                + hit.get().getClass().getSimpleName());
      }
      if (mode == VCRMode.PLAYBACK) {
        throw new VCRCassetteMissingException(key.serializedKey(), testId);
      }
    }
    float[] vector = delegate.embed(text).content().vector();
    store.store(
        key, new CassetteRecord.Embedding(testId, modelName, System.currentTimeMillis(), vector));
    return vector;
  }

  private List<float[]> embedBatch(List<String> texts) {
    if (mode == VCRMode.OFF) {
      return callDelegateBatch(texts);
    }
    CassetteKey key = new CassetteKey(TYPE_BATCH, testId, batchCalls.incrementAndGet());
    if (mode.isPlaybackMode()) {
      Optional<CassetteRecord> hit = store.retrieve(key);
      if (hit.isPresent()) {
        if (!(hit.get() instanceof CassetteRecord.BatchEmbedding b)) {
          throw new IllegalStateException(
              "Expected BatchEmbedding cassette for key "
                  + key.serializedKey()
                  + " but got "
                  + hit.get().getClass().getSimpleName());
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
    List<float[]> vectors = callDelegateBatch(texts);
    store.store(
        key,
        new CassetteRecord.BatchEmbedding(
            testId, modelName, System.currentTimeMillis(), vectors.toArray(new float[0][])));
    return vectors;
  }

  private List<float[]> callDelegateBatch(List<String> texts) {
    List<TextSegment> segs = texts.stream().map(TextSegment::from).toList();
    List<Embedding> out = delegate.embedAll(segs).content();
    List<float[]> vectors = new ArrayList<>(out.size());
    for (Embedding e : out) {
      vectors.add(e.vector());
    }
    return vectors;
  }

  /**
   * @return the underlying delegate
   */
  public EmbeddingModel getDelegate() {
    return delegate;
  }

  private static int detectDimensions(EmbeddingModel model) {
    try {
      return model.dimension();
    } catch (Exception e) {
      return -1;
    }
  }
}
