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
package com.integrallis.vectors.cache.springai;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.Embedding;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.embedding.EmbeddingRequest;
import org.springframework.ai.embedding.EmbeddingResponse;

/** Deterministic test embedding model: trigram hash over the text, 8 dims. */
final class FakeEmbeddingModel implements EmbeddingModel {

  static final int DIM = 8;

  final AtomicInteger singleCalls = new AtomicInteger();
  final AtomicInteger batchCalls = new AtomicInteger();

  @Override
  public float[] embed(String text) {
    singleCalls.incrementAndGet();
    return encode(text);
  }

  @Override
  public float[] embed(Document document) {
    singleCalls.incrementAndGet();
    String text = document.getText();
    return encode(text == null ? "" : text);
  }

  @Override
  public List<float[]> embed(List<String> texts) {
    batchCalls.incrementAndGet();
    List<float[]> out = new ArrayList<>(texts.size());
    for (String t : texts) {
      out.add(encode(t));
    }
    return out;
  }

  @Override
  public EmbeddingResponse call(EmbeddingRequest request) {
    List<float[]> vectors = embed(request.getInstructions());
    List<Embedding> out = new ArrayList<>(vectors.size());
    for (int i = 0; i < vectors.size(); i++) {
      out.add(new Embedding(vectors.get(i), i));
    }
    return new EmbeddingResponse(out);
  }

  @Override
  public int dimensions() {
    return DIM;
  }

  static float[] encode(String text) {
    float[] v = new float[DIM];
    for (int i = 0; i + 2 < text.length(); i++) {
      int h = (text.charAt(i) * 31 + text.charAt(i + 1)) * 31 + text.charAt(i + 2);
      v[Math.floorMod(h, DIM)] += 1.0f;
    }
    return v;
  }
}
