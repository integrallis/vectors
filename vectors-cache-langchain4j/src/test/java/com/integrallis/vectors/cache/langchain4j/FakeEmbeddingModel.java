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
package com.integrallis.vectors.cache.langchain4j;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/** Deterministic test embedding model: trigram hash, 8 dims. */
final class FakeEmbeddingModel implements EmbeddingModel {

  static final int DIM = 8;

  final AtomicInteger singleCalls = new AtomicInteger();
  final AtomicInteger batchCalls = new AtomicInteger();

  @Override
  public Response<Embedding> embed(String text) {
    singleCalls.incrementAndGet();
    return Response.from(Embedding.from(encode(text)));
  }

  @Override
  public Response<Embedding> embed(TextSegment segment) {
    return embed(segment.text());
  }

  @Override
  public Response<List<Embedding>> embedAll(List<TextSegment> segments) {
    batchCalls.incrementAndGet();
    List<Embedding> out = new ArrayList<>(segments.size());
    for (TextSegment s : segments) {
      out.add(Embedding.from(encode(s.text())));
    }
    return Response.from(out);
  }

  @Override
  public int dimension() {
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
