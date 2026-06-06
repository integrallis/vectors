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
package com.integrallis.vectors.demo.springai;

import java.util.List;
import java.util.Objects;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.embedding.EmbeddingRequest;
import org.springframework.ai.embedding.EmbeddingResponse;

/**
 * Zero-dependency, deterministic {@link EmbeddingModel} for the demo.
 *
 * <p>Maps each input to a bag-of-character-trigrams vector of the configured dimension. Similar
 * texts produce similar vectors — enough to make RAG retrieval feel meaningful without requiring
 * model downloads or API keys.
 *
 * <p><b>DEMO ONLY — NOT FOR PRODUCTION.</b> These are synthetic character-trigram hashes, not real
 * semantic embeddings; retrieval quality is meaningless on real workloads. In a real application
 * substitute e.g. {@code TransformersEmbeddingModel} or {@code OpenAiEmbeddingModel}. The
 * constructor logs a warning so this never sneaks into production unnoticed.
 */
final class DeterministicEmbeddingModel implements EmbeddingModel {

  private static final System.Logger LOG =
      System.getLogger(DeterministicEmbeddingModel.class.getName());

  private final int dimension;

  DeterministicEmbeddingModel(int dimension) {
    if (dimension <= 0) {
      throw new IllegalArgumentException("dimension must be positive: " + dimension);
    }
    LOG.log(
        System.Logger.Level.WARNING,
        "DeterministicEmbeddingModel is a DEMO-ONLY synthetic embedder (character-trigram hashing)"
            + " — NOT FOR PRODUCTION. Substitute a real EmbeddingModel for any real workload.");
    this.dimension = dimension;
  }

  @Override
  public int dimensions() {
    return dimension;
  }

  @Override
  public float[] embed(String text) {
    Objects.requireNonNull(text, "text");
    float[] out = new float[dimension];
    String normalized = text.toLowerCase();
    if (normalized.length() < 3) {
      normalized = "  " + normalized + "  ";
    }
    for (int i = 0; i <= normalized.length() - 3; i++) {
      int h = trigramHash(normalized, i);
      int idx = Math.floorMod(h, dimension);
      out[idx] += 1.0f;
    }
    // L2-normalize
    float norm = 0f;
    for (float v : out) {
      norm += v * v;
    }
    norm = (float) Math.sqrt(norm);
    if (norm > 0f) {
      for (int i = 0; i < dimension; i++) {
        out[i] /= norm;
      }
    }
    return out;
  }

  @Override
  public float[] embed(Document document) {
    String text = document.getText();
    return embed(text != null ? text : document.getId());
  }

  @Override
  public EmbeddingResponse call(EmbeddingRequest request) {
    List<float[]> vectors = request.getInstructions().stream().map(this::embed).toList();
    List<org.springframework.ai.embedding.Embedding> items =
        java.util.stream.IntStream.range(0, vectors.size())
            .mapToObj(i -> new org.springframework.ai.embedding.Embedding(vectors.get(i), i))
            .toList();
    return new EmbeddingResponse(items);
  }

  private static int trigramHash(String s, int pos) {
    int h = 0;
    h = 31 * h + s.charAt(pos);
    h = 31 * h + s.charAt(pos + 1);
    h = 31 * h + s.charAt(pos + 2);
    return h;
  }
}
