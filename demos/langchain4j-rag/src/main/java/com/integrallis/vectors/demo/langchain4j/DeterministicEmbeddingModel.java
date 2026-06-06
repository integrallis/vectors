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
package com.integrallis.vectors.demo.langchain4j;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;
import java.util.List;
import java.util.Objects;

/**
 * Zero-dependency, deterministic LangChain4j {@link EmbeddingModel}.
 *
 * <p>Bag-of-character-trigrams vector of the configured dimension, L2-normalized. Similar texts
 * produce similar vectors — enough to make retrieval feel meaningful without requiring model
 * downloads.
 *
 * <p><b>DEMO ONLY — NOT FOR PRODUCTION.</b> These are synthetic character-trigram hashes, not real
 * semantic embeddings; retrieval quality is meaningless on real workloads. In a real application
 * substitute e.g. {@code AllMiniLmL6V2EmbeddingModel} ({@code
 * langchain4j-embeddings-all-minilm-l6-v2}) or {@code OpenAiEmbeddingModel}. The constructor logs a
 * warning so this never sneaks into production unnoticed.
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
  public int dimension() {
    return dimension;
  }

  @Override
  public Response<Embedding> embed(String text) {
    Objects.requireNonNull(text, "text");
    return Response.from(Embedding.from(trigramVector(text)));
  }

  @Override
  public Response<Embedding> embed(TextSegment textSegment) {
    return embed(textSegment.text());
  }

  @Override
  public Response<List<Embedding>> embedAll(List<TextSegment> textSegments) {
    List<Embedding> out =
        textSegments.stream().map(s -> Embedding.from(trigramVector(s.text()))).toList();
    return Response.from(out);
  }

  private float[] trigramVector(String text) {
    float[] out = new float[dimension];
    String normalized = text.toLowerCase();
    if (normalized.length() < 3) {
      normalized = "  " + normalized + "  ";
    }
    for (int i = 0; i <= normalized.length() - 3; i++) {
      int h =
          31 * (31 * normalized.charAt(i) + normalized.charAt(i + 1)) + normalized.charAt(i + 2);
      int idx = Math.floorMod(h, dimension);
      out[idx] += 1.0f;
    }
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
}
