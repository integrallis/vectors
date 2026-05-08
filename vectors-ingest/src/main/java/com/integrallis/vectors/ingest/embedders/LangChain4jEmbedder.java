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
package com.integrallis.vectors.ingest.embedders;

import com.integrallis.vectors.ingest.Embedder;
import com.integrallis.vectors.ingest.IngestDoc;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Adapts a LangChain4j {@link EmbeddingModel} to the ingest {@link Embedder} SPI. The langchain4j
 * dependency is {@code compileOnly} on this module, so callers must bring their own embedding-model
 * implementation on the runtime classpath.
 *
 * <p>Documents whose {@link IngestDoc#text()} is {@code null} cause an {@link
 * IllegalArgumentException} — for binary or precomputed payloads use a different embedder. Order of
 * the returned vectors matches the input.
 */
public final class LangChain4jEmbedder implements Embedder {

  private final EmbeddingModel model;
  private final int dimension;
  private final String name;

  /** Convenience constructor; queries {@code model.dimension()} and uses a generic name. */
  public LangChain4jEmbedder(EmbeddingModel model) {
    this(model, dimensionOf(model), defaultName(model));
  }

  /** Full constructor; useful when {@code model.dimension()} is unreliable or for clearer logs. */
  public LangChain4jEmbedder(EmbeddingModel model, int dimension, String name) {
    Objects.requireNonNull(model, "model");
    if (dimension <= 0) {
      throw new IllegalArgumentException("dimension must be > 0");
    }
    if (name == null || name.isBlank()) {
      throw new IllegalArgumentException("name must be non-blank");
    }
    this.model = model;
    this.dimension = dimension;
    this.name = name;
  }

  @Override
  public String name() {
    return name;
  }

  @Override
  public int dimension() {
    return dimension;
  }

  @Override
  public List<float[]> embedAll(List<IngestDoc> docs) {
    if (docs.isEmpty()) {
      return List.of();
    }
    List<TextSegment> segments = new ArrayList<>(docs.size());
    for (int i = 0; i < docs.size(); i++) {
      IngestDoc d = docs.get(i);
      String text = d.text();
      if (text == null) {
        throw new IllegalArgumentException(
            "doc["
                + i
                + "] id="
                + d.id()
                + " has no text payload (LangChain4jEmbedder requires text)");
      }
      segments.add(TextSegment.from(text));
    }
    List<Embedding> response = model.embedAll(segments).content();
    if (response.size() != docs.size()) {
      throw new IllegalStateException(
          "embedding model returned " + response.size() + " vectors for " + docs.size() + " docs");
    }
    List<float[]> out = new ArrayList<>(response.size());
    for (Embedding e : response) {
      float[] v = e.vector();
      if (v.length != dimension) {
        throw new IllegalStateException(
            "embedding model returned vector of length "
                + v.length
                + " != configured "
                + dimension);
      }
      out.add(v);
    }
    return out;
  }

  private static int dimensionOf(EmbeddingModel model) {
    Objects.requireNonNull(model, "model");
    return model.dimension();
  }

  private static String defaultName(EmbeddingModel model) {
    return "langchain4j:" + model.getClass().getSimpleName();
  }
}
