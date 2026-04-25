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

import java.util.Map;
import java.util.Objects;

/**
 * Sealed hierarchy of recorded cassette payloads.
 *
 * <p>Three payload shapes are supported:
 *
 * <ul>
 *   <li>{@link Embedding} — a single dense float vector plus metadata
 *   <li>{@link BatchEmbedding} — a batch of dense float vectors plus metadata
 *   <li>{@link Chat} — a chat prompt/response pair plus metadata
 * </ul>
 */
public sealed interface CassetteRecord
    permits CassetteRecord.Embedding, CassetteRecord.BatchEmbedding, CassetteRecord.Chat {

  /**
   * @return the test identifier that produced this record
   */
  String testId();

  /**
   * @return the model name (human-readable, used for metadata only)
   */
  String model();

  /**
   * @return the epoch-milli timestamp at which the record was produced
   */
  long timestamp();

  /** A single embedding vector. */
  record Embedding(String testId, String model, long timestamp, float[] embedding)
      implements CassetteRecord {
    /** Compact constructor with null/shape checks. */
    public Embedding {
      Objects.requireNonNull(testId, "testId");
      Objects.requireNonNull(model, "model");
      Objects.requireNonNull(embedding, "embedding");
    }
  }

  /** A batch of embedding vectors produced by a single batch call. */
  record BatchEmbedding(String testId, String model, long timestamp, float[][] embeddings)
      implements CassetteRecord {
    /** Compact constructor with null/shape checks. */
    public BatchEmbedding {
      Objects.requireNonNull(testId, "testId");
      Objects.requireNonNull(model, "model");
      Objects.requireNonNull(embeddings, "embeddings");
    }
  }

  /** A chat exchange (prompt + response pair). */
  record Chat(
      String testId,
      String model,
      long timestamp,
      String prompt,
      String response,
      Map<String, String> metadata)
      implements CassetteRecord {
    /** Compact constructor defensively copies the metadata map and validates non-null text. */
    public Chat {
      Objects.requireNonNull(testId, "testId");
      Objects.requireNonNull(model, "model");
      Objects.requireNonNull(prompt, "prompt");
      Objects.requireNonNull(response, "response");
      metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
    }
  }
}
