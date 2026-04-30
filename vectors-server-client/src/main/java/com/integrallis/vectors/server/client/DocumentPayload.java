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
package com.integrallis.vectors.server.client;

import java.util.Map;

/**
 * Document payload for upsert requests.
 *
 * @param id the document identifier
 * @param vector the embedding vector
 * @param text optional text content
 * @param metadata optional key-value metadata
 * @param blob optional Base64-encoded binary data (e.g. image bytes)
 */
public record DocumentPayload(
    String id, float[] vector, String text, Map<String, Object> metadata, String blob) {

  /** Convenience factory without blob. */
  public static DocumentPayload of(String id, float[] vector, String text) {
    return new DocumentPayload(id, vector, text, null, null);
  }

  /** Convenience factory with metadata, no blob. */
  public static DocumentPayload of(
      String id, float[] vector, String text, Map<String, Object> metadata) {
    return new DocumentPayload(id, vector, text, metadata, null);
  }
}
