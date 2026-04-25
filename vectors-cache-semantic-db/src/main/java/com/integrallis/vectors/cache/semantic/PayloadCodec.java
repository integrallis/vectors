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
package com.integrallis.vectors.cache.semantic;

/**
 * Bidirectional codec between cache payload {@code V} and a String representation stored in a
 * {@link com.integrallis.vectors.db.MetadataValue.Str} on the cached document.
 *
 * <p>Simple and format-agnostic. Callers pick their own serialization (raw string, JSON, Base64 of
 * a binary protocol, Avro, etc.). For {@code String} payloads use {@link #identity()}.
 */
public interface PayloadCodec<V> {

  /** Returns an identity codec for {@code String} payloads. */
  static PayloadCodec<String> identity() {
    return new PayloadCodec<>() {
      @Override
      public String encode(String value) {
        return value;
      }

      @Override
      public String decode(String encoded) {
        return encoded;
      }
    };
  }

  String encode(V value);

  V decode(String encoded);
}
