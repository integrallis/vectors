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

import java.util.ServiceLoader;

/**
 * SPI for serializing and deserializing {@link CassetteRecord} values.
 *
 * <p>Implementations are discovered via {@link ServiceLoader}. Ship exactly one implementation on
 * the classpath: {@code vectors-vcr-serde-avaje} (default, Avaje JsonB) or {@code
 * vectors-vcr-serde-jackson} (alternate, Jackson).
 *
 * <p>The on-disk shape is implementation-defined. The SPI contract is "whatever wrote it reads it
 * back" — cross-implementation interoperability is not guaranteed.
 */
public interface CassetteSerializer {

  /**
   * Serializes a cassette record to bytes.
   *
   * @param record the record to serialize (must not be null)
   * @return the serialized bytes
   */
  byte[] serialize(CassetteRecord record);

  /**
   * Deserializes bytes back into a cassette record.
   *
   * @param bytes the serialized bytes (must not be null)
   * @return the decoded record
   */
  CassetteRecord deserialize(byte[] bytes);

  /**
   * @return the MIME content type produced by this serializer
   */
  default String contentType() {
    return "application/json";
  }

  /**
   * Loads the first {@link CassetteSerializer} discovered on the classpath via {@link
   * ServiceLoader}.
   *
   * @return the loaded serializer
   * @throws IllegalStateException if no serializer implementation is present on the classpath
   */
  static CassetteSerializer load() {
    return ServiceLoader.load(CassetteSerializer.class)
        .findFirst()
        .orElseThrow(
            () ->
                new IllegalStateException(
                    "No CassetteSerializer on classpath; add vectors-vcr-serde-avaje or"
                        + " vectors-vcr-serde-jackson"));
  }
}
