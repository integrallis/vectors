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
package com.integrallis.vectors.hybrid.text;

/**
 * Factory for creating {@link TextIndexSpi} instances. Implementations are discovered via {@link
 * java.util.ServiceLoader}.
 */
public interface TextIndexSpiFactory {

  /**
   * Create a new text index instance for the given collection name.
   *
   * @param collectionName the name of the collection this index belongs to
   * @return a new text index instance
   */
  TextIndexSpi create(String collectionName);

  /**
   * Create a new text index instance backed by persistent storage in the given data directory. When
   * {@code dataDir} is {@code null}, the implementation falls back to in-memory storage.
   *
   * @param collectionName the name of the collection this index belongs to
   * @param dataDir directory in which to store the index data, or {@code null} for in-memory
   * @return a new text index instance
   */
  default TextIndexSpi create(String collectionName, java.nio.file.Path dataDir) {
    return create(collectionName);
  }
}
