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
package com.integrallis.vectors.core;

/** Immutable SIMD and GGUF execution facts resolved by vectors-core at startup. */
public record VectorRuntimeCapabilities(
    String providerName,
    boolean vectorApi,
    int configuredMaxVectorBits,
    int preferredVectorBits,
    int activeVectorBits,
    boolean fastVectorFma,
    boolean fastScalarFma,
    boolean sve,
    boolean q4ShortPairwiseSupported,
    boolean ggufParallel,
    String ggufExecutor,
    int ggufThreads,
    int ggufChunksPerThread) {

  public VectorRuntimeCapabilities {
    if (providerName == null || providerName.isBlank()) {
      throw new IllegalArgumentException("providerName must not be blank");
    }
    providerName = providerName.trim();
    if (configuredMaxVectorBits <= 0 || preferredVectorBits <= 0) {
      throw new IllegalArgumentException("configured and preferred vector bits must be > 0");
    }
    if (activeVectorBits < 0) {
      throw new IllegalArgumentException("activeVectorBits must be >= 0");
    }
    if (vectorApi != (activeVectorBits > 0)) {
      throw new IllegalArgumentException("vectorApi and activeVectorBits disagree");
    }
    if (ggufExecutor == null || ggufExecutor.isBlank()) {
      throw new IllegalArgumentException("ggufExecutor must not be blank");
    }
    ggufExecutor = ggufExecutor.trim();
    if (ggufThreads <= 0 || ggufChunksPerThread <= 0) {
      throw new IllegalArgumentException("GGUF thread and chunk counts must be > 0");
    }
  }
}
