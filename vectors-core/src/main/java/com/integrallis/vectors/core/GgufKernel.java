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

/**
 * Marker interface for GGUF (GGML Universal Format) quantized matrix-vector and batch operations.
 *
 * <p>Supports Q4_0, Q4_K, Q5_0, Q5_K, Q6_K, and Q8_0 quantization formats, including mixed-format
 * batched operations (e.g. Q4_0 weights with Q8_0 activations).
 *
 * <p>All method implementations currently live in {@link VectorUtilSupport} as default scalar
 * fallbacks. This interface exists to enable future composition-based refactoring where individual
 * kernel implementations can be selected and composed at runtime.
 *
 * @see VectorUtilSupport
 */
public interface GgufKernel {}
