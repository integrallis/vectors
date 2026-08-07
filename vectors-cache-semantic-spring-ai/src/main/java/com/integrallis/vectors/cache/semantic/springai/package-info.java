/*
 * Copyright 2026 Integrallis Software, LLC
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

/**
 * Similarity-matched response caching.
 *
 * <p>Kept apart from the exact-key decorators so the two cannot be confused: those key on a
 * normalized prompt string and only ever hit on a verbatim repeat, while these embed the request
 * and hit on a paraphrase.
 */
package com.integrallis.vectors.cache.semantic.springai;
