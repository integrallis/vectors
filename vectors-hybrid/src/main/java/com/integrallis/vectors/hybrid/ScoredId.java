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
package com.integrallis.vectors.hybrid;

import java.util.Objects;

/**
 * A scored document identifier — the universal currency across retrievers and fusion strategies.
 *
 * @param id the document identifier
 * @param score the relevance score (higher is better)
 */
public record ScoredId(String id, float score) {

  public ScoredId {
    Objects.requireNonNull(id, "id");
  }
}
