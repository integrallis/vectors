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
package com.integrallis.vectors.demo.rag.model;

/** Types of semantic caching available in the demo. */
public enum CacheType {
  /** No caching - always call LLM */
  NONE("No Cache", "Always call LLM, no caching"),

  /** Local in-memory semantic cache */
  LOCAL("Local Cache", "In-memory semantic cache (local)");

  private final String displayName;
  private final String description;

  CacheType(String displayName, String description) {
    this.displayName = displayName;
    this.description = description;
  }

  public String getDisplayName() {
    return displayName;
  }

  public String getDescription() {
    return description;
  }

  @Override
  public String toString() {
    return displayName;
  }
}
