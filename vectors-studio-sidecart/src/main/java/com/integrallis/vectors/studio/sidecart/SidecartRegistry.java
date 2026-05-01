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
package com.integrallis.vectors.studio.sidecart;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/**
 * In-memory mapping from collection name to a {@link SidecartSource}. The web layer uses the
 * registry to resolve sidecart records when answering blob requests; collections without a
 * registered source fall back to whatever the active {@code StudioBackend} can provide.
 */
public final class SidecartRegistry implements AutoCloseable {

  private final ConcurrentHashMap<String, SidecartSource> sources = new ConcurrentHashMap<>();

  public static SidecartRegistry empty() {
    return new SidecartRegistry();
  }

  /** Builds a registry from a pre-resolved name → source map. */
  public static SidecartRegistry of(Map<String, SidecartSource> bindings) {
    SidecartRegistry r = new SidecartRegistry();
    for (Map.Entry<String, SidecartSource> e : bindings.entrySet()) {
      r.bind(e.getKey(), e.getValue());
    }
    return r;
  }

  public SidecartRegistry bind(String collection, SidecartSource source) {
    Objects.requireNonNull(collection, "collection");
    Objects.requireNonNull(source, "source");
    SidecartSource prev = sources.put(collection, source);
    if (prev != null) {
      try {
        prev.close();
      } catch (RuntimeException ignored) {
      }
    }
    return this;
  }

  public Optional<SidecartSource> get(String collection) {
    if (collection == null) return Optional.empty();
    return Optional.ofNullable(sources.get(collection));
  }

  public boolean isEmpty() {
    return sources.isEmpty();
  }

  /** Returns a snapshot of the current bindings (collection → source). */
  public Map<String, SidecartSource> bindings() {
    return new HashMap<>(sources);
  }

  @Override
  public void close() {
    for (SidecartSource s : sources.values()) {
      try {
        s.close();
      } catch (RuntimeException ignored) {
      }
    }
    sources.clear();
  }
}
