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
package com.integrallis.vectors.ingest.embedders;

import com.integrallis.vectors.ingest.Embedder;
import com.integrallis.vectors.ingest.IngestDoc;
import java.util.ArrayList;
import java.util.List;

/**
 * Embedder for sources whose docs already carry a {@link IngestDoc#precomputedVector()}. Throws
 * {@link IllegalArgumentException} if any input doc lacks a precomputed vector or if its dimension
 * disagrees with the configured one.
 */
public final class PrecomputedEmbedder implements Embedder {

  private final int dimension;
  private final String name;

  public PrecomputedEmbedder(int dimension) {
    this(dimension, "precomputed");
  }

  public PrecomputedEmbedder(int dimension, String name) {
    if (dimension <= 0) {
      throw new IllegalArgumentException("dimension must be > 0");
    }
    if (name == null || name.isBlank()) {
      throw new IllegalArgumentException("name must be non-blank");
    }
    this.dimension = dimension;
    this.name = name;
  }

  @Override
  public String name() {
    return name;
  }

  @Override
  public int dimension() {
    return dimension;
  }

  @Override
  public List<float[]> embedAll(List<IngestDoc> docs) {
    List<float[]> out = new ArrayList<>(docs.size());
    for (int i = 0; i < docs.size(); i++) {
      IngestDoc d = docs.get(i);
      float[] v = d.precomputedVector();
      if (v == null) {
        throw new IllegalArgumentException(
            "doc[" + i + "] id=" + d.id() + " has no precomputedVector");
      }
      if (v.length != dimension) {
        throw new IllegalArgumentException(
            "doc[" + i + "] id=" + d.id() + " vector length " + v.length + " != " + dimension);
      }
      out.add(v);
    }
    return out;
  }
}
