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
package com.integrallis.vectors.hnsw;

import com.integrallis.vectors.storage.store.VectorStore;

/**
 * {@link RandomAccessVectors} implementation backed by a {@link VectorStore}. Copies vectors into a
 * reusable buffer on each call to {@link #getVector(int)}.
 *
 * <p>Not thread-safe: the shared buffer means concurrent calls would overwrite each other's data.
 */
public final class VectorStoreVectors implements RandomAccessVectors {

  private final VectorStore store;
  private final float[] buffer;

  public VectorStoreVectors(VectorStore store) {
    this.store = store;
    this.buffer = new float[store.dimension()];
  }

  @Override
  public int size() {
    return store.size();
  }

  @Override
  public int dimension() {
    return store.dimension();
  }

  @Override
  public float[] getVector(int ordinal) {
    store.getVector(ordinal, buffer);
    return buffer;
  }
}
