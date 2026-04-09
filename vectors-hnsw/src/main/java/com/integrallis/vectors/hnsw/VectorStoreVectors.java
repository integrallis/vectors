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
