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

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.VectorEncoding;
import com.integrallis.vectors.storage.store.VectorStore;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class VectorStoreVectorsTest {

  /** Minimal in-memory {@link VectorStore} exposing only the float[] read path used here. */
  private static final class FakeStore implements VectorStore {
    private final float[][] data;

    FakeStore(float[][] data) {
      this.data = data;
    }

    @Override
    public int size() {
      return data.length;
    }

    @Override
    public int dimension() {
      return data[0].length;
    }

    @Override
    public VectorEncoding encoding() {
      return VectorEncoding.FLOAT32;
    }

    @Override
    public int vectorByteSize() {
      return data[0].length * Float.BYTES;
    }

    @Override
    public void getVector(int ordinal, float[] dst) {
      System.arraycopy(data[ordinal], 0, dst, 0, dst.length);
    }

    @Override
    public void getVector(int ordinal, byte[] dst) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void getVector(int ordinal, MemorySegment dst, long dstByteOffset) {
      throw new UnsupportedOperationException();
    }

    @Override
    public MemorySegment vectorSlice(int ordinal) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void close() {}
  }

  @Test
  void exposesStoreSizeDimensionAndVectors() {
    float[][] data = {{1f, 2f, 3f}, {4f, 5f, 6f}, {7f, 8f, 9f}};
    VectorStoreVectors v = new VectorStoreVectors(new FakeStore(data));

    assertThat(v.size()).isEqualTo(3);
    assertThat(v.dimension()).isEqualTo(3);
    assertThat(v.getVector(0)).containsExactly(1f, 2f, 3f);
    assertThat(v.getVector(2)).containsExactly(7f, 8f, 9f);
  }

  @Test
  void reusesASharedReturnBuffer() {
    float[][] data = {{1f, 1f}, {2f, 2f}};
    VectorStoreVectors v = new VectorStoreVectors(new FakeStore(data));

    // Default RandomAccessVectors contract: the returned array may be overwritten across calls.
    assertThat(v.sharesReturnBuffer()).isTrue();
    float[] first = v.getVector(1);
    float[] second = v.getVector(0);
    assertThat(second).isSameAs(first); // one reusable buffer, overwritten on each call
    assertThat(second).containsExactly(1f, 1f); // now holds ordinal 0's data
  }
}
