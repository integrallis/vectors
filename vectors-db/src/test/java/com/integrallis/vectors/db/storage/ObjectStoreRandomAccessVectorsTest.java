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
package com.integrallis.vectors.db.storage;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.vectors.storage.backend.HeapStorageBackend;
import com.integrallis.vectors.storage.memory.AlignmentUtil;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Random;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class ObjectStoreRandomAccessVectorsTest {

  private static final String KEY = "vectors.bin";

  /** Encodes vectors into the exact {@code vectors.bin} layout: 64-byte-aligned LE float32 rows. */
  private static byte[] encodeVectorsBin(float[][] vectors, int dim) {
    long stride = AlignmentUtil.alignUp((long) dim * Float.BYTES, AlignmentUtil.VECTOR_ALIGNMENT);
    byte[] blob = new byte[(int) (stride * vectors.length)];
    ByteBuffer buf = ByteBuffer.wrap(blob).order(ByteOrder.LITTLE_ENDIAN);
    for (int i = 0; i < vectors.length; i++) {
      buf.position((int) (i * stride));
      for (int d = 0; d < dim; d++) {
        buf.putFloat(vectors[i][d]);
      }
      // remaining bytes to stride stay zero (alignment padding)
    }
    return blob;
  }

  private static float[][] randomVectors(int n, int dim, long seed) {
    Random r = new Random(seed);
    float[][] v = new float[n][dim];
    for (int i = 0; i < n; i++) {
      for (int d = 0; d < dim; d++) {
        v[i][d] = r.nextFloat() * 2f - 1f;
      }
    }
    return v;
  }

  @Test
  void rangedGetReturnsExactVectors_paddedAndUnpaddedDims() throws Exception {
    // dim=128 -> stride 512 (no padding); dim=100 -> stride 448 (48 bytes padding). Cover both.
    for (int dim : new int[] {128, 100, 1}) {
      int n = 200;
      float[][] vectors = randomVectors(n, dim, 7L + dim);
      HeapStorageBackend backend = new HeapStorageBackend();
      backend.put(KEY, encodeVectorsBin(vectors, dim));

      ObjectStoreRandomAccessVectors ra = new ObjectStoreRandomAccessVectors(backend, KEY, n, dim);
      assertThat(ra.size()).isEqualTo(n);
      assertThat(ra.dimension()).isEqualTo(dim);
      assertThat(ra.supportsSegments()).isFalse();
      assertThat(ra.vectorSegment(0)).isNull();

      for (int i = 0; i < n; i++) {
        // copy out (getVector shares a per-thread scratch buffer)
        float[] got = ra.getVector(i).clone();
        assertThat(got).as("dim=%d ordinal=%d", dim, i).containsExactly(vectors[i]);
      }
    }
  }

  @Test
  void outOfRangeOrdinalThrows() throws Exception {
    int n = 10;
    int dim = 32;
    HeapStorageBackend backend = new HeapStorageBackend();
    backend.put(KEY, encodeVectorsBin(randomVectors(n, dim, 3L), dim));
    ObjectStoreRandomAccessVectors ra = new ObjectStoreRandomAccessVectors(backend, KEY, n, dim);

    assertThatThrownBy(() -> ra.getVector(n)).isInstanceOf(IndexOutOfBoundsException.class);
    assertThatThrownBy(() -> ra.getVector(-1)).isInstanceOf(IndexOutOfBoundsException.class);
  }

  @Test
  void perThreadScratchIsolatesConcurrentReaders() throws Exception {
    int n = 500;
    int dim = 64;
    float[][] vectors = randomVectors(n, dim, 11L);
    HeapStorageBackend backend = new HeapStorageBackend();
    backend.put(KEY, encodeVectorsBin(vectors, dim));
    ObjectStoreRandomAccessVectors ra = new ObjectStoreRandomAccessVectors(backend, KEY, n, dim);

    var pool = Executors.newFixedThreadPool(8);
    try {
      Future<?>[] tasks = new Future<?>[8];
      for (int t = 0; t < tasks.length; t++) {
        final int base = t;
        tasks[t] =
            pool.submit(
                () -> {
                  for (int rep = 0; rep < 1000; rep++) {
                    int ord = (base * 131 + rep * 17) % n;
                    float[] got = ra.getVector(ord).clone();
                    assertThat(got).containsExactly(vectors[ord]);
                  }
                });
      }
      for (Future<?> f : tasks) {
        f.get();
      }
    } catch (ExecutionException e) {
      throw new AssertionError(e.getCause());
    } finally {
      pool.shutdownNow();
    }
  }
}
