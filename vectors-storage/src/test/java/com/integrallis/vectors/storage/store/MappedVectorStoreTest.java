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
package com.integrallis.vectors.storage.store;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import com.integrallis.vectors.core.VectorEncoding;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/** Tests for {@link MappedVectorStore} and {@link VectorStoreWriter}. */
class MappedVectorStoreTest {

  @TempDir Path tempDir;

  @Nested
  class Float32Vectors {

    @ParameterizedTest
    @ValueSource(ints = {3, 4, 16, 100, 128, 256, 768, 1536})
    void writeAndRead_variousDimensions(int dim) throws IOException {
      int count = 50;
      float[][] vectors = generateFloatVectors(count, dim, 42L);

      Path file = tempDir.resolve("float32_d" + dim + ".bin");
      try (var writer = VectorStoreWriter.open(file, dim, VectorEncoding.FLOAT32)) {
        for (float[] v : vectors) {
          writer.writeVector(v);
        }
        assertThat(writer.count()).isEqualTo(count);
      }

      try (var store = MappedVectorStore.open(file, count, dim, VectorEncoding.FLOAT32)) {
        assertThat(store.size()).isEqualTo(count);
        assertThat(store.dimension()).isEqualTo(dim);
        assertThat(store.encoding()).isEqualTo(VectorEncoding.FLOAT32);
        assertThat(store.vectorByteSize()).isEqualTo(dim * Float.BYTES);

        // Verify each vector
        float[] readBack = new float[dim];
        for (int i = 0; i < count; i++) {
          store.getVector(i, readBack);
          assertThat(readBack).as("vector %d", i).containsExactly(vectors[i]);
        }
      }
    }

    @Test
    void vectorSlice_returnsZeroCopySegment() throws IOException {
      int dim = 128;
      float[][] vectors = generateFloatVectors(10, dim, 99L);

      Path file = tempDir.resolve("slice_test.bin");
      try (var writer = VectorStoreWriter.open(file, dim, VectorEncoding.FLOAT32)) {
        for (float[] v : vectors) writer.writeVector(v);
      }

      try (var store = MappedVectorStore.open(file, 10, dim, VectorEncoding.FLOAT32)) {
        for (int i = 0; i < 10; i++) {
          MemorySegment slice = store.vectorSlice(i);
          assertThat(slice.byteSize()).isEqualTo((long) dim * Float.BYTES);

          // Verify slice contents
          for (int d = 0; d < dim; d++) {
            float expected = vectors[i][d];
            float actual =
                slice.get(
                    ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(java.nio.ByteOrder.LITTLE_ENDIAN),
                    (long) d * Float.BYTES);
            assertThat(actual).isCloseTo(expected, within(0.0f));
          }
        }
      }
    }

    @Test
    void getVector_intoMemorySegment() throws IOException {
      int dim = 64;
      float[][] vectors = generateFloatVectors(5, dim, 77L);

      Path file = tempDir.resolve("memseg_dst.bin");
      try (var writer = VectorStoreWriter.open(file, dim, VectorEncoding.FLOAT32)) {
        for (float[] v : vectors) writer.writeVector(v);
      }

      try (var store = MappedVectorStore.open(file, 5, dim, VectorEncoding.FLOAT32);
          Arena arena = Arena.ofConfined()) {
        MemorySegment dst = arena.allocate((long) dim * Float.BYTES);
        for (int i = 0; i < 5; i++) {
          store.getVector(i, dst, 0);
          for (int d = 0; d < dim; d++) {
            assertThat(dst.getAtIndex(ValueLayout.JAVA_FLOAT, d)).isEqualTo(vectors[i][d]);
          }
        }
      }
    }

    @Test
    void alignment_strideIs64ByteAligned() throws IOException {
      // dim=3 has raw size 12 bytes, stride should be 64
      int dim = 3;
      Path file = tempDir.resolve("stride.bin");
      float[] v = {1.0f, 2.0f, 3.0f};
      try (var writer = VectorStoreWriter.open(file, dim, VectorEncoding.FLOAT32)) {
        writer.writeVector(v);
        writer.writeVector(v);
      }

      try (var store = MappedVectorStore.open(file, 2, dim, VectorEncoding.FLOAT32)) {
        assertThat(store.stride()).isEqualTo(64);
        // Both vectors should read correctly despite padding
        float[] readBack = new float[dim];
        store.getVector(0, readBack);
        assertThat(readBack).containsExactly(v);
        store.getVector(1, readBack);
        assertThat(readBack).containsExactly(v);
      }
    }

    @Test
    void ordinalOutOfBounds_throws() throws IOException {
      int dim = 4;
      Path file = tempDir.resolve("bounds.bin");
      try (var writer = VectorStoreWriter.open(file, dim, VectorEncoding.FLOAT32)) {
        writer.writeVector(new float[] {1, 2, 3, 4});
      }

      try (var store = MappedVectorStore.open(file, 1, dim, VectorEncoding.FLOAT32)) {
        float[] buf = new float[dim];
        assertThatThrownBy(() -> store.getVector(-1, buf))
            .isInstanceOf(IndexOutOfBoundsException.class);
        assertThatThrownBy(() -> store.getVector(1, buf))
            .isInstanceOf(IndexOutOfBoundsException.class);
      }
    }

    @Test
    void getVectorRejectsUndersizedFloatDestination() throws IOException {
      int dim = 4;
      Path file = tempDir.resolve("small_float_dst.bin");
      try (var writer = VectorStoreWriter.open(file, dim, VectorEncoding.FLOAT32)) {
        writer.writeVector(new float[] {1, 2, 3, 4});
      }

      try (var store = MappedVectorStore.open(file, 1, dim, VectorEncoding.FLOAT32)) {
        assertThatThrownBy(() -> store.getVector(0, new float[dim - 1]))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("destination float length");
      }
    }
  }

  @Nested
  class Int8Vectors {

    @Test
    void writeAndRead_int8() throws IOException {
      int dim = 128;
      int count = 20;
      byte[][] vectors = generateByteVectors(count, dim, 42L);

      Path file = tempDir.resolve("int8.bin");
      try (var writer = VectorStoreWriter.open(file, dim, VectorEncoding.INT8)) {
        for (byte[] v : vectors) writer.writeVector(v);
      }

      try (var store = MappedVectorStore.open(file, count, dim, VectorEncoding.INT8)) {
        assertThat(store.size()).isEqualTo(count);
        assertThat(store.dimension()).isEqualTo(dim);
        assertThat(store.encoding()).isEqualTo(VectorEncoding.INT8);

        byte[] readBack = new byte[dim];
        for (int i = 0; i < count; i++) {
          store.getVector(i, readBack);
          assertThat(readBack).as("vector %d", i).containsExactly(vectors[i]);
        }
      }
    }

    @Test
    void int8_getVectorAsFloat() throws IOException {
      int dim = 4;
      Path file = tempDir.resolve("int8_float.bin");
      byte[] v = {10, -20, 30, -40};
      try (var writer = VectorStoreWriter.open(file, dim, VectorEncoding.INT8)) {
        writer.writeVector(v);
      }

      try (var store = MappedVectorStore.open(file, 1, dim, VectorEncoding.INT8)) {
        float[] readBack = new float[dim];
        store.getVector(0, readBack);
        assertThat(readBack).containsExactly(10f, -20f, 30f, -40f);
      }
    }
  }

  @Nested
  class BinaryVectors {

    @Test
    void writeAndRead_binaryRawBytes() throws IOException {
      int dim = 70;
      byte[] bits = new byte[VectorEncoding.BINARY.vectorByteSize(dim)];
      for (int i = 0; i < bits.length; i++) {
        bits[i] = (byte) (0xA0 + i);
      }

      Path file = tempDir.resolve("binary.bin");
      try (var writer = VectorStoreWriter.open(file, dim, VectorEncoding.BINARY)) {
        writer.writeVector(bits);
      }

      try (var store = MappedVectorStore.open(file, 1, dim, VectorEncoding.BINARY)) {
        assertThat(store.vectorByteSize()).isEqualTo(16);
        byte[] readBack = new byte[bits.length];
        store.getVector(0, readBack);
        assertThat(readBack).containsExactly(bits);
        assertThat(store.vectorSlice(0).byteSize()).isEqualTo(bits.length);
        assertThatThrownBy(() -> store.getVector(0, new float[dim]))
            .isInstanceOf(UnsupportedOperationException.class);
      }
    }

    @Test
    void getVectorRejectsUndersizedByteDestination() throws IOException {
      int dim = 70;
      Path file = tempDir.resolve("small_byte_dst.bin");
      try (var writer = VectorStoreWriter.open(file, dim, VectorEncoding.BINARY)) {
        writer.writeVector(new byte[VectorEncoding.BINARY.vectorByteSize(dim)]);
      }

      try (var store = MappedVectorStore.open(file, 1, dim, VectorEncoding.BINARY)) {
        assertThatThrownBy(() -> store.getVector(0, new byte[store.vectorByteSize() - 1]))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("destination byte length");
      }
    }
  }

  @Nested
  class FileValidation {

    @Test
    void emptyStoreCanOpenEmptyFile() throws IOException {
      Path file = tempDir.resolve("empty.bin");
      Files.write(file, new byte[0]);

      try (var store = MappedVectorStore.open(file, 0, 4, VectorEncoding.FLOAT32)) {
        assertThat(store.size()).isZero();
        assertThatThrownBy(() -> store.vectorSlice(0))
            .isInstanceOf(IndexOutOfBoundsException.class);
      }
    }

    @Test
    void undersizedFileIsRejectedAtOpen() throws IOException {
      Path file = tempDir.resolve("undersized.bin");
      Files.write(file, new byte[Float.BYTES * 3]);

      assertThatThrownBy(() -> MappedVectorStore.open(file, 1, 4, VectorEncoding.FLOAT32, 0, 4))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("requires 16 bytes but file has 12");
    }

    @Test
    void getVectorRejectsUndersizedMemorySegmentDestination() throws IOException {
      int dim = 4;
      Path file = tempDir.resolve("small_segment_dst.bin");
      try (var writer = VectorStoreWriter.open(file, dim, VectorEncoding.FLOAT32)) {
        writer.writeVector(new float[] {1, 2, 3, 4});
      }

      try (var store = MappedVectorStore.open(file, 1, dim, VectorEncoding.FLOAT32);
          Arena arena = Arena.ofConfined()) {
        MemorySegment dst = arena.allocate(store.vectorByteSize() - 1);
        assertThatThrownBy(() -> store.getVector(0, dst, 0))
            .isInstanceOf(IndexOutOfBoundsException.class)
            .hasMessageContaining("outside segment length");
      }
    }
  }

  @Nested
  class ConcurrentAccess {

    @Test
    void concurrentReads_allCorrect() throws Exception {
      int dim = 128;
      int count = 1000;
      float[][] vectors = generateFloatVectors(count, dim, 42L);

      Path file = tempDir.resolve("concurrent.bin");
      try (var writer = VectorStoreWriter.open(file, dim, VectorEncoding.FLOAT32)) {
        for (float[] v : vectors) writer.writeVector(v);
      }

      try (var store = MappedVectorStore.open(file, count, dim, VectorEncoding.FLOAT32)) {
        int numThreads = 8;
        Thread[] threads = new Thread[numThreads];
        boolean[] results = new boolean[numThreads];

        for (int t = 0; t < numThreads; t++) {
          final int threadId = t;
          threads[t] =
              Thread.ofVirtual()
                  .start(
                      () -> {
                        float[] buf = new float[dim];
                        boolean ok = true;
                        int start = threadId * (count / numThreads);
                        int end = start + (count / numThreads);
                        for (int i = start; i < end; i++) {
                          store.getVector(i, buf);
                          for (int d = 0; d < dim; d++) {
                            if (Float.compare(buf[d], vectors[i][d]) != 0) {
                              ok = false;
                              break;
                            }
                          }
                          if (!ok) break;
                        }
                        results[threadId] = ok;
                      });
        }

        for (Thread thread : threads) thread.join();
        for (int t = 0; t < numThreads; t++) {
          assertThat(results[t]).as("Thread %d", t).isTrue();
        }
      }
    }
  }

  @Nested
  class CustomAlignment {

    @Test
    void noAlignment_packedLayout() throws IOException {
      int dim = 3;
      float[] v1 = {1.0f, 2.0f, 3.0f};
      float[] v2 = {4.0f, 5.0f, 6.0f};

      Path file = tempDir.resolve("packed.bin");
      // alignment=4 means natural float alignment only, no extra padding
      try (var writer = VectorStoreWriter.open(file, dim, VectorEncoding.FLOAT32, 4)) {
        writer.writeVector(v1);
        writer.writeVector(v2);
      }

      try (var store = MappedVectorStore.open(file, 2, dim, VectorEncoding.FLOAT32, 0, 4)) {
        float[] readBack = new float[dim];
        store.getVector(0, readBack);
        assertThat(readBack).containsExactly(v1);
        store.getVector(1, readBack);
        assertThat(readBack).containsExactly(v2);

        // With alignment=4, stride should equal raw size (12 bytes) since 12 is a multiple of 4
        assertThat(store.stride()).isEqualTo(12);
      }
    }
  }

  @Nested
  class WriterValidation {

    @Test
    void wrongDimension_throws() throws IOException {
      Path file = tempDir.resolve("wrong_dim.bin");
      try (var writer = VectorStoreWriter.open(file, 3, VectorEncoding.FLOAT32)) {
        assertThatThrownBy(() -> writer.writeVector(new float[] {1, 2}))
            .isInstanceOf(IllegalArgumentException.class);
      }
    }

    @Test
    void wrongEncoding_throws() throws IOException {
      Path file = tempDir.resolve("wrong_enc.bin");
      try (var writer = VectorStoreWriter.open(file, 3, VectorEncoding.INT8)) {
        assertThatThrownBy(() -> writer.writeVector(new float[] {1, 2, 3}))
            .isInstanceOf(IllegalArgumentException.class);
      }
    }
  }

  // --- Helper methods ---

  private static float[][] generateFloatVectors(int count, int dim, long seed) {
    Random rng = new Random(seed);
    float[][] vectors = new float[count][dim];
    for (int i = 0; i < count; i++) {
      for (int d = 0; d < dim; d++) {
        vectors[i][d] = rng.nextFloat() * 2 - 1; // [-1, 1]
      }
    }
    return vectors;
  }

  private static byte[][] generateByteVectors(int count, int dim, long seed) {
    Random rng = new Random(seed);
    byte[][] vectors = new byte[count][dim];
    for (byte[] v : vectors) {
      rng.nextBytes(v);
    }
    return vectors;
  }
}
