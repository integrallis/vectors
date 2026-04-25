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
import static org.assertj.core.api.Assertions.assertThatExceptionOfType;
import static org.assertj.core.api.Assertions.assertThatIOException;
import static org.assertj.core.api.Assertions.assertThatNullPointerException;

import com.integrallis.vectors.core.VectorEncoding;
import com.integrallis.vectors.storage.store.VectorStoreWriter;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class MemorySegmentVectorsTest {

  private static final ValueLayout.OfFloat FLOAT_LE =
      ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

  private static Path writeVectorsBin(Path tmp, String name, float[][] vectors, int dim)
      throws IOException {
    Path file = tmp.resolve(name);
    try (VectorStoreWriter writer = VectorStoreWriter.open(file, dim, VectorEncoding.FLOAT32)) {
      for (float[] v : vectors) {
        writer.writeVector(v);
      }
    }
    return file;
  }

  private static float[][] randomVectors(int n, int dim, long seed) {
    Random rng = new Random(seed);
    float[][] out = new float[n][dim];
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < dim; j++) {
        out[i][j] = rng.nextFloat() * 2f - 1f;
      }
    }
    return out;
  }

  private static float[] sliceToFloats(MemorySegment slice, int dim) {
    float[] out = new float[dim];
    for (int i = 0; i < dim; i++) {
      out[i] = slice.get(FLOAT_LE, (long) i * Float.BYTES);
    }
    return out;
  }

  @Nested
  class RoundTrip {

    @Test
    void singleVectorNoPadding(@TempDir Path tmp) throws IOException {
      // dim=128 → stride=512 (no padding, since 128*4 == 512 == alignUp to 64)
      int dim = 128;
      float[][] vectors = randomVectors(1, dim, 1L);
      Path file = writeVectorsBin(tmp, "vectors.bin", vectors, dim);

      try (Arena arena = Arena.ofConfined()) {
        MemorySegmentVectors v = MemorySegmentVectors.open(file, 1, dim, arena);
        assertThat(v.size()).isEqualTo(1);
        assertThat(v.dimension()).isEqualTo(dim);
        assertThat(v.stride()).isEqualTo(512L);
        assertThat(v.rawVectorByteSize()).isEqualTo(512);
        assertThat(v.vectorOffsetFor(0)).isEqualTo(0L);

        MemorySegment slice = v.vectorSlice(0);
        assertThat(slice.byteSize()).isEqualTo(dim * Float.BYTES);
        assertThat(sliceToFloats(slice, dim)).containsExactly(vectors[0]);
      }
    }

    @Test
    void singleVectorWithPadding(@TempDir Path tmp) throws IOException {
      // dim=100 → raw=400, stride=alignUp(400, 64)=448, padding=48
      int dim = 100;
      float[][] vectors = randomVectors(1, dim, 2L);
      Path file = writeVectorsBin(tmp, "vectors.bin", vectors, dim);

      try (Arena arena = Arena.ofConfined()) {
        MemorySegmentVectors v = MemorySegmentVectors.open(file, 1, dim, arena);
        assertThat(v.stride()).isEqualTo(448L);
        assertThat(v.rawVectorByteSize()).isEqualTo(400);
        MemorySegment slice = v.vectorSlice(0);
        // Slice is exactly the raw bytes — no padding in the slice itself.
        assertThat(slice.byteSize()).isEqualTo(400L);
        assertThat(sliceToFloats(slice, dim)).containsExactly(vectors[0]);
      }
    }

    @Test
    void manyVectorsAligned(@TempDir Path tmp) throws IOException {
      int dim = 64;
      int n = 500;
      float[][] vectors = randomVectors(n, dim, 42L);
      Path file = writeVectorsBin(tmp, "vectors.bin", vectors, dim);

      try (Arena arena = Arena.ofConfined()) {
        MemorySegmentVectors v = MemorySegmentVectors.open(file, n, dim, arena);
        assertThat(v.size()).isEqualTo(n);
        assertThat(v.stride()).isEqualTo(256L); // 64*4 = 256, aligned

        for (int i = 0; i < n; i++) {
          MemorySegment slice = v.vectorSlice(i);
          assertThat(sliceToFloats(slice, dim)).as("vector %d", i).containsExactly(vectors[i]);
          assertThat(v.vectorOffsetFor(i)).isEqualTo((long) i * 256);
        }
      }
    }

    @Test
    void manyVectorsWithPadding(@TempDir Path tmp) throws IOException {
      // dim=7 → raw=28, stride=alignUp(28, 64)=64, padding=36
      int dim = 7;
      int n = 200;
      float[][] vectors = randomVectors(n, dim, 99L);
      Path file = writeVectorsBin(tmp, "vectors.bin", vectors, dim);

      try (Arena arena = Arena.ofConfined()) {
        MemorySegmentVectors v = MemorySegmentVectors.open(file, n, dim, arena);
        assertThat(v.stride()).isEqualTo(64L);
        assertThat(v.rawVectorByteSize()).isEqualTo(28);

        for (int i = 0; i < n; i++) {
          assertThat(sliceToFloats(v.vectorSlice(i), dim))
              .as("vector %d", i)
              .containsExactly(vectors[i]);
        }
      }
    }

    @Test
    void segmentAccessorMatchesSliceAccessor(@TempDir Path tmp) throws IOException {
      // Verifies the tight inner-loop access pattern:
      // segment().asSlice(vectorOffsetFor(i), rawVectorByteSize()) == vectorSlice(i)
      int dim = 32;
      int n = 50;
      float[][] vectors = randomVectors(n, dim, 7L);
      Path file = writeVectorsBin(tmp, "vectors.bin", vectors, dim);

      try (Arena arena = Arena.ofConfined()) {
        MemorySegmentVectors v = MemorySegmentVectors.open(file, n, dim, arena);
        MemorySegment whole = v.segment();
        int rawBytes = v.rawVectorByteSize();

        for (int i = 0; i < n; i++) {
          MemorySegment viaSlice = v.vectorSlice(i);
          MemorySegment viaOffset = whole.asSlice(v.vectorOffsetFor(i), rawBytes);
          // Byte-for-byte equality: both slices cover the same region of the same segment.
          assertThat(viaOffset.byteSize()).isEqualTo(viaSlice.byteSize());
          for (long b = 0; b < rawBytes; b++) {
            assertThat(viaOffset.get(ValueLayout.JAVA_BYTE, b))
                .as("byte %d of vector %d", b, i)
                .isEqualTo(viaSlice.get(ValueLayout.JAVA_BYTE, b));
          }
        }
      }
    }

    @Test
    void emptyVectorSet(@TempDir Path tmp) throws IOException {
      // size=0 with a file that has no vectors in it. VectorStoreWriter produces an empty file.
      int dim = 128;
      Path file = writeVectorsBin(tmp, "vectors.bin", new float[0][], dim);

      try (Arena arena = Arena.ofConfined()) {
        MemorySegmentVectors v = MemorySegmentVectors.open(file, 0, dim, arena);
        assertThat(v.size()).isEqualTo(0);
        assertThat(v.dimension()).isEqualTo(dim);
        assertThatExceptionOfType(IndexOutOfBoundsException.class)
            .isThrownBy(() -> v.vectorSlice(0));
      }
    }
  }

  @Nested
  class Validation {

    @Test
    void nullFileThrows(@TempDir Path tmp) {
      try (Arena arena = Arena.ofConfined()) {
        assertThatNullPointerException()
            .isThrownBy(() -> MemorySegmentVectors.open(null, 1, 64, arena))
            .withMessageContaining("file");
      }
    }

    @Test
    void nullArenaThrows(@TempDir Path tmp) {
      Path file = tmp.resolve("vectors.bin");
      assertThatNullPointerException()
          .isThrownBy(() -> MemorySegmentVectors.open(file, 1, 64, null))
          .withMessageContaining("arena");
    }

    @Test
    void negativeSizeRejected(@TempDir Path tmp) {
      Path file = tmp.resolve("vectors.bin");
      try (Arena arena = Arena.ofConfined()) {
        assertThatExceptionOfType(IllegalArgumentException.class)
            .isThrownBy(() -> MemorySegmentVectors.open(file, -1, 64, arena))
            .withMessageContaining("size");
      }
    }

    @Test
    void zeroDimensionRejected(@TempDir Path tmp) {
      Path file = tmp.resolve("vectors.bin");
      try (Arena arena = Arena.ofConfined()) {
        assertThatExceptionOfType(IllegalArgumentException.class)
            .isThrownBy(() -> MemorySegmentVectors.open(file, 1, 0, arena))
            .withMessageContaining("dimension");
      }
    }

    @Test
    void missingFileThrows(@TempDir Path tmp) {
      Path file = tmp.resolve("does-not-exist.bin");
      try (Arena arena = Arena.ofConfined()) {
        assertThatIOException().isThrownBy(() -> MemorySegmentVectors.open(file, 1, 64, arena));
      }
    }

    @Test
    void truncatedFileThrows(@TempDir Path tmp) throws IOException {
      // Write only 2 vectors, but claim the file has 10.
      int dim = 64;
      float[][] vectors = randomVectors(2, dim, 3L);
      Path file = writeVectorsBin(tmp, "vectors.bin", vectors, dim);

      try (Arena arena = Arena.ofConfined()) {
        assertThatIOException()
            .isThrownBy(() -> MemorySegmentVectors.open(file, 10, dim, arena))
            .withMessageContaining("truncated");
      }
    }
  }

  @Nested
  class ThreadSafety {

    @Test
    void concurrentSlicesReturnCorrectBytes(@TempDir Path tmp) throws Exception {
      int dim = 128;
      int n = 1000;
      float[][] expected = randomVectors(n, dim, 12345L);
      Path file = writeVectorsBin(tmp, "vectors.bin", expected, dim);

      // Shared arena used across multiple reader threads — this is exactly how the Step 4a
      // Generation model works in production.
      try (Arena arena = Arena.ofShared()) {
        MemorySegmentVectors v = MemorySegmentVectors.open(file, n, dim, arena);

        int threads = 4;
        int iterationsPerThread = 500;
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        CountDownLatch start = new CountDownLatch(1);
        CountDownLatch done = new CountDownLatch(threads);
        AtomicReference<Throwable> failure = new AtomicReference<>();

        for (int t = 0; t < threads; t++) {
          final int seed = t;
          pool.submit(
              () -> {
                try {
                  start.await();
                  Random rng = new Random(seed);
                  for (int iter = 0; iter < iterationsPerThread; iter++) {
                    int ord = rng.nextInt(n);
                    MemorySegment slice = v.vectorSlice(ord);
                    float[] got = sliceToFloats(slice, dim);
                    for (int d = 0; d < dim; d++) {
                      if (Float.floatToRawIntBits(got[d])
                          != Float.floatToRawIntBits(expected[ord][d])) {
                        throw new AssertionError(
                            "mismatch at ord="
                                + ord
                                + " dim="
                                + d
                                + " got="
                                + got[d]
                                + " want="
                                + expected[ord][d]);
                      }
                    }
                  }
                } catch (Throwable th) {
                  failure.compareAndSet(null, th);
                } finally {
                  done.countDown();
                }
              });
        }

        start.countDown();
        assertThat(done.await(10, TimeUnit.SECONDS)).isTrue();
        pool.shutdownNow();

        if (failure.get() != null) {
          throw new AssertionError("concurrent read failed", failure.get());
        }
      }
    }

    @Test
    void concurrentSameOrdinalReturnsIndependentViews(@TempDir Path tmp) throws Exception {
      // Two threads slice the SAME ordinal simultaneously. Each should get a view onto the same
      // underlying mmap bytes — no shared mutable buffer, no cross-thread interference.
      int dim = 64;
      int n = 10;
      float[][] expected = randomVectors(n, dim, 99L);
      Path file = writeVectorsBin(tmp, "vectors.bin", expected, dim);

      try (Arena arena = Arena.ofShared()) {
        MemorySegmentVectors v = MemorySegmentVectors.open(file, n, dim, arena);

        int threads = 8;
        int target = 5;
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        CountDownLatch start = new CountDownLatch(1);
        CountDownLatch done = new CountDownLatch(threads);
        AtomicReference<Throwable> failure = new AtomicReference<>();

        for (int t = 0; t < threads; t++) {
          pool.submit(
              () -> {
                try {
                  start.await();
                  for (int iter = 0; iter < 2000; iter++) {
                    MemorySegment slice = v.vectorSlice(target);
                    float[] got = sliceToFloats(slice, dim);
                    for (int d = 0; d < dim; d++) {
                      if (Float.floatToRawIntBits(got[d])
                          != Float.floatToRawIntBits(expected[target][d])) {
                        throw new AssertionError("shared-buffer bug: saw torn data");
                      }
                    }
                  }
                } catch (Throwable th) {
                  failure.compareAndSet(null, th);
                } finally {
                  done.countDown();
                }
              });
        }

        start.countDown();
        assertThat(done.await(10, TimeUnit.SECONDS)).isTrue();
        pool.shutdownNow();

        if (failure.get() != null) {
          throw new AssertionError("concurrent same-ordinal read failed", failure.get());
        }
      }
    }
  }

  @Nested
  class Bounds {

    @Test
    void negativeOrdinalThrows(@TempDir Path tmp) throws IOException {
      int dim = 32;
      float[][] vectors = randomVectors(5, dim, 1L);
      Path file = writeVectorsBin(tmp, "vectors.bin", vectors, dim);
      try (Arena arena = Arena.ofConfined()) {
        MemorySegmentVectors v = MemorySegmentVectors.open(file, 5, dim, arena);
        assertThatExceptionOfType(IndexOutOfBoundsException.class)
            .isThrownBy(() -> v.vectorSlice(-1));
        assertThatExceptionOfType(IndexOutOfBoundsException.class)
            .isThrownBy(() -> v.vectorOffsetFor(-1));
      }
    }

    @Test
    void ordinalAtSizeThrows(@TempDir Path tmp) throws IOException {
      int dim = 32;
      float[][] vectors = randomVectors(5, dim, 1L);
      Path file = writeVectorsBin(tmp, "vectors.bin", vectors, dim);
      try (Arena arena = Arena.ofConfined()) {
        MemorySegmentVectors v = MemorySegmentVectors.open(file, 5, dim, arena);
        assertThatExceptionOfType(IndexOutOfBoundsException.class)
            .isThrownBy(() -> v.vectorSlice(5));
        assertThatExceptionOfType(IndexOutOfBoundsException.class)
            .isThrownBy(() -> v.vectorOffsetFor(5));
      }
    }
  }

  @Nested
  class CloseSemantics {

    @Test
    void closeIsNoOpArenaOwnsLifetime(@TempDir Path tmp) throws IOException {
      // Closing the MemorySegmentVectors must NOT close the arena — subsequent slices
      // inside the same arena scope must still work.
      int dim = 64;
      float[][] vectors = randomVectors(10, dim, 1L);
      Path file = writeVectorsBin(tmp, "vectors.bin", vectors, dim);

      try (Arena arena = Arena.ofConfined()) {
        MemorySegmentVectors v = MemorySegmentVectors.open(file, 10, dim, arena);
        v.close(); // no-op

        // Still usable after close().
        MemorySegment slice = v.vectorSlice(3);
        assertThat(sliceToFloats(slice, dim)).containsExactly(vectors[3]);
      }
    }
  }
}
