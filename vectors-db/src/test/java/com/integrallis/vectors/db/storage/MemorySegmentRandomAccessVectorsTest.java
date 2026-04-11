package com.integrallis.vectors.db.storage;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatExceptionOfType;
import static org.assertj.core.api.Assertions.assertThatNullPointerException;

import com.integrallis.vectors.core.VectorEncoding;
import com.integrallis.vectors.storage.store.VectorStoreWriter;
import java.io.IOException;
import java.lang.foreign.Arena;
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

/**
 * Unit tests for {@link MemorySegmentRandomAccessVectors}. Covers the shared per-thread scratch
 * contract that the persistent HNSW read path depends on: every {@code getVector} call on a single
 * thread overwrites the previous call's buffer, but threads never alias each other.
 */
@Tag("unit")
class MemorySegmentRandomAccessVectorsTest {

  private static Path writeVectorsBin(Path tmp, float[][] vectors, int dim) throws IOException {
    Path file = tmp.resolve("vectors.bin");
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

  @Nested
  class Basics {

    @Test
    void getVectorReturnsExpectedValues(@TempDir Path tmp) throws IOException {
      int dim = 16;
      int n = 8;
      float[][] expected = randomVectors(n, dim, 1L);
      Path file = writeVectorsBin(tmp, expected, dim);

      try (Arena arena = Arena.ofConfined()) {
        MemorySegmentVectors mapped = MemorySegmentVectors.open(file, n, dim, arena);
        MemorySegmentRandomAccessVectors rav = new MemorySegmentRandomAccessVectors(mapped);

        assertThat(rav.size()).isEqualTo(n);
        assertThat(rav.dimension()).isEqualTo(dim);

        // Copy out each call's result BEFORE the next getVector call so the scratch contract
        // isn't violated.
        for (int i = 0; i < n; i++) {
          float[] scratch = rav.getVector(i);
          float[] snapshot = scratch.clone();
          assertThat(snapshot).as("vector %d", i).containsExactly(expected[i]);
        }
      }
    }

    @Test
    void sizeAndDimensionDelegateToMapped(@TempDir Path tmp) throws IOException {
      int dim = 64;
      int n = 3;
      Path file = writeVectorsBin(tmp, randomVectors(n, dim, 2L), dim);
      try (Arena arena = Arena.ofConfined()) {
        MemorySegmentVectors mapped = MemorySegmentVectors.open(file, n, dim, arena);
        MemorySegmentRandomAccessVectors rav = new MemorySegmentRandomAccessVectors(mapped);
        assertThat(rav.size()).isEqualTo(mapped.size());
        assertThat(rav.dimension()).isEqualTo(mapped.dimension());
      }
    }
  }

  @Nested
  class SharedBufferContract {

    @Test
    void consecutiveCallsReturnSameArrayReference(@TempDir Path tmp) throws IOException {
      // The per-thread scratch pattern requires that two calls on the same thread return the
      // SAME float[] reference (zero allocation). This is the whole point of the scratch buffer.
      int dim = 32;
      int n = 10;
      Path file = writeVectorsBin(tmp, randomVectors(n, dim, 3L), dim);
      try (Arena arena = Arena.ofConfined()) {
        MemorySegmentVectors mapped = MemorySegmentVectors.open(file, n, dim, arena);
        MemorySegmentRandomAccessVectors rav = new MemorySegmentRandomAccessVectors(mapped);

        float[] first = rav.getVector(0);
        float[] second = rav.getVector(1);
        // Same reference — the second call OVERWROTE the first call's data in place.
        assertThat(second).isSameAs(first);
      }
    }

    @Test
    void secondCallOverwritesFirstCallContents(@TempDir Path tmp) throws IOException {
      // Document the "must not retain across calls" contract by proving that holding a reference
      // across a subsequent call produces silently wrong data.
      int dim = 16;
      int n = 4;
      float[][] expected = randomVectors(n, dim, 4L);
      Path file = writeVectorsBin(tmp, expected, dim);
      try (Arena arena = Arena.ofConfined()) {
        MemorySegmentVectors mapped = MemorySegmentVectors.open(file, n, dim, arena);
        MemorySegmentRandomAccessVectors rav = new MemorySegmentRandomAccessVectors(mapped);

        float[] heldFromFirstCall = rav.getVector(0);
        // Snapshot expected[0] BEFORE the overwrite so we have something to compare against.
        float[] expected0 = expected[0].clone();

        // Now the caller violates the scratch contract: they call getVector again while still
        // holding heldFromFirstCall. The held reference now contains expected[2]'s data.
        rav.getVector(2);
        assertThat(heldFromFirstCall).containsExactly(expected[2]);
        assertThat(heldFromFirstCall).doesNotContain(expected0[0], expected0[1]);
      }
    }
  }

  @Nested
  class ThreadIsolation {

    @Test
    void separateThreadsHaveIndependentScratch(@TempDir Path tmp) throws Exception {
      // The scratch is a ThreadLocal<float[]>, so two threads racing on the same ordinal MUST
      // each get correct, uncontaminated data regardless of how they interleave. A shared-buffer
      // bug would produce torn reads (one thread's copy partially overwritten by the other's).
      int dim = 64;
      int n = 100;
      float[][] expected = randomVectors(n, dim, 5L);
      Path file = writeVectorsBin(tmp, expected, dim);

      try (Arena arena = Arena.ofShared()) {
        MemorySegmentVectors mapped = MemorySegmentVectors.open(file, n, dim, arena);
        MemorySegmentRandomAccessVectors rav = new MemorySegmentRandomAccessVectors(mapped);

        int threads = 4;
        int iterationsPerThread = 1000;
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
                    float[] got = rav.getVector(ord);
                    // Verify the full contents before another getVector call can possibly
                    // interfere. Since each thread has its own scratch, this must ALWAYS
                    // succeed regardless of concurrent activity on other threads.
                    for (int d = 0; d < dim; d++) {
                      if (Float.floatToRawIntBits(got[d])
                          != Float.floatToRawIntBits(expected[ord][d])) {
                        throw new AssertionError(
                            "cross-thread contamination at ord="
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
          throw new AssertionError("concurrent getVector failed", failure.get());
        }
      }
    }

    @Test
    void twoThreadsHoldDistinctScratchReferences(@TempDir Path tmp) throws Exception {
      int dim = 32;
      int n = 5;
      Path file = writeVectorsBin(tmp, randomVectors(n, dim, 6L), dim);

      try (Arena arena = Arena.ofShared()) {
        MemorySegmentVectors mapped = MemorySegmentVectors.open(file, n, dim, arena);
        MemorySegmentRandomAccessVectors rav = new MemorySegmentRandomAccessVectors(mapped);

        AtomicReference<float[]> ref1 = new AtomicReference<>();
        AtomicReference<float[]> ref2 = new AtomicReference<>();
        CountDownLatch barrier = new CountDownLatch(2);
        CountDownLatch done = new CountDownLatch(2);

        Thread t1 =
            new Thread(
                () -> {
                  try {
                    barrier.countDown();
                    barrier.await();
                    ref1.set(rav.getVector(0));
                  } catch (InterruptedException ignored) {
                    Thread.currentThread().interrupt();
                  } finally {
                    done.countDown();
                  }
                });
        Thread t2 =
            new Thread(
                () -> {
                  try {
                    barrier.countDown();
                    barrier.await();
                    ref2.set(rav.getVector(0));
                  } catch (InterruptedException ignored) {
                    Thread.currentThread().interrupt();
                  } finally {
                    done.countDown();
                  }
                });
        t1.start();
        t2.start();
        assertThat(done.await(5, TimeUnit.SECONDS)).isTrue();

        // Distinct thread-local scratch buffers — different identity even for the same ordinal.
        assertThat(ref1.get()).isNotSameAs(ref2.get());
        // Same contents, though (they read the same ordinal).
        assertThat(ref1.get()).containsExactly(ref2.get());
      }
    }
  }

  @Nested
  class NullGuards {

    @Test
    void nullMappedThrows() {
      assertThatNullPointerException()
          .isThrownBy(() -> new MemorySegmentRandomAccessVectors(null))
          .withMessageContaining("mapped");
    }
  }

  @Nested
  class Bounds {

    @Test
    void negativeOrdinalThrows(@TempDir Path tmp) throws IOException {
      int dim = 16;
      int n = 4;
      Path file = writeVectorsBin(tmp, randomVectors(n, dim, 7L), dim);
      try (Arena arena = Arena.ofConfined()) {
        MemorySegmentVectors mapped = MemorySegmentVectors.open(file, n, dim, arena);
        MemorySegmentRandomAccessVectors rav = new MemorySegmentRandomAccessVectors(mapped);
        assertThatExceptionOfType(IndexOutOfBoundsException.class)
            .isThrownBy(() -> rav.getVector(-1));
      }
    }

    @Test
    void ordinalAtSizeThrows(@TempDir Path tmp) throws IOException {
      int dim = 16;
      int n = 4;
      Path file = writeVectorsBin(tmp, randomVectors(n, dim, 8L), dim);
      try (Arena arena = Arena.ofConfined()) {
        MemorySegmentVectors mapped = MemorySegmentVectors.open(file, n, dim, arena);
        MemorySegmentRandomAccessVectors rav = new MemorySegmentRandomAccessVectors(mapped);
        assertThatExceptionOfType(IndexOutOfBoundsException.class)
            .isThrownBy(() -> rav.getVector(n));
      }
    }
  }
}
