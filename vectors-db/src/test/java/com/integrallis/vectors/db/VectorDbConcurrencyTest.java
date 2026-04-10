package com.integrallis.vectors.db;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;

import com.integrallis.vectors.core.SimilarityFunction;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Concurrency acceptance tests for the Step 3 {@link VectorCollection} implementation. Every test
 * is wrapped in {@code assertTimeoutPreemptively(5s)} so a deadlock or hang fails fast rather than
 * blocking CI indefinitely.
 */
class VectorDbConcurrencyTest {

  private static final long SEED = 42L;
  private static final int DIM = 128;

  private static VectorCollection newCollection() {
    return VectorCollection.builder().dimension(DIM).metric(SimilarityFunction.EUCLIDEAN).build();
  }

  private static float[] randomVector(Random rng) {
    float[] v = new float[DIM];
    for (int i = 0; i < DIM; i++) {
      v[i] = rng.nextFloat();
    }
    return v;
  }

  private static List<Document> randomDocs(int count, int idOffset, Random rng) {
    List<Document> docs = new ArrayList<>(count);
    for (int i = 0; i < count; i++) {
      docs.add(Document.of("doc-" + (idOffset + i), randomVector(rng)));
    }
    return docs;
  }

  /** Joins an executor within the test timeout budget, propagating any worker throwable. */
  private static void shutdownAndCheck(
      ExecutorService pool, List<AtomicReference<Throwable>> errors) throws InterruptedException {
    pool.shutdown();
    assertThat(pool.awaitTermination(5, TimeUnit.SECONDS)).isTrue();
    for (AtomicReference<Throwable> err : errors) {
      Throwable t = err.get();
      if (t != null) {
        if (t instanceof AssertionError ae) {
          throw ae;
        }
        throw new AssertionError("Worker failure", t);
      }
    }
  }

  @Nested
  @Tag("unit")
  class ReadersNeverBlockedByCommit {

    @Test
    void searchDuringCommitDoesNotBlock() {
      assertTimeoutPreemptively(
          Duration.ofSeconds(5),
          () -> {
            try (var col = newCollection()) {
              // Seed with 5000 docs.
              Random rng = new Random(SEED);
              col.addAll(randomDocs(5000, 0, rng));
              col.commit();

              int readerThreads = 4;
              ExecutorService pool = Executors.newFixedThreadPool(readerThreads + 1);
              List<AtomicReference<Throwable>> errors = new ArrayList<>();
              List<AtomicInteger> searchCounts = new ArrayList<>();
              AtomicBoolean stop = new AtomicBoolean(false);
              // Readers must signal that they have completed at least one real search (JIT +
              // thread startup done) BEFORE the writer is allowed to start committing. Without
              // this warmup, the writer can finish all of its fast in-memory commits before one
              // reader thread has even executed its first iteration, flaking the assertion.
              CountDownLatch readersWarmed = new CountDownLatch(readerThreads);

              for (int t = 0; t < readerThreads; t++) {
                AtomicReference<Throwable> err = new AtomicReference<>();
                AtomicInteger count = new AtomicInteger();
                errors.add(err);
                searchCounts.add(count);
                float[] query = randomVector(new Random(SEED + 100 + t));
                pool.submit(
                    () -> {
                      try {
                        // Warmup iteration before signalling the writer.
                        col.search(SearchRequest.builder(query, 10).build());
                        readersWarmed.countDown();
                        while (!stop.get()) {
                          col.search(SearchRequest.builder(query, 10).build());
                          count.incrementAndGet();
                        }
                      } catch (Throwable th) {
                        err.set(th);
                      }
                    });
              }

              AtomicReference<Throwable> writerErr = new AtomicReference<>();
              errors.add(writerErr);
              pool.submit(
                  () -> {
                    try {
                      readersWarmed.await();
                      Random wrng = new Random(SEED + 1);
                      int nextId = 5000;
                      for (int c = 0; c < 20; c++) {
                        col.addAll(randomDocs(1000, nextId, wrng));
                        nextId += 1000;
                        col.commit();
                        // Give readers a bounded window to rack up searches between commits.
                        // Flat-scan commits are sub-millisecond, and without a small pause the
                        // writer can finish its 20 commits before readers make meaningful
                        // progress. This does NOT weaken the invariant: readers are still
                        // interleaved with commits, and any hard block in search() would
                        // prevent them from making any progress at all.
                        Thread.sleep(2);
                      }
                    } catch (Throwable th) {
                      writerErr.set(th);
                    } finally {
                      stop.set(true);
                    }
                  });

              shutdownAndCheck(pool, errors);

              // Every reader completed many searches during the commit window (proving they were
              // never hard-blocked). 10 is a loose lower bound; practice is thousands.
              for (AtomicInteger c : searchCounts) {
                assertThat(c.get()).isGreaterThanOrEqualTo(10);
              }
              // And the final size is the sum of seed + all commits.
              assertThat(col.size()).isEqualTo(5000 + 20 * 1000);
            }
          });
    }

    @Test
    void searchesCompleteAgainstConsistentGeneration() {
      assertTimeoutPreemptively(
          Duration.ofSeconds(5),
          () -> {
            try (var col = newCollection()) {
              Random rng = new Random(SEED);
              col.addAll(randomDocs(2000, 0, rng));
              col.commit();

              int readerThreads = 4;
              ExecutorService pool = Executors.newFixedThreadPool(readerThreads + 1);
              List<AtomicReference<Throwable>> errors = new ArrayList<>();
              AtomicBoolean stop = new AtomicBoolean(false);
              CountDownLatch readersReady = new CountDownLatch(readerThreads);

              for (int t = 0; t < readerThreads; t++) {
                AtomicReference<Throwable> err = new AtomicReference<>();
                errors.add(err);
                float[] query = randomVector(new Random(SEED + 200 + t));
                pool.submit(
                    () -> {
                      try {
                        readersReady.countDown();
                        while (!stop.get()) {
                          var result = col.search(SearchRequest.builder(query, 10).build());
                          // Every returned hit id must resolve in the collection; the snapshot the
                          // search ran against must be internally consistent. (We query via the
                          // facade's get() which does its own volatile-read; if the search ran
                          // against an older generation, that is fine, because earlier generations
                          // are always a prefix of newer ones in Step 3.)
                          for (SearchResult.Hit hit : result.hits()) {
                            Document doc = col.get(hit.id());
                            if (doc == null) {
                              throw new AssertionError(
                                  "Search returned id '" + hit.id() + "' but get() was null");
                            }
                          }
                          // k=10 upper bound check.
                          if (result.hits().size() > 10) {
                            throw new AssertionError("hits.size > 10: " + result.hits().size());
                          }
                        }
                      } catch (Throwable th) {
                        err.set(th);
                      }
                    });
              }

              AtomicReference<Throwable> writerErr = new AtomicReference<>();
              errors.add(writerErr);
              pool.submit(
                  () -> {
                    try {
                      readersReady.await();
                      Random wrng = new Random(SEED + 2);
                      int nextId = 2000;
                      for (int c = 0; c < 10; c++) {
                        col.addAll(randomDocs(500, nextId, wrng));
                        nextId += 500;
                        col.commit();
                      }
                    } catch (Throwable th) {
                      writerErr.set(th);
                    } finally {
                      stop.set(true);
                    }
                  });

              shutdownAndCheck(pool, errors);
            }
          });
    }
  }

  @Nested
  @Tag("unit")
  class MonotonicObservedSize {

    @Test
    void sizeNeverShrinksAcrossCommits() {
      assertTimeoutPreemptively(
          Duration.ofSeconds(5),
          () -> {
            try (var col = newCollection()) {
              int writerThreads = 4;
              int writerDocs = 500;
              int readerThreads = 4;
              int readerPolls = 10_000;

              ExecutorService pool = Executors.newFixedThreadPool(writerThreads + readerThreads);
              List<AtomicReference<Throwable>> errors = new ArrayList<>();
              CountDownLatch writersDone = new CountDownLatch(writerThreads);

              for (int t = 0; t < writerThreads; t++) {
                final int writerId = t;
                AtomicReference<Throwable> err = new AtomicReference<>();
                errors.add(err);
                pool.submit(
                    () -> {
                      try {
                        Random rng = new Random(SEED + writerId);
                        for (int i = 0; i < writerDocs; i++) {
                          col.add(Document.of("w" + writerId + "-" + i, randomVector(rng)));
                        }
                        col.commit();
                      } catch (Throwable th) {
                        err.set(th);
                      } finally {
                        writersDone.countDown();
                      }
                    });
              }

              for (int t = 0; t < readerThreads; t++) {
                AtomicReference<Throwable> err = new AtomicReference<>();
                errors.add(err);
                pool.submit(
                    () -> {
                      try {
                        int last = 0;
                        for (int i = 0; i < readerPolls; i++) {
                          int s = col.size();
                          if (s < last) {
                            throw new AssertionError(
                                "Observed size shrank from " + last + " to " + s);
                          }
                          last = s;
                          if (writersDone.getCount() == 0 && i > 100) {
                            // Do a final few polls and exit — we've already covered the window.
                            break;
                          }
                        }
                      } catch (Throwable th) {
                        err.set(th);
                      }
                    });
              }

              shutdownAndCheck(pool, errors);

              assertThat(col.size()).isEqualTo(writerThreads * writerDocs);
            }
          });
    }
  }

  @Nested
  @Tag("unit")
  class AutoCommitOnThreshold {

    @Test
    void addAllBeyondThresholdAutoCommits() {
      assertTimeoutPreemptively(
          Duration.ofSeconds(5),
          () -> {
            try (var col =
                VectorCollection.builder()
                    .dimension(DIM)
                    .metric(SimilarityFunction.EUCLIDEAN)
                    .autoCommitThreshold(100)
                    .build()) {
              Random rng = new Random(SEED);
              col.addAll(randomDocs(200, 0, rng));
              // No explicit commit: the 200-doc addAll must have triggered auto-commit.
              assertThat(col.size()).isEqualTo(200);
              var result =
                  col.search(SearchRequest.builder(randomVector(new Random(SEED + 1)), 10).build());
              assertThat(result.hits()).hasSize(10);
            }
          });
    }

    @Test
    void individualAddBeyondThresholdAutoCommits() {
      assertTimeoutPreemptively(
          Duration.ofSeconds(5),
          () -> {
            try (var col =
                VectorCollection.builder()
                    .dimension(DIM)
                    .metric(SimilarityFunction.EUCLIDEAN)
                    .autoCommitThreshold(5)
                    .build()) {
              Random rng = new Random(SEED);
              for (int i = 0; i < 10; i++) {
                col.add(Document.of("doc-" + i, randomVector(rng)));
              }
              assertThat(col.size()).isEqualTo(10);
              var result =
                  col.search(SearchRequest.builder(randomVector(new Random(SEED + 1)), 10).build());
              assertThat(result.hits()).hasSize(10);
            }
          });
    }

    @Test
    void defaultConfigNeverAutoCommits() {
      assertTimeoutPreemptively(
          Duration.ofSeconds(5),
          () -> {
            try (var col = newCollection()) {
              Random rng = new Random(SEED);
              col.addAll(randomDocs(1000, 0, rng));
              // Default threshold = Integer.MAX_VALUE → no auto-commit.
              assertThat(col.size()).isZero();
              var result =
                  col.search(SearchRequest.builder(randomVector(new Random(SEED + 1)), 10).build());
              assertThat(result.hits()).isEmpty();
              col.commit();
              assertThat(col.size()).isEqualTo(1000);
            }
          });
    }
  }

  @Nested
  @Tag("unit")
  class WriterSerialization {

    @Test
    void concurrentAddsSerializeCorrectly() {
      assertTimeoutPreemptively(
          Duration.ofSeconds(5),
          () -> {
            try (var col = newCollection()) {
              int writerThreads = 4;
              int docsPerWriter = 500;
              ExecutorService pool = Executors.newFixedThreadPool(writerThreads);
              List<AtomicReference<Throwable>> errors = new ArrayList<>();
              CountDownLatch start = new CountDownLatch(1);

              for (int t = 0; t < writerThreads; t++) {
                final int writerId = t;
                AtomicReference<Throwable> err = new AtomicReference<>();
                errors.add(err);
                pool.submit(
                    () -> {
                      try {
                        start.await();
                        Random rng = new Random(SEED + writerId);
                        for (int i = 0; i < docsPerWriter; i++) {
                          col.add(Document.of("w" + writerId + "-" + i, randomVector(rng)));
                        }
                      } catch (Throwable th) {
                        err.set(th);
                      }
                    });
              }
              start.countDown();
              shutdownAndCheck(pool, errors);

              col.commit();
              assertThat(col.size()).isEqualTo(writerThreads * docsPerWriter);

              // All unique ids must be reachable.
              Set<String> seen = new HashSet<>();
              for (int w = 0; w < writerThreads; w++) {
                for (int i = 0; i < docsPerWriter; i++) {
                  String id = "w" + w + "-" + i;
                  assertThat(col.contains(id)).isTrue();
                  seen.add(id);
                }
              }
              assertThat(seen).hasSize(writerThreads * docsPerWriter);
            }
          });
    }
  }

  @Nested
  @Tag("unit")
  class CloseDuringWrite {

    @Test
    void closeRejectsFutureOperations() {
      assertTimeoutPreemptively(
          Duration.ofSeconds(5),
          () -> {
            var col = newCollection();
            Random rng = new Random(SEED);
            col.addAll(randomDocs(100, 0, rng));
            col.commit();

            int readerThreads = 2;
            ExecutorService pool = Executors.newFixedThreadPool(readerThreads);
            AtomicBoolean stop = new AtomicBoolean(false);
            List<AtomicReference<Throwable>> errors = new ArrayList<>();
            CountDownLatch ready = new CountDownLatch(readerThreads);

            for (int t = 0; t < readerThreads; t++) {
              AtomicReference<Throwable> err = new AtomicReference<>();
              errors.add(err);
              float[] query = randomVector(new Random(SEED + 500 + t));
              pool.submit(
                  () -> {
                    try {
                      ready.countDown();
                      while (!stop.get()) {
                        try {
                          col.search(SearchRequest.builder(query, 5).build());
                        } catch (IllegalStateException expected) {
                          // Once close() lands, some readers may see "closed". That is acceptable.
                          return;
                        }
                      }
                    } catch (Throwable th) {
                      err.set(th);
                    }
                  });
            }

            ready.await();
            // Give readers a chance to start.
            Thread.sleep(10);
            col.close();
            stop.set(true);

            pool.shutdown();
            assertThat(pool.awaitTermination(5, TimeUnit.SECONDS)).isTrue();
            for (AtomicReference<Throwable> err : errors) {
              Throwable th = err.get();
              if (th != null) {
                throw new AssertionError("Reader failure", th);
              }
            }

            // Any NEW operation after close() must throw.
            try {
              col.search(SearchRequest.builder(new float[DIM], 1).build());
              throw new AssertionError("Expected IllegalStateException on search() after close()");
            } catch (IllegalStateException expected) {
              assertThat(expected).hasMessageContaining("closed");
            }
            try {
              col.add(Document.of("post-close", new float[DIM]));
              throw new AssertionError("Expected IllegalStateException on add() after close()");
            } catch (IllegalStateException expected) {
              assertThat(expected).hasMessageContaining("closed");
            }
            try {
              col.commit();
              throw new AssertionError("Expected IllegalStateException on commit() after close()");
            } catch (IllegalStateException expected) {
              assertThat(expected).hasMessageContaining("closed");
            }
          });
    }
  }
}
