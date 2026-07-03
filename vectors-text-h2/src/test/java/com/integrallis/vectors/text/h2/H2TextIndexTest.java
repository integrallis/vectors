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
package com.integrallis.vectors.text.h2;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.vectors.hybrid.text.TextIndexSpi;
import com.integrallis.vectors.hybrid.text.TextIndexSpi.StoredContent;
import com.integrallis.vectors.hybrid.text.TextIndexSpi.TextDocument;
import com.integrallis.vectors.hybrid.text.TextIndexSpiFactory;
import com.integrallis.vectors.hybrid.text.TextSearchOutcome;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.ServiceLoader;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class H2TextIndexTest {

  private H2TextIndex index;

  @BeforeEach
  void setUp() {
    index = new H2TextIndex("test_" + System.nanoTime());
  }

  @AfterEach
  void tearDown() {
    if (index != null) {
      index.close();
    }
  }

  @Nested
  @Tag("unit")
  class IndexAndSearch {

    @Test
    void indexAndSearchRoundTrip() {
      index.index(
          List.of(
              new TextDocument(
                  "doc-1", "HNSW is a graph-based vector search algorithm", Map.of(), null),
              new TextDocument("doc-2", "Product quantization compresses vectors", Map.of(), null),
              new TextDocument("doc-3", "Scalar quantization uses int8 encoding", Map.of(), null)));

      assertThat(index.size()).isEqualTo(3);

      TextSearchOutcome outcome = index.search("quantization", 10);
      assertThat(outcome.size()).isGreaterThanOrEqualTo(1);
      // At least one of the quantization docs should match
      boolean hasQuantizationDoc = false;
      for (String id : outcome.ids()) {
        if ("doc-2".equals(id) || "doc-3".equals(id)) {
          hasQuantizationDoc = true;
          break;
        }
      }
      assertThat(hasQuantizationDoc).isTrue();
    }

    @Test
    void searchEmptyIndex() {
      TextSearchOutcome outcome = index.search("anything", 5);
      assertThat(outcome.size()).isZero();
    }

    @Test
    void searchBlankQuery() {
      index.index(List.of(new TextDocument("doc-1", "some text", Map.of(), null)));
      TextSearchOutcome outcome = index.search("", 5);
      assertThat(outcome.size()).isZero();
    }

    @Test
    void upsertOverwritesExisting() {
      index.index(List.of(new TextDocument("doc-1", "original text", Map.of(), null)));
      index.index(List.of(new TextDocument("doc-1", "updated text", Map.of(), null)));

      assertThat(index.size()).isEqualTo(1);
      Optional<StoredContent> content = index.get("doc-1");
      assertThat(content).isPresent();
      assertThat(content.get().text()).isEqualTo("updated text");
    }
  }

  @Nested
  @Tag("unit")
  class BlobStorage {

    @Test
    void storageAndRetrieval() {
      byte[] imageData = "fake-image-data".getBytes(StandardCharsets.UTF_8);
      index.index(
          List.of(
              new TextDocument("img-1", "A picture of a cat", Map.of("type", "image"), imageData)));

      Optional<byte[]> blob = index.getBlob("img-1");
      assertThat(blob).isPresent();
      assertThat(blob.get()).isEqualTo(imageData);
    }

    @Test
    void getBlobMissing() {
      assertThat(index.getBlob("nonexistent")).isEmpty();
    }

    @Test
    void getBlobNullBlob() {
      index.index(List.of(new TextDocument("no-blob", "text only", Map.of(), null)));
      Optional<byte[]> blob = index.getBlob("no-blob");
      // Should return empty or present-with-null depending on H2 behavior
      assertThat(blob.isPresent() && blob.get() != null || blob.isEmpty()).isTrue();
    }
  }

  @Nested
  @Tag("unit")
  class MetadataAndLifecycle {

    @Test
    void metadataRoundTrip() {
      Map<String, String> meta = Map.of("source", "page-3", "type", "text");
      index.index(List.of(new TextDocument("meta-1", "content", meta, null)));

      Optional<StoredContent> content = index.get("meta-1");
      assertThat(content).isPresent();
      assertThat(content.get().metadata()).containsEntry("source", "page-3");
      assertThat(content.get().metadata()).containsEntry("type", "text");
    }

    @Test
    void metadataRoundTripPreservesDelimitersAndJsonCharacters() {
      Map<String, String> meta =
          Map.of(
              "source|kind", "page=3|section=2",
              "json", "{\"quoted\":true,\"list\":[1,2]}");
      index.index(List.of(new TextDocument("meta-special", "content", meta, null)));

      Optional<StoredContent> content = index.get("meta-special");

      assertThat(content).isPresent();
      assertThat(content.get().metadata()).isEqualTo(meta);
    }

    @Test
    void indexBatchRollsBackWhenLaterDocumentFailsEncoding() {
      Map<String, String> invalidMetadata = new HashMap<>();
      invalidMetadata.put(null, "null keys cannot be represented as JSON object field names");
      List<TextDocument> batch =
          List.of(
              new TextDocument("valid-before-invalid", "content", Map.of(), null),
              new TextDocument("invalid-metadata", "content", invalidMetadata, null));

      assertThatThrownBy(() -> index.index(batch))
          .isInstanceOf(IllegalStateException.class)
          .hasMessageContaining("failed to index documents");
      assertThat(index.get("valid-before-invalid")).isEmpty();
      assertThat(index.size()).isZero();
    }

    @Test
    void indexBatchRejectsInvalidDocumentBeforeWritingEarlierDocuments() {
      List<TextDocument> batch = new ArrayList<>();
      batch.add(new TextDocument("valid-before-null", "content", Map.of(), null));
      batch.add(null);

      assertThatThrownBy(() -> index.index(batch))
          .isInstanceOf(NullPointerException.class)
          .hasMessageContaining("documents must not contain null entries");
      assertThat(index.get("valid-before-null")).isEmpty();
      assertThat(index.size()).isZero();
    }

    @Test
    void removeDocument() {
      index.index(List.of(new TextDocument("del-1", "to be deleted", Map.of(), null)));
      assertThat(index.size()).isEqualTo(1);

      index.remove("del-1");
      assertThat(index.size()).isZero();
      assertThat(index.get("del-1")).isEmpty();
    }

    @Test
    void clearAll() {
      index.index(
          List.of(
              new TextDocument("c-1", "text 1", Map.of(), null),
              new TextDocument("c-2", "text 2", Map.of(), null)));
      assertThat(index.size()).isEqualTo(2);

      index.clear();
      assertThat(index.size()).isZero();
    }
  }

  @Nested
  @Tag("unit")
  class ServiceLoaderDiscovery {

    @Test
    void factoryIsDiscoverable() {
      ServiceLoader<TextIndexSpiFactory> loader = ServiceLoader.load(TextIndexSpiFactory.class);
      boolean found = false;
      for (TextIndexSpiFactory factory : loader) {
        if (factory instanceof H2TextIndexFactory) {
          TextIndexSpi created = factory.create("serviceloader_test_" + System.nanoTime());
          assertThat(created).isInstanceOf(H2TextIndex.class);
          created.close();
          found = true;
        }
      }
      assertThat(found).as("H2TextIndexFactory should be discoverable via ServiceLoader").isTrue();
    }
  }

  /**
   * The H2 lock file ({@code <db>.lock.db}) is exactly how H2 prevents two processes from opening
   * the same database. Deleting it on startup defeats that — a second server (or a crashed-but-
   * still-running first one) gets silently stomped, leading to data corruption. The constructor
   * must therefore not touch H2's housekeeping files. We use the {@code .trace.db} file as a
   * deterministic witness because H2 never creates, modifies, or deletes it on its own (tracing is
   * off by default), so its survival is solely a function of our own cleanup behaviour.
   */
  @Nested
  @Tag("unit")
  class HousekeepingFiles {

    @Test
    void constructorDoesNotDeleteTraceFile(@TempDir Path tempDir) throws Exception {
      Path traceFile = tempDir.resolve("text-index.trace.db");
      Files.writeString(traceFile, "diagnostic-from-prior-run");

      try (var ix = new H2TextIndex("preserve-trace_" + System.nanoTime(), tempDir)) {
        // Open the DB; the act of constructing must not touch unrelated files.
        ix.index(java.util.List.of(new TextDocument("d", "x", java.util.Map.of(), null)));
      }

      assertThat(traceFile)
          .as("H2TextIndex constructor must not delete pre-existing .trace.db")
          .exists();
      assertThat(Files.readString(traceFile))
          .as(".trace.db content must be preserved (cleanup is a bug)")
          .isEqualTo("diagnostic-from-prior-run");
    }

    /**
     * Pins that the "FT_CREATE_INDEX already exists" detection survives an H2 message change. The
     * previous check matched on {@code e.getMessage().contains("already")}, which silently
     * swallowed any other creation error whose message happened to include the word and would break
     * on a future H2 release that reworded the failure. The hardened check uses H2's stable
     * error-code constants.
     */
    @Test
    void reopeningPersistentIndexDoesNotThrow(@TempDir Path tempDir) throws Exception {
      String collection = "reopen_" + System.nanoTime();
      try (H2TextIndex first = new H2TextIndex(collection, tempDir)) {
        first.index(java.util.List.of(new TextDocument("d1", "hello world", Map.of(), null)));
      }
      // Re-opening the same on-disk DB triggers FT_CREATE_INDEX a second time. The duplicate
      // create must be recognized by error-code (not message text) and silently ignored.
      try (H2TextIndex second = new H2TextIndex(collection, tempDir)) {
        second.index(java.util.List.of(new TextDocument("d2", "another doc", Map.of(), null)));
        assertThat(second.size()).isEqualTo(2);
      }
    }
  }

  /**
   * The server invokes {@code index()}, {@code search()}, {@code get()}, {@code remove()} from
   * Helidon virtual-thread handlers, so an H2 index that is not thread-safe corrupts results /
   * loses writes under concurrent load. These tests pin that contract.
   */
  @Nested
  @Tag("unit")
  class Concurrency {

    @Test
    void concurrentIndexAcrossThreads_allWritesVisible() throws Exception {
      int writers = 16;
      int docsPerWriter = 25;
      int total = writers * docsPerWriter;

      try (var pool = java.util.concurrent.Executors.newFixedThreadPool(writers)) {
        var ready = new java.util.concurrent.CountDownLatch(writers);
        var go = new java.util.concurrent.CountDownLatch(1);
        var done = new java.util.concurrent.CountDownLatch(writers);
        var errors = new java.util.concurrent.ConcurrentLinkedQueue<Throwable>();

        for (int w = 0; w < writers; w++) {
          final int wid = w;
          pool.submit(
              () -> {
                ready.countDown();
                try {
                  go.await();
                  for (int i = 0; i < docsPerWriter; i++) {
                    String id = "w" + wid + "-d" + i;
                    index.index(
                        List.of(
                            new TextDocument(
                                id, "payload for " + id, Map.of("w", "" + wid), null)));
                  }
                } catch (Throwable t) {
                  errors.add(t);
                } finally {
                  done.countDown();
                }
              });
        }

        ready.await();
        go.countDown();
        done.await();

        assertThat(errors).as("no concurrent-write exceptions").isEmpty();
        assertThat(index.size())
            .as("every concurrent index() write must be visible afterwards")
            .isEqualTo(total);
        // Spot-check a sample survives — protects against silent overwrites.
        for (int wid = 0; wid < writers; wid++) {
          String id = "w" + wid + "-d0";
          Optional<StoredContent> stored = index.get(id);
          assertThat(stored)
              .as("doc %s should be retrievable after concurrent writes", id)
              .isPresent();
        }
      }
    }

    @Test
    void concurrentReadersAndWriters_noExceptionsAndReadOwnWrites() throws Exception {
      int writers = 8;
      int readers = 8;
      int opsPerThread = 30;

      try (var pool = java.util.concurrent.Executors.newFixedThreadPool(writers + readers)) {
        var ready = new java.util.concurrent.CountDownLatch(writers + readers);
        var go = new java.util.concurrent.CountDownLatch(1);
        var done = new java.util.concurrent.CountDownLatch(writers + readers);
        var errors = new java.util.concurrent.ConcurrentLinkedQueue<Throwable>();

        for (int w = 0; w < writers; w++) {
          final int wid = w;
          pool.submit(
              () -> {
                ready.countDown();
                try {
                  go.await();
                  for (int i = 0; i < opsPerThread; i++) {
                    String id = "rw-w" + wid + "-d" + i;
                    index.index(List.of(new TextDocument(id, "text " + id, Map.of(), null)));
                    // Read-your-writes: a thread-safe index must see this writer's own write.
                    assertThat(index.get(id))
                        .as("writer should observe its own just-completed write %s", id)
                        .isPresent();
                  }
                } catch (Throwable t) {
                  errors.add(t);
                } finally {
                  done.countDown();
                }
              });
        }
        for (int r = 0; r < readers; r++) {
          pool.submit(
              () -> {
                ready.countDown();
                try {
                  go.await();
                  for (int i = 0; i < opsPerThread; i++) {
                    // Run search and size concurrently with writers; only assert no throw here.
                    index.search("text", 5);
                    index.size();
                  }
                } catch (Throwable t) {
                  errors.add(t);
                } finally {
                  done.countDown();
                }
              });
        }

        ready.await();
        go.countDown();
        done.await();

        assertThat(errors).as("no concurrent read/write exceptions").isEmpty();
        assertThat(index.size()).isEqualTo(writers * opsPerThread);
      }
    }
  }
}
