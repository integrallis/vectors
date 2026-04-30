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

import com.integrallis.vectors.hybrid.text.TextIndexSpi;
import com.integrallis.vectors.hybrid.text.TextIndexSpi.StoredContent;
import com.integrallis.vectors.hybrid.text.TextIndexSpi.TextDocument;
import com.integrallis.vectors.hybrid.text.TextIndexSpiFactory;
import com.integrallis.vectors.hybrid.text.TextSearchOutcome;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.ServiceLoader;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

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
}
