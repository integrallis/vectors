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
package com.integrallis.vectors.db;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.ContentHash;
import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.core.EmbeddingRecipe;
import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.storage.FileFormat;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Stream;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

/**
 * #216: a content hash the caller supplied is the hash the collection gives back, on every read
 * path, in memory and across a reopen. A hash that is merely the SHA-256 of the stored text still
 * costs nothing on disk: such a collection keeps writing the version 1 {@code metadata.bin}.
 */
@Tag("unit")
class ContentHashRoundTripTest {

  /** What a recipe with the prefix {@code "search_document: "} hashes for the text "alpha". */
  private static final String PREFIXED = ContentHash.of("search_document: alpha");

  private static final String BARE = ContentHash.of("alpha");
  private static final String ELSEWHERE = ContentHash.of("bytes held in a system of record");

  enum Storage {
    IN_MEMORY,
    ON_DISK,
    ON_DISK_REOPENED
  }

  private static VectorCollectionBuilder builder(Storage storage, Path root) {
    VectorCollectionBuilder builder =
        VectorCollection.builder()
            .dimension(4)
            .metric(SimilarityFunction.COSINE)
            .indexType(IndexType.FLAT)
            .autoCommitThreshold(Integer.MAX_VALUE);
    return storage == Storage.IN_MEMORY ? builder : builder.storagePath(root);
  }

  /** Runs {@code check} against the collection as written, and again after a reopen if asked. */
  private static void writeThenRead(
      Storage storage,
      Path root,
      List<Document> docs,
      java.util.function.Consumer<VectorCollection> check)
      throws Exception {
    VectorCollection collection = builder(storage, root).build();
    try {
      collection.addAll(docs);
      collection.commit();
      if (storage == Storage.ON_DISK_REOPENED) {
        collection.close();
        collection = builder(storage, root).build();
      }
      check.accept(collection);
    } finally {
      collection.close();
    }
  }

  private static Document textWithSuppliedHash() {
    return new Document("a", new float[] {1, 0, 0, 0}, "alpha", Map.of(), PREFIXED);
  }

  private static Document noTextWithHash() {
    return Document.withHash("b", new float[] {0, 1, 0, 0}, ELSEWHERE, Map.of());
  }

  private static Document textOnly() {
    return new Document("c", new float[] {0, 0, 1, 0}, "alpha", Map.of());
  }

  @ParameterizedTest
  @EnumSource(Storage.class)
  void aSuppliedHashForATextDocumentComesBackOnEveryReadPath(Storage storage, @TempDir Path root)
      throws Exception {
    writeThenRead(
        storage,
        root,
        List.of(textWithSuppliedHash()),
        collection -> {
          assertThat(collection.get("a").contentHash()).isEqualTo(PREFIXED);
          assertThat(collection.documents())
              .extracting(Document::contentHash)
              .containsExactly(PREFIXED);
          SearchResult withEverything =
              collection.search(SearchRequest.builder(new float[] {1, 0, 0, 0}, 1).build());
          assertThat(withEverything.hits().get(0).document().contentHash()).isEqualTo(PREFIXED);
          SearchResult withoutText =
              collection.search(
                  SearchRequest.builder(new float[] {1, 0, 0, 0}, 1).includeText(false).build());
          assertThat(withoutText.hits().get(0).document().text()).isNull();
          assertThat(withoutText.hits().get(0).document().contentHash()).isEqualTo(PREFIXED);
          assertThat(collection.get("a").text()).isEqualTo("alpha");
        });
  }

  @ParameterizedTest
  @EnumSource(Storage.class)
  void aDocumentStoredWithoutItsSourceKeepsItsHash(Storage storage, @TempDir Path root)
      throws Exception {
    writeThenRead(
        storage,
        root,
        List.of(noTextWithHash()),
        collection -> {
          Document back = collection.get("b");
          assertThat(back.text()).isNull();
          assertThat(back.hasProvenance()).isTrue();
          assertThat(back.contentHash()).isEqualTo(ELSEWHERE);
        });
  }

  @ParameterizedTest
  @EnumSource(Storage.class)
  void aHashDerivedFromTheTextIsStillDerived(Storage storage, @TempDir Path root) throws Exception {
    writeThenRead(
        storage,
        root,
        List.of(textOnly()),
        collection -> assertThat(collection.get("c").contentHash()).isEqualTo(BARE));
  }

  @ParameterizedTest
  @EnumSource(Storage.class)
  void suppliedDerivedAndSourcelessDocumentsCoexist(Storage storage, @TempDir Path root)
      throws Exception {
    writeThenRead(
        storage,
        root,
        List.of(textOnly(), textWithSuppliedHash(), noTextWithHash()),
        collection -> {
          assertThat(collection.get("a").contentHash()).isEqualTo(PREFIXED);
          assertThat(collection.get("b").contentHash()).isEqualTo(ELSEWHERE);
          assertThat(collection.get("c").contentHash()).isEqualTo(BARE);
        });
  }

  @Test
  void aCollectionWithOnlyDerivedHashesKeepsWritingTheVersionOneFile(@TempDir Path root)
      throws Exception {
    try (VectorCollection collection = builder(Storage.ON_DISK, root).build()) {
      collection.add(textOnly());
      collection.add(new Document("d", new float[] {0, 0, 0, 1}, null, Map.of()));
      collection.commit();
    }

    assertThat(metadataVersion(root)).isEqualTo(FileFormat.VERSION_METADATA).isEqualTo(1);
  }

  @Test
  void theFileOnlyMovesToVersionTwoOnceAHashThatCannotBeDerivedArrives(@TempDir Path root)
      throws Exception {
    try (VectorCollection collection = builder(Storage.ON_DISK, root).build()) {
      collection.add(textOnly());
      collection.commit();
      assertThat(metadataVersion(root)).isEqualTo(1);

      collection.add(textWithSuppliedHash());
      collection.commit();
      assertThat(metadataVersion(root)).isEqualTo(2);
    }
    try (VectorCollection reopened = builder(Storage.ON_DISK, root).build()) {
      // The document that was already there keeps the hash it had, and the next commit stays on
      // version 2 even though it adds nothing that needs it.
      assertThat(reopened.get("c").contentHash()).isEqualTo(BARE);
      assertThat(reopened.get("a").contentHash()).isEqualTo(PREFIXED);
      reopened.add(new Document("e", new float[] {0, 0, 0, 1}, "later", Map.of()));
      reopened.commit();
      assertThat(metadataVersion(root)).isEqualTo(2);
      assertThat(reopened.get("a").contentHash()).isEqualTo(PREFIXED);
      assertThat(reopened.get("e").contentHash()).isEqualTo(ContentHash.of("later"));
    }
  }

  @ParameterizedTest
  @EnumSource(Storage.class)
  void anUpsertReplacesTheHashAlongWithTheDocument(Storage storage, @TempDir Path root)
      throws Exception {
    VectorCollection collection = builder(storage, root).build();
    try {
      collection.add(textWithSuppliedHash());
      collection.commit();
      collection.upsert(new Document("a", new float[] {1, 0, 0, 0}, "beta", Map.of()));
      collection.commit();
      if (storage == Storage.ON_DISK_REOPENED) {
        collection.close();
        collection = builder(storage, root).build();
      }
      assertThat(collection.get("a").contentHash()).isEqualTo(ContentHash.of("beta"));
    } finally {
      collection.close();
    }
  }

  @ParameterizedTest
  @EnumSource(Storage.class)
  void hashesSurviveADeleteAndCompaction(Storage storage, @TempDir Path root) throws Exception {
    VectorCollection collection = builder(storage, root).build();
    try {
      collection.addAll(List.of(textOnly(), textWithSuppliedHash(), noTextWithHash()));
      collection.commit();
      collection.delete("c");
      collection.commit();
      collection.compact();
      if (storage == Storage.ON_DISK_REOPENED) {
        collection.close();
        collection = builder(storage, root).build();
      }
      assertThat(collection.get("a").contentHash()).isEqualTo(PREFIXED);
      assertThat(collection.get("b").contentHash()).isEqualTo(ELSEWHERE);
    } finally {
      collection.close();
    }
  }

  @Test
  void aRecipeCollectionStillHasProvenanceForASourcelessDocumentAfterReopen(@TempDir Path root)
      throws Exception {
    EmbeddingRecipe recipe =
        new EmbeddingRecipe(
            "nomic-embed-text",
            "1.5",
            Optional.of("e".repeat(64)),
            4,
            SimilarityFunction.COSINE,
            true,
            EmbeddingRecipe.Pooling.MEAN,
            Optional.of("search_document: "),
            Optional.of("search_query: "),
            8192,
            EmbeddingRecipe.Truncation.REJECT,
            Map.of());
    try (VectorCollection collection =
        builder(Storage.ON_DISK, root).embeddingRecipe(recipe).build()) {
      collection.add(noTextWithHash());
      collection.add(textWithSuppliedHash());
      collection.commit();
    }
    try (VectorCollection reopened =
        builder(Storage.ON_DISK, root).embeddingRecipe(recipe).build()) {
      recipe.requireProvenance(reopened.get("b"));
      assertThat(reopened.get("b").contentHash()).isEqualTo(ELSEWHERE);
      assertThat(reopened.get("a").matchesContent(recipe.documentInput("alpha"))).isTrue();
    }
  }

  @ParameterizedTest
  @EnumSource(
      value = IndexType.class,
      names = {"FLAT", "HNSW", "VAMANA"})
  void hashesSurviveAReopenWhateverTheIndex(IndexType indexType, @TempDir Path root)
      throws Exception {
    java.util.function.Supplier<VectorCollectionBuilder> builder =
        () ->
            VectorCollection.builder()
                .dimension(4)
                .metric(SimilarityFunction.COSINE)
                .indexType(indexType)
                .autoCommitThreshold(Integer.MAX_VALUE)
                .storagePath(root);
    try (VectorCollection collection = builder.get().build()) {
      collection.addAll(List.of(textOnly(), textWithSuppliedHash(), noTextWithHash()));
      for (int i = 0; i < 40; i++) {
        collection.add(
            new Document("f" + i, new float[] {i + 1, 1, 0.5f, 0.25f}, "filler " + i, Map.of()));
      }
      collection.commit();
    }
    try (VectorCollection reopened = builder.get().build()) {
      assertThat(reopened.get("a").contentHash()).isEqualTo(PREFIXED);
      assertThat(reopened.get("b").contentHash()).isEqualTo(ELSEWHERE);
      assertThat(reopened.get("c").contentHash()).isEqualTo(BARE);
      assertThat(reopened.get("f7").contentHash()).isEqualTo(ContentHash.of("filler 7"));
      Document hit =
          reopened
              .search(SearchRequest.builder(new float[] {1, 0, 0, 0}, 1).build())
              .hits()
              .get(0)
              .document();
      assertThat(hit.id()).isEqualTo("a");
      assertThat(hit.contentHash()).isEqualTo(PREFIXED);
    }
  }

  /** The format version of the newest generation's {@code metadata.bin}. */
  private static int metadataVersion(Path root) throws IOException {
    try (Stream<Path> files = Files.walk(root)) {
      Path newest =
          files
              .filter(path -> path.getFileName().toString().equals(FileFormat.METADATA_FILE))
              .max(Comparator.comparing(path -> path.getParent().getFileName().toString()))
              .orElseThrow();
      byte[] header = Files.readAllBytes(newest);
      return ByteBuffer.wrap(header).order(ByteOrder.LITTLE_ENDIAN).getInt(4);
    }
  }
}
