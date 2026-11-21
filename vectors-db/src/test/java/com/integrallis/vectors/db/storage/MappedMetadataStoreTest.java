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

import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.core.MetadataValue;
import com.integrallis.vectors.db.metadata.MetadataStore;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class MappedMetadataStoreTest {

  private static Document plainDoc(String id) {
    return new Document(id, new float[] {1f, 2f, 3f}, null, Map.of());
  }

  private static Document docWithText(String id, String text) {
    return new Document(id, new float[] {1f, 2f, 3f}, text, Map.of());
  }

  private static Document richDoc(String id) {
    Map<String, MetadataValue> metadata = new LinkedHashMap<>();
    metadata.put("title", new MetadataValue.Str("hello"));
    metadata.put("price", MetadataValue.Num.of(9.99));
    metadata.put("inStock", new MetadataValue.Bool(true));
    metadata.put("tags", new MetadataValue.Tags(List.of("a", "b", "c")));
    return new Document(id, new float[] {1f, 2f, 3f}, "raw text", metadata);
  }

  @Nested
  class RoundTrip {

    @Test
    void emptyStore(@TempDir Path tmp) throws IOException {
      Path file = tmp.resolve("metadata.bin");
      MappedMetadataStore.Writer.writeTo(file, List.of());
      try (Arena arena = Arena.ofConfined()) {
        MappedMetadataStore store = MappedMetadataStore.open(file, arena);
        assertThat(store.size()).isEqualTo(0);
        assertThat(store.get(0)).isNull();
      }
    }

    @Test
    void singleDocumentNoMetadataNoText(@TempDir Path tmp) throws IOException {
      Path file = tmp.resolve("metadata.bin");
      Document doc = plainDoc("doc-0");
      MappedMetadataStore.Writer.writeTo(file, List.of(doc));
      try (Arena arena = Arena.ofConfined()) {
        MappedMetadataStore store = MappedMetadataStore.open(file, arena);
        assertThat(store.size()).isEqualTo(1);
        Document read = store.get(0);
        assertThat(read).isNotNull();
        // Vector is never stored in metadata.bin — always null on read.
        assertThat(read.vector()).isNull();
        assertThat(read.id()).isEqualTo("doc-0");
        assertThat(read.text()).isNull();
        assertThat(read.metadata()).isEmpty();
      }
    }

    @Test
    void singleDocumentWithText(@TempDir Path tmp) throws IOException {
      Path file = tmp.resolve("metadata.bin");
      Document doc = docWithText("doc-0", "hello, world — 日本語");
      MappedMetadataStore.Writer.writeTo(file, List.of(doc));
      try (Arena arena = Arena.ofConfined()) {
        MappedMetadataStore store = MappedMetadataStore.open(file, arena);
        Document read = store.get(0);
        assertThat(read.vector()).isNull();
        assertThat(read.id()).isEqualTo("doc-0");
        assertThat(read.text()).isEqualTo("hello, world — 日本語");
        assertThat(read.metadata()).isEmpty();
      }
    }

    @Test
    void allMetadataVariantsRoundTrip(@TempDir Path tmp) throws IOException {
      Path file = tmp.resolve("metadata.bin");
      Document doc = richDoc("doc-42");
      MappedMetadataStore.Writer.writeTo(file, List.of(doc));
      try (Arena arena = Arena.ofConfined()) {
        MappedMetadataStore store = MappedMetadataStore.open(file, arena);
        Document read = store.get(0);
        assertThat(read.vector()).isNull();
        assertThat(read.id()).isEqualTo("doc-42");
        assertThat(read.text()).isEqualTo("raw text");
        assertThat(read.metadata())
            .containsEntry("title", new MetadataValue.Str("hello"))
            .containsEntry("price", new MetadataValue.Num(9.99))
            .containsEntry("inStock", new MetadataValue.Bool(true))
            .containsEntry("tags", new MetadataValue.Tags(List.of("a", "b", "c")));
      }
    }

    @Test
    void equalityIgnoringVector(@TempDir Path tmp) throws IOException {
      Path file = tmp.resolve("metadata.bin");
      Document doc = richDoc("doc-42");
      MappedMetadataStore.Writer.writeTo(file, List.of(doc));
      try (Arena arena = Arena.ofConfined()) {
        MappedMetadataStore store = MappedMetadataStore.open(file, arena);
        Document expected = new Document(doc.id(), null, doc.text(), doc.metadata());
        assertThat(store.get(0)).isEqualTo(expected);
      }
    }

    @Test
    void manyDocuments(@TempDir Path tmp) throws IOException {
      Path file = tmp.resolve("metadata.bin");
      List<Document> docs = new ArrayList<>(500);
      for (int i = 0; i < 500; i++) {
        Map<String, MetadataValue> md = new LinkedHashMap<>();
        md.put("idx", MetadataValue.Num.of(i));
        md.put("even", new MetadataValue.Bool(i % 2 == 0));
        docs.add(new Document("doc-" + i, new float[] {i}, "text-" + i, md));
      }
      MappedMetadataStore.Writer.writeTo(file, docs);
      try (Arena arena = Arena.ofConfined()) {
        MappedMetadataStore store = MappedMetadataStore.open(file, arena);
        assertThat(store.size()).isEqualTo(500);
        for (int i = 0; i < 500; i++) {
          Document read = store.get(i);
          assertThat(read.id()).isEqualTo("doc-" + i);
          assertThat(read.text()).isEqualTo("text-" + i);
          assertThat(((MetadataValue.Num) read.metadata().get("idx")).value())
              .isEqualTo((double) i);
          assertThat(((MetadataValue.Bool) read.metadata().get("even")).value())
              .isEqualTo(i % 2 == 0);
          assertThat(read.vector()).isNull();
        }
      }
    }

    @Test
    void emptyTagList(@TempDir Path tmp) throws IOException {
      Path file = tmp.resolve("metadata.bin");
      Map<String, MetadataValue> md = Map.of("tags", new MetadataValue.Tags(List.of()));
      Document doc = new Document("doc-0", null, null, md);
      MappedMetadataStore.Writer.writeTo(file, List.of(doc));
      try (Arena arena = Arena.ofConfined()) {
        MappedMetadataStore store = MappedMetadataStore.open(file, arena);
        MetadataValue.Tags tags = (MetadataValue.Tags) store.get(0).metadata().get("tags");
        assertThat(tags.values()).isEmpty();
      }
    }

    @Test
    void numericEdgeValues(@TempDir Path tmp) throws IOException {
      Path file = tmp.resolve("metadata.bin");
      Map<String, MetadataValue> md = new LinkedHashMap<>();
      md.put("neg", MetadataValue.Num.of(-1.5));
      md.put("inf", new MetadataValue.Num(Double.POSITIVE_INFINITY));
      md.put("ninf", new MetadataValue.Num(Double.NEGATIVE_INFINITY));
      md.put("zero", new MetadataValue.Num(0.0));
      md.put("max", new MetadataValue.Num(Double.MAX_VALUE));
      Document doc = new Document("doc-0", null, null, md);
      MappedMetadataStore.Writer.writeTo(file, List.of(doc));
      try (Arena arena = Arena.ofConfined()) {
        MappedMetadataStore store = MappedMetadataStore.open(file, arena);
        Map<String, MetadataValue> read = store.get(0).metadata();
        assertThat(((MetadataValue.Num) read.get("neg")).value()).isEqualTo(-1.5);
        assertThat(((MetadataValue.Num) read.get("inf")).value())
            .isEqualTo(Double.POSITIVE_INFINITY);
        assertThat(((MetadataValue.Num) read.get("ninf")).value())
            .isEqualTo(Double.NEGATIVE_INFINITY);
        assertThat(((MetadataValue.Num) read.get("zero")).value()).isEqualTo(0.0);
        assertThat(((MetadataValue.Num) read.get("max")).value()).isEqualTo(Double.MAX_VALUE);
      }
    }
  }

  @Nested
  class BulkRead {

    @Test
    void bulkReadReturnsParallelArray(@TempDir Path tmp) throws IOException {
      Path file = tmp.resolve("metadata.bin");
      List<Document> docs = List.of(plainDoc("a"), plainDoc("b"), plainDoc("c"), plainDoc("d"));
      MappedMetadataStore.Writer.writeTo(file, docs);
      try (Arena arena = Arena.ofConfined()) {
        MappedMetadataStore store = MappedMetadataStore.open(file, arena);
        Document[] out = store.bulkRead(new int[] {2, 0, 3, 1});
        assertThat(out).hasSize(4);
        assertThat(out[0].id()).isEqualTo("c");
        assertThat(out[1].id()).isEqualTo("a");
        assertThat(out[2].id()).isEqualTo("d");
        assertThat(out[3].id()).isEqualTo("b");
      }
    }

    @Test
    void bulkReadReturnsNullForOutOfRange(@TempDir Path tmp) throws IOException {
      Path file = tmp.resolve("metadata.bin");
      MappedMetadataStore.Writer.writeTo(file, List.of(plainDoc("a")));
      try (Arena arena = Arena.ofConfined()) {
        MappedMetadataStore store = MappedMetadataStore.open(file, arena);
        Document[] out = store.bulkRead(new int[] {-1, 0, 99});
        assertThat(out[0]).isNull();
        assertThat(out[1].id()).isEqualTo("a");
        assertThat(out[2]).isNull();
      }
    }
  }

  @Nested
  class Validation {

    @Test
    void missingFileThrows(@TempDir Path tmp) {
      Path file = tmp.resolve("does-not-exist.bin");
      try (Arena arena = Arena.ofConfined()) {
        assertThatIOException().isThrownBy(() -> MappedMetadataStore.open(file, arena));
      }
    }

    @Test
    void truncatedHeaderThrows(@TempDir Path tmp) throws IOException {
      Path file = tmp.resolve("metadata.bin");
      Files.write(file, new byte[8]); // shorter than the 16-byte header
      try (Arena arena = Arena.ofConfined()) {
        assertThatIOException()
            .isThrownBy(() -> MappedMetadataStore.open(file, arena))
            .withMessageContaining("truncated");
      }
    }

    @Test
    void wrongMagicThrows(@TempDir Path tmp) throws IOException {
      Path file = tmp.resolve("metadata.bin");
      MappedMetadataStore.Writer.writeTo(file, List.of(plainDoc("a")));
      byte[] bytes = Files.readAllBytes(file);
      bytes[0] ^= (byte) 0xFF;
      Files.write(file, bytes);
      try (Arena arena = Arena.ofConfined()) {
        assertThatIOException()
            .isThrownBy(() -> MappedMetadataStore.open(file, arena))
            .withMessageContaining("magic");
      }
    }

    @Test
    void wrongVersionThrows(@TempDir Path tmp) throws IOException {
      Path file = tmp.resolve("metadata.bin");
      MappedMetadataStore.Writer.writeTo(file, List.of(plainDoc("a")));
      byte[] bytes = Files.readAllBytes(file);
      ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).putInt(4, 999);
      Files.write(file, bytes);
      try (Arena arena = Arena.ofConfined()) {
        assertThatIOException()
            .isThrownBy(() -> MappedMetadataStore.open(file, arena))
            .withMessageContaining("version");
      }
    }

    @Test
    void negativeCountThrows(@TempDir Path tmp) throws IOException {
      Path file = tmp.resolve("metadata.bin");
      MappedMetadataStore.Writer.writeTo(file, List.of(plainDoc("a")));
      byte[] bytes = Files.readAllBytes(file);
      ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).putInt(8, -1);
      Files.write(file, bytes);
      try (Arena arena = Arena.ofConfined()) {
        assertThatIOException()
            .isThrownBy(() -> MappedMetadataStore.open(file, arena))
            .withMessageContaining("count");
      }
    }

    @Test
    void declaredHeapLargerThanFileThrows(@TempDir Path tmp) throws IOException {
      Path file = tmp.resolve("metadata.bin");
      MappedMetadataStore.Writer.writeTo(file, List.of(plainDoc("a")));
      byte[] bytes = Files.readAllBytes(file);
      ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).putInt(12, 1_000_000);
      Files.write(file, bytes);
      try (Arena arena = Arena.ofConfined()) {
        assertThatIOException()
            .isThrownBy(() -> MappedMetadataStore.open(file, arena))
            .withMessageContaining("truncated");
      }
    }

    @Test
    void nullDocumentInWriterThrows(@TempDir Path tmp) {
      Path file = tmp.resolve("metadata.bin");
      List<Document> docsWithNull = new ArrayList<>();
      docsWithNull.add(plainDoc("a"));
      docsWithNull.add(null);
      assertThatIOException()
          .isThrownBy(() -> MappedMetadataStore.Writer.writeTo(file, docsWithNull))
          .withMessageContaining("null");
    }
  }

  @Nested
  class ReadOnly {

    @Test
    void putThrowsUnsupported(@TempDir Path tmp) throws IOException {
      Path file = tmp.resolve("metadata.bin");
      MappedMetadataStore.Writer.writeTo(file, List.of(plainDoc("a")));
      try (Arena arena = Arena.ofConfined()) {
        MetadataStore store = MappedMetadataStore.open(file, arena);
        assertThatExceptionOfType(UnsupportedOperationException.class)
            .isThrownBy(() -> store.put(0, plainDoc("b")));
      }
    }

    @Test
    void deleteThrowsUnsupported(@TempDir Path tmp) throws IOException {
      Path file = tmp.resolve("metadata.bin");
      MappedMetadataStore.Writer.writeTo(file, List.of(plainDoc("a")));
      try (Arena arena = Arena.ofConfined()) {
        MetadataStore store = MappedMetadataStore.open(file, arena);
        assertThatExceptionOfType(UnsupportedOperationException.class)
            .isThrownBy(() -> store.delete(0));
      }
    }

    @Test
    void getOutOfRangeReturnsNull(@TempDir Path tmp) throws IOException {
      Path file = tmp.resolve("metadata.bin");
      MappedMetadataStore.Writer.writeTo(file, List.of(plainDoc("a")));
      try (Arena arena = Arena.ofConfined()) {
        MappedMetadataStore store = MappedMetadataStore.open(file, arena);
        assertThat(store.get(-1)).isNull();
        assertThat(store.get(1)).isNull();
      }
    }
  }

  @Nested
  class Reopen {

    @Test
    void reopenedStoreSeesSameDocuments(@TempDir Path tmp) throws IOException {
      Path file = tmp.resolve("metadata.bin");
      List<Document> docs = List.of(richDoc("a"), richDoc("b"), richDoc("c"));
      MappedMetadataStore.Writer.writeTo(file, docs);

      try (Arena a1 = Arena.ofConfined()) {
        MappedMetadataStore store = MappedMetadataStore.open(file, a1);
        assertThat(store.size()).isEqualTo(3);
        assertThat(store.get(0).id()).isEqualTo("a");
      }

      try (Arena a2 = Arena.ofConfined()) {
        MappedMetadataStore store = MappedMetadataStore.open(file, a2);
        assertThat(store.size()).isEqualTo(3);
        assertThat(store.get(0).id()).isEqualTo("a");
        assertThat(store.get(1).id()).isEqualTo("b");
        assertThat(store.get(2).id()).isEqualTo("c");
        assertThat(store.get(2).metadata()).containsEntry("price", new MetadataValue.Num(9.99));
      }
    }
  }

  /**
   * #216. Version 1 does not store content hashes and re-derives them from text; version 2 stores
   * them. Version 1 output must not move by a byte, because it is what every existing collection
   * holds and what a collection with nothing but derivable hashes keeps writing.
   */
  @Nested
  class ContentHashes {

    private static final String SUPPLIED = "ab".repeat(32);
    private static final String OTHER = "0123456789abcdef".repeat(4);

    private List<Document> mixed() {
      return List.of(
          new Document("derived", null, "alpha", Map.of()),
          new Document("supplied", null, "alpha", Map.of(), SUPPLIED),
          Document.withHash("sourceless", null, OTHER, Map.of("k", new MetadataValue.Str("v"))),
          new Document("bare", null, null, Map.of()));
    }

    private int version(byte[] file) {
      return ByteBuffer.wrap(file).order(ByteOrder.LITTLE_ENDIAN).getInt(4);
    }

    @Test
    void theLegacyWriterStillProducesVersionOneAndTheSameBytesAsTheExplicitForm()
        throws IOException {
      byte[] legacy = MappedMetadataStore.Writer.toBytes(mixed());

      assertThat(version(legacy)).isEqualTo(1);
      assertThat(MappedMetadataStore.Writer.toBytes(mixed(), false)).isEqualTo(legacy);
    }

    @Test
    void versionOneIsByteForByteTheLayoutItAlwaysWas() throws IOException {
      // One document, id "a", text "hi", no metadata, laid out by hand from the format's
      // documentation: header (magic, version 1, count 1, heap length 7), one offset, then the
      // seven heap bytes: len(id)=1 'a', has-text=1, len(text)=2 'h' 'i', metadata count 0.
      ByteBuffer expected = ByteBuffer.allocate(16 + 8 + 7).order(ByteOrder.LITTLE_ENDIAN);
      expected.putInt(FileFormat.MAGIC_METADATA).putInt(1).putInt(1).putInt(7).putLong(0L);
      expected.put(new byte[] {1, 'a', 1, 2, 'h', 'i', 0});

      assertThat(
              MappedMetadataStore.Writer.toBytes(List.of(new Document("a", null, "hi", Map.of()))))
          .isEqualTo(expected.array());
    }

    @Test
    void versionOneDropsASuppliedHashWhichIsWhyTheCallerMustChooseVersionTwo(@TempDir Path tmp)
        throws IOException {
      Path file = tmp.resolve("metadata.bin");
      MappedMetadataStore.Writer.writeTo(file, mixed());
      try (Arena arena = Arena.ofConfined()) {
        MappedMetadataStore store = MappedMetadataStore.open(file, arena);

        assertThat(store.storesContentHashes()).isFalse();
        assertThat(store.get(1).contentHash()).isEqualTo(store.get(0).contentHash());
        assertThat(store.get(2).contentHash()).isNull();
      }
    }

    @Test
    void versionTwoGivesBackExactlyTheHashEachDocumentHad(@TempDir Path tmp) throws IOException {
      Path file = tmp.resolve("metadata.bin");
      MappedMetadataStore.Writer.writeTo(file, mixed(), true);
      try (Arena arena = Arena.ofConfined()) {
        MappedMetadataStore store = MappedMetadataStore.open(file, arena);

        assertThat(version(Files.readAllBytes(file))).isEqualTo(2);
        assertThat(store.storesContentHashes()).isTrue();
        assertThat(store.get(0).contentHash())
            .isEqualTo(com.integrallis.vectors.core.ContentHash.of("alpha"));
        assertThat(store.get(1).contentHash()).isEqualTo(SUPPLIED);
        assertThat(store.get(1).text()).isEqualTo("alpha");
        assertThat(store.get(2).contentHash()).isEqualTo(OTHER);
        assertThat(store.get(2).text()).isNull();
        assertThat(store.get(2).metadata()).containsEntry("k", new MetadataValue.Str("v"));
        assertThat(store.get(3).contentHash()).isNull();
        assertThat(store.get(3).text()).isNull();
      }
    }

    @Test
    void theStreamingImageWritesTheSameVersionTwoBytes(@TempDir Path tmp) throws IOException {
      Path file = tmp.resolve("metadata.bin");
      byte[] buffered = MappedMetadataStore.Writer.toBytes(mixed(), true);
      MappedMetadataStore.Writer.Image image = new MappedMetadataStore.Writer.Image(mixed(), true);

      image.writeTo(file);

      assertThat(Files.readAllBytes(file)).isEqualTo(buffered);
      assertThat(image.length()).isEqualTo(buffered.length);
      java.util.zip.CRC32 crc = new java.util.zip.CRC32();
      crc.update(buffered);
      assertThat(image.crc32()).isEqualTo(crc.getValue());
      assertThat(new MappedMetadataStore.Writer.Image(mixed(), true).crc32())
          .isEqualTo(crc.getValue());
      assertThat(new MappedMetadataStore.Writer.Image(mixed()).length())
          .isEqualTo(MappedMetadataStore.Writer.toBytes(mixed()).length);
    }

    @Test
    void aHashFlagInAVersionOneFileIsRefused(@TempDir Path tmp) throws IOException {
      Path file = tmp.resolve("metadata.bin");
      byte[] bytes = MappedMetadataStore.Writer.toBytes(mixed(), true);
      ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).putInt(4, 1);
      Files.write(file, bytes);
      try (Arena arena = Arena.ofConfined()) {
        assertThatIOException()
            .isThrownBy(() -> MappedMetadataStore.open(file, arena))
            .withMessageContaining("has-text flag must be 0 or 1");
      }
    }

    @Test
    void anUnknownFlagBitInAVersionTwoFileIsRefused(@TempDir Path tmp) throws IOException {
      Path file = tmp.resolve("metadata.bin");
      byte[] bytes =
          MappedMetadataStore.Writer.toBytes(
              List.of(new Document("a", null, null, Map.of())), true);
      // header 16 + one offset 8, then len(id)=1, 'a', flags
      bytes[16 + 8 + 2] = 4;
      Files.write(file, bytes);
      try (Arena arena = Arena.ofConfined()) {
        assertThatIOException()
            .isThrownBy(() -> MappedMetadataStore.open(file, arena))
            .withMessageContaining("flags must be 0..3");
      }
    }

    @Test
    void aHashCutOffByTheEndOfTheHeapIsRefused(@TempDir Path tmp) throws IOException {
      Path file = tmp.resolve("metadata.bin");
      byte[] whole =
          MappedMetadataStore.Writer.toBytes(
              List.of(Document.withHash("a", null, SUPPLIED, Map.of())), true);
      // Drop the metadata count and the last 10 bytes of the hash, and shrink the declared heap.
      byte[] cut = java.util.Arrays.copyOf(whole, whole.length - 11);
      ByteBuffer.wrap(cut).order(ByteOrder.LITTLE_ENDIAN).putInt(12, cut.length - 16 - 8);
      Files.write(file, cut);
      try (Arena arena = Arena.ofConfined()) {
        assertThatIOException()
            .isThrownBy(() -> MappedMetadataStore.open(file, arena))
            .withMessageContaining("content hash out of range");
      }
    }

    @Test
    void aVersionTwoFileWithATextDocumentAndNoHashIsRefused(@TempDir Path tmp) throws IOException {
      Path file = tmp.resolve("metadata.bin");
      // A version 1 body relabelled as version 2: text present, no stored hash. This writer never
      // produces that, so reading it must not invent a hash for the document.
      byte[] bytes =
          MappedMetadataStore.Writer.toBytes(List.of(new Document("a", null, "hi", Map.of())));
      ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).putInt(4, 2);
      Files.write(file, bytes);
      try (Arena arena = Arena.ofConfined()) {
        assertThatIOException()
            .isThrownBy(() -> MappedMetadataStore.open(file, arena))
            .withMessageContaining("stores no content hash");
      }
    }
  }
}
