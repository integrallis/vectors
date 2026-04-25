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
import static org.assertj.core.api.Assertions.offset;

import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.core.MetadataValue;
import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.arrow.ArrowIpcExporter;
import com.integrallis.vectors.db.arrow.ArrowIpcIngester;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Gate tests for Arrow IPC export / ingestion (G6). */
class VectorDbArrowIpcTest {

  private static final int DIM = 4;

  private static VectorCollection newCollection() {
    return VectorCollection.builder()
        .dimension(DIM)
        .metric(SimilarityFunction.EUCLIDEAN)
        .indexType(IndexType.FLAT)
        .quantizer(QuantizerKind.NONE)
        .build();
  }

  /** Round-trip helpers. */
  private static byte[] exportToBytes(VectorCollection col) throws Exception {
    ByteArrayOutputStream bos = new ByteArrayOutputStream();
    ArrowIpcExporter.export(col, bos);
    return bos.toByteArray();
  }

  private static List<Document> importFromBytes(byte[] bytes) throws Exception {
    return ArrowIpcIngester.read(new ByteArrayInputStream(bytes));
  }

  // ---------------------------------------------------------------------------
  // Nested test class
  // ---------------------------------------------------------------------------

  @Nested
  @Tag("unit")
  class ArrowIpcTests {

    @Test
    void export_then_read_roundTrip_preservesIdsAndVectors() throws Exception {
      try (VectorCollection col = newCollection()) {
        col.add(new Document("a", new float[] {1f, 2f, 3f, 4f}, null, Map.of()));
        col.add(new Document("b", new float[] {5f, 6f, 7f, 8f}, null, Map.of()));
        col.commit();

        List<Document> restored = importFromBytes(exportToBytes(col));

        assertThat(restored).hasSize(2);
        Map<String, Document> byId = new java.util.HashMap<>();
        restored.forEach(d -> byId.put(d.id(), d));

        assertThat(byId).containsKey("a");
        assertThat(byId.get("a").vector()).containsExactly(1f, 2f, 3f, 4f);
        assertThat(byId).containsKey("b");
        assertThat(byId.get("b").vector()).containsExactly(5f, 6f, 7f, 8f);
      }
    }

    @Test
    void export_then_read_preservesTextColumn() throws Exception {
      try (VectorCollection col = newCollection()) {
        col.add(new Document("doc1", new float[] {1f, 0f, 0f, 0f}, "hello world", Map.of()));
        col.add(new Document("doc2", new float[] {0f, 1f, 0f, 0f}, null, Map.of()));
        col.commit();

        List<Document> restored = importFromBytes(exportToBytes(col));
        assertThat(restored).hasSize(2);
        Map<String, Document> byId = new java.util.HashMap<>();
        restored.forEach(d -> byId.put(d.id(), d));

        assertThat(byId.get("doc1").text()).isEqualTo("hello world");
        assertThat(byId.get("doc2").text()).isNull();
      }
    }

    @Test
    void export_then_read_preservesAllMetadataTypes() throws Exception {
      Map<String, MetadataValue> meta = new LinkedHashMap<>();
      meta.put("title", MetadataValue.of("vector search"));
      meta.put("count", MetadataValue.of(42L));
      meta.put("ratio", MetadataValue.of(3.14));
      meta.put("active", MetadataValue.of(true));
      meta.put("tags", MetadataValue.tags("a", "b", "c"));

      try (VectorCollection col = newCollection()) {
        col.add(new Document("rich", new float[] {1f, 2f, 3f, 4f}, "text", meta));
        col.commit();

        List<Document> restored = importFromBytes(exportToBytes(col));
        assertThat(restored).hasSize(1);
        Map<String, MetadataValue> got = restored.get(0).metadata();

        assertThat(((MetadataValue.Str) got.get("title")).value()).isEqualTo("vector search");
        assertThat(((MetadataValue.Num) got.get("count")).value()).isEqualTo(42.0);
        assertThat(((MetadataValue.Num) got.get("ratio")).value()).isCloseTo(3.14, offset(1e-9));
        assertThat(((MetadataValue.Bool) got.get("active")).value()).isTrue();
        assertThat(((MetadataValue.Tags) got.get("tags")).values()).containsExactly("a", "b", "c");
      }
    }

    @Test
    void ingest_addsDocumentsToCollection() throws Exception {
      // Build source docs and export to bytes without a collection
      List<Document> source = new ArrayList<>();
      source.add(new Document("x1", new float[] {1f, 0f, 0f, 0f}, null, Map.of()));
      source.add(new Document("x2", new float[] {0f, 1f, 0f, 0f}, null, Map.of()));
      ByteArrayOutputStream bos = new ByteArrayOutputStream();
      ArrowIpcExporter.export(source, DIM, bos);

      // Ingest into a fresh collection
      try (VectorCollection col = newCollection()) {
        int count = ArrowIpcIngester.ingest(col, new ByteArrayInputStream(bos.toByteArray()));
        col.commit();

        assertThat(count).isEqualTo(2);
        assertThat(col.size()).isEqualTo(2);
        assertThat(col.contains("x1")).isTrue();
        assertThat(col.contains("x2")).isTrue();
      }
    }

    @Test
    void emptyCollection_exportAndRead_returnsEmptyList() throws Exception {
      try (VectorCollection col = newCollection()) {
        col.commit();
        List<Document> restored = importFromBytes(exportToBytes(col));
        assertThat(restored).isEmpty();
      }
    }

    @Test
    void largerBatch_moreThanDefaultBatchSize_roundTrips() throws Exception {
      int n = ArrowIpcExporter.DEFAULT_BATCH_SIZE + 50;
      List<Document> source = new ArrayList<>(n);
      for (int i = 0; i < n; i++) {
        source.add(new Document("d" + i, new float[] {i, i + 1f, i + 2f, i + 3f}, null, Map.of()));
      }
      ByteArrayOutputStream bos = new ByteArrayOutputStream();
      ArrowIpcExporter.export(source, DIM, bos);

      List<Document> restored = importFromBytes(bos.toByteArray());
      assertThat(restored).hasSize(n);
    }

    @Test
    void roundTrip_preservesQuotesAndBackslashes() throws Exception {
      Map<String, MetadataValue> meta =
          Map.of("desc", MetadataValue.of("He said \"hello\\world\""));

      try (VectorCollection col = newCollection()) {
        col.add(new Document("esc", new float[] {1f, 0f, 0f, 0f}, null, meta));
        col.commit();

        List<Document> restored = importFromBytes(exportToBytes(col));
        assertThat(restored).hasSize(1);
        assertThat(((MetadataValue.Str) restored.get(0).metadata().get("desc")).value())
            .isEqualTo("He said \"hello\\world\"");
      }
    }

    @Test
    void roundTrip_preservesNewlinesTabsCarriageReturns() throws Exception {
      String text = "line1\nline2\rline3\tend";
      Map<String, MetadataValue> meta = Map.of("content", MetadataValue.of(text));

      try (VectorCollection col = newCollection()) {
        col.add(new Document("ws", new float[] {0f, 1f, 0f, 0f}, null, meta));
        col.commit();

        List<Document> restored = importFromBytes(exportToBytes(col));
        assertThat(((MetadataValue.Str) restored.get(0).metadata().get("content")).value())
            .isEqualTo(text);
      }
    }

    @Test
    void roundTrip_preservesBackspaceAndFormFeed() throws Exception {
      String text = "before\bafter\fend";
      Map<String, MetadataValue> meta = Map.of("ctrl", MetadataValue.of(text));

      try (VectorCollection col = newCollection()) {
        col.add(new Document("bf", new float[] {0f, 0f, 1f, 0f}, null, meta));
        col.commit();

        List<Document> restored = importFromBytes(exportToBytes(col));
        assertThat(((MetadataValue.Str) restored.get(0).metadata().get("ctrl")).value())
            .isEqualTo(text);
      }
    }

    @Test
    void roundTrip_preservesControlCharacters() throws Exception {
      // U+0001 (SOH), U+001F (US) — must be escaped as \u0001 and \u001f
      String text = "start\u0001middle\u001Fend";
      Map<String, MetadataValue> meta = Map.of("ctrl", MetadataValue.of(text));

      try (VectorCollection col = newCollection()) {
        col.add(new Document("ctrl", new float[] {0f, 0f, 0f, 1f}, null, meta));
        col.commit();

        List<Document> restored = importFromBytes(exportToBytes(col));
        assertThat(((MetadataValue.Str) restored.get(0).metadata().get("ctrl")).value())
            .isEqualTo(text);
      }
    }

    @Test
    void roundTrip_preservesNullCharacter() throws Exception {
      // U+0000 (NUL) — the most important control char to handle
      String text = "null\u0000byte";
      Map<String, MetadataValue> meta = Map.of("nul", MetadataValue.of(text));

      try (VectorCollection col = newCollection()) {
        col.add(new Document("nul", new float[] {1f, 1f, 0f, 0f}, null, meta));
        col.commit();

        List<Document> restored = importFromBytes(exportToBytes(col));
        assertThat(((MetadataValue.Str) restored.get(0).metadata().get("nul")).value())
            .isEqualTo(text);
      }
    }

    @Test
    void roundTrip_preservesTagsWithSpecialCharacters() throws Exception {
      Map<String, MetadataValue> meta =
          Map.of("tags", MetadataValue.tags("say \"hi\"", "back\\slash", "new\nline"));

      try (VectorCollection col = newCollection()) {
        col.add(new Document("tags", new float[] {1f, 1f, 1f, 0f}, null, meta));
        col.commit();

        List<Document> restored = importFromBytes(exportToBytes(col));
        assertThat(((MetadataValue.Tags) restored.get(0).metadata().get("tags")).values())
            .containsExactly("say \"hi\"", "back\\slash", "new\nline");
      }
    }
  }
}
