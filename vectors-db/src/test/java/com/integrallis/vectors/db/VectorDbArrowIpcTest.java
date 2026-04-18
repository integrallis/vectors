package com.integrallis.vectors.db;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.offset;

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
  }
}
