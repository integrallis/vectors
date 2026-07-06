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
package com.integrallis.vectors.studio.core.dataset;

import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.core.MetadataValue;
import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.VectorCollection;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Pure, transport-agnostic loader that turns a paged stream of dataset rows into a committed {@link
 * VectorCollection}. The row source is abstracted behind {@link RowsFetcher} so this class is fully
 * testable without any network access; {@link HuggingFaceRowsClient} supplies the HTTP-backed
 * implementation for the HuggingFace datasets-server {@code /rows} JSON API.
 */
public final class DatasetLoader {

  /** Rows are fetched a page at a time. */
  private static final int PAGE_SIZE = 100;

  private DatasetLoader() {}

  /**
   * Fetches one page of rows. Each returned map is the flat column-&gt;value map (the inner {@code
   * "row"} object of the datasets-server response) for a single row.
   */
  public interface RowsFetcher {
    List<Map<String, Object>> fetch(int offset, int length) throws IOException;
  }

  /**
   * Loader configuration.
   *
   * @param vectorColumn column holding the embedding (a JSON array of numbers); required
   * @param textColumn column holding raw text, or {@code null} for no text
   * @param idColumn column holding the document id, or {@code null} to generate {@code row-<n>}
   * @param limit maximum number of documents to load
   * @param metric similarity function for the HNSW index
   */
  public record Config(
      String vectorColumn,
      String textColumn,
      String idColumn,
      int limit,
      SimilarityFunction metric) {}

  /**
   * Pages through {@code fetcher} (100 rows/page) up to {@code cfg.limit()}, infers the dimension
   * from the first usable vector, builds an HNSW {@link VectorCollection}, maps each row to a
   * {@link Document}, commits, and returns the collection. Rows whose vector column is missing or
   * is not a numeric list are skipped defensively.
   *
   * @throws IllegalStateException if no usable vectors were found
   */
  public static VectorCollection load(Config cfg, RowsFetcher fetcher) throws IOException {
    int limit = Math.max(0, cfg.limit());
    List<Row> staged = new ArrayList<>();
    int dimension = -1;

    int offset = 0;
    while (staged.size() < limit) {
      int want = Math.min(PAGE_SIZE, limit - staged.size());
      List<Map<String, Object>> page = fetcher.fetch(offset, want);
      if (page == null || page.isEmpty()) {
        break; // exhausted
      }
      for (Map<String, Object> row : page) {
        if (staged.size() >= limit) break;
        float[] vector = toFloatArray(row.get(cfg.vectorColumn()));
        if (vector == null || vector.length == 0) {
          continue; // skip rows with a missing/invalid vector
        }
        if (dimension < 0) {
          dimension = vector.length;
        } else if (vector.length != dimension) {
          continue; // skip ragged vectors
        }
        staged.add(new Row(row, vector));
      }
      offset += page.size();
      if (page.size() < want) {
        break; // last page was short => no more rows
      }
    }

    if (dimension < 0) {
      throw new IllegalStateException("no usable vectors found in dataset");
    }

    VectorCollection collection =
        VectorCollection.builder()
            .dimension(dimension)
            .metric(cfg.metric())
            .indexType(IndexType.HNSW)
            .autoCommitThreshold(Integer.MAX_VALUE)
            .build();

    List<Document> docs = new ArrayList<>(staged.size());
    int n = 0;
    for (Row row : staged) {
      String id = idFor(cfg, row.raw(), n);
      String text = cfg.textColumn() == null ? null : asString(row.raw().get(cfg.textColumn()));
      Map<String, MetadataValue> metadata = metadataFrom(cfg, row.raw());
      docs.add(new Document(id, row.vector(), text, metadata));
      n++;
    }
    collection.addAll(docs);
    collection.commit();
    return collection;
  }

  private record Row(Map<String, Object> raw, float[] vector) {}

  private static String idFor(Config cfg, Map<String, Object> row, int index) {
    if (cfg.idColumn() != null) {
      Object v = row.get(cfg.idColumn());
      if (v != null) {
        String s = String.valueOf(v);
        if (!s.isEmpty()) return s;
      }
    }
    return "row-" + index;
  }

  /**
   * Collects the remaining scalar (String/Number) columns as metadata, excluding the vector, text,
   * and id columns already consumed.
   */
  private static Map<String, MetadataValue> metadataFrom(Config cfg, Map<String, Object> row) {
    Map<String, MetadataValue> metadata = new java.util.LinkedHashMap<>();
    for (Map.Entry<String, Object> e : row.entrySet()) {
      String key = e.getKey();
      if (key.equals(cfg.vectorColumn())
          || key.equals(cfg.textColumn())
          || key.equals(cfg.idColumn())) {
        continue;
      }
      Object v = e.getValue();
      if (v instanceof Number num) {
        metadata.put(key, MetadataValue.of(num.doubleValue()));
      } else if (v instanceof String s) {
        metadata.put(key, MetadataValue.of(s));
      }
      // non-scalar (lists/maps/nested) columns are ignored
    }
    return metadata;
  }

  private static String asString(Object v) {
    return v == null ? null : String.valueOf(v);
  }

  /**
   * Converts a JSON array cell ({@code List<Number>}) into a {@code float[]}, or null if unusable.
   */
  private static float[] toFloatArray(Object cell) {
    if (!(cell instanceof List<?> list) || list.isEmpty()) {
      return null;
    }
    float[] out = new float[list.size()];
    for (int i = 0; i < out.length; i++) {
      Object element = list.get(i);
      if (!(element instanceof Number num)) {
        return null; // not a numeric vector
      }
      out[i] = num.floatValue();
    }
    return out;
  }
}
