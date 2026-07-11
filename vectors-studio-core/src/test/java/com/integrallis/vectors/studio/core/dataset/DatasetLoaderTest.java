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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.core.MetadataValue;
import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.VectorCollection;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class DatasetLoaderTest {

  /** In-memory fetcher over a fixed list of rows, honouring offset/length paging. */
  private static final class FakeFetcher implements DatasetLoader.RowsFetcher {
    private final List<Map<String, Object>> rows;
    int calls = 0;

    FakeFetcher(List<Map<String, Object>> rows) {
      this.rows = rows;
    }

    @Override
    public List<Map<String, Object>> fetch(int offset, int length) {
      calls++;
      int from = Math.min(offset, rows.size());
      int to = Math.min(from + length, rows.size());
      return new ArrayList<>(rows.subList(from, to));
    }
  }

  private static Map<String, Object> row(String id, String text, double meta, List<Number> vec) {
    Map<String, Object> m = new LinkedHashMap<>();
    m.put("_id", id);
    m.put("text", text);
    m.put("year", meta);
    m.put("embedding", vec);
    return m;
  }

  @Test
  void loadsCollectionWithTextMetadataAndDimension() throws IOException {
    List<Map<String, Object>> rows =
        List.of(
            row("a", "alpha", 2001, List.of(1.0, 2.0, 3.0, 4.0)),
            row("b", "beta", 2002, List.of(5.0, 6.0, 7.0, 8.0)),
            row("c", "gamma", 2003, List.of(0.1, 0.2, 0.3, 0.4)));
    FakeFetcher fetcher = new FakeFetcher(rows);

    DatasetLoader.Config cfg =
        new DatasetLoader.Config("embedding", "text", "_id", 100, SimilarityFunction.COSINE);
    VectorCollection collection = DatasetLoader.load(cfg, fetcher);

    assertThat(collection.size()).isEqualTo(3);
    assertThat(collection.config().dimension()).isEqualTo(4);

    Document a = collection.get("a");
    assertThat(a).isNotNull();
    assertThat(a.text()).isEqualTo("alpha");
    assertThat(a.vector()).containsExactly(1.0f, 2.0f, 3.0f, 4.0f);
    // vector/text/id columns are excluded from metadata; the extra scalar survives as Num.
    assertThat(a.metadata()).containsKey("year");
    assertThat(a.metadata().get("year")).isEqualTo(MetadataValue.of(2001.0));
    assertThat(a.metadata()).doesNotContainKeys("embedding", "text", "_id");
  }

  @Test
  void respectsLimitAndPagesAcrossMultipleFetches() throws IOException {
    List<Map<String, Object>> rows = new ArrayList<>();
    for (int i = 0; i < 250; i++) {
      rows.add(row("id-" + i, "t" + i, i, List.of((double) i, (double) i, 1.0, 0.0)));
    }
    FakeFetcher fetcher = new FakeFetcher(rows);

    DatasetLoader.Config cfg =
        new DatasetLoader.Config("embedding", "text", "_id", 150, SimilarityFunction.COSINE);
    VectorCollection collection = DatasetLoader.load(cfg, fetcher);

    // limit honoured
    assertThat(collection.size()).isEqualTo(150);
    // 150 rows at 100/page => two fetches (100 + 50)
    assertThat(fetcher.calls).isEqualTo(2);
    assertThat(collection.get("id-0")).isNotNull();
    assertThat(collection.get("id-149")).isNotNull();
    assertThat(collection.get("id-150")).isNull();
  }

  @Test
  void generatesIdsWhenIdColumnNullAndNullTextWhenTextColumnNull() throws IOException {
    List<Map<String, Object>> rows =
        List.of(row("x", "hello", 1, List.of(1.0, 0.0)), row("y", "world", 2, List.of(0.0, 1.0)));
    FakeFetcher fetcher = new FakeFetcher(rows);

    DatasetLoader.Config cfg =
        new DatasetLoader.Config("embedding", null, null, 100, SimilarityFunction.COSINE);
    VectorCollection collection = DatasetLoader.load(cfg, fetcher);

    assertThat(collection.size()).isEqualTo(2);
    Document first = collection.get("row-0");
    assertThat(first).isNotNull();
    assertThat(first.text()).isNull();
    // With no id/text columns, all remaining scalar columns become metadata.
    assertThat(first.metadata()).containsKeys("_id", "text", "year");
  }

  @Test
  void skipsRowsWithMissingOrInvalidVector() throws IOException {
    Map<String, Object> good = row("g", "good", 1, List.of(1.0, 2.0));
    Map<String, Object> noVector = new LinkedHashMap<>();
    noVector.put("_id", "n");
    noVector.put("text", "no vector here");
    Map<String, Object> badVector = row("b", "bad", 2, null);
    badVector.put("embedding", "not-a-list");

    FakeFetcher fetcher = new FakeFetcher(List.of(good, noVector, badVector));
    DatasetLoader.Config cfg =
        new DatasetLoader.Config("embedding", "text", "_id", 100, SimilarityFunction.COSINE);
    VectorCollection collection = DatasetLoader.load(cfg, fetcher);

    assertThat(collection.size()).isEqualTo(1);
    assertThat(collection.get("g")).isNotNull();
  }

  @Test
  void throwsWhenNoVectorsFound() {
    Map<String, Object> noVector = new LinkedHashMap<>();
    noVector.put("_id", "n");
    noVector.put("text", "no vector");
    FakeFetcher fetcher = new FakeFetcher(List.of(noVector));
    DatasetLoader.Config cfg =
        new DatasetLoader.Config("embedding", "text", "_id", 100, SimilarityFunction.COSINE);

    assertThatThrownBy(() -> DatasetLoader.load(cfg, fetcher))
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("no usable vectors");
  }
}
