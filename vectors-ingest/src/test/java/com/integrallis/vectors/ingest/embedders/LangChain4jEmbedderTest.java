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
package com.integrallis.vectors.ingest.embedders;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.vectors.ingest.IngestDoc;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class LangChain4jEmbedderTest {

  /**
   * Deterministic fake: vector = [text length, batchCallIndex, perDocIndex] padded to {@code dim}.
   */
  private static final class FakeEmbeddingModel implements EmbeddingModel {
    private final int dim;
    private final AtomicInteger batchCalls = new AtomicInteger();
    final List<List<String>> seenBatches = new ArrayList<>();

    FakeEmbeddingModel(int dim) {
      this.dim = dim;
    }

    @Override
    public int dimension() {
      return dim;
    }

    @Override
    public Response<List<Embedding>> embedAll(List<TextSegment> segments) {
      int call = batchCalls.incrementAndGet();
      List<String> texts = segments.stream().map(TextSegment::text).toList();
      seenBatches.add(texts);
      List<Embedding> out = new ArrayList<>(segments.size());
      for (int i = 0; i < segments.size(); i++) {
        float[] v = new float[dim];
        v[0] = segments.get(i).text().length();
        if (dim > 1) v[1] = call;
        if (dim > 2) v[2] = i;
        out.add(Embedding.from(v));
      }
      return Response.from(out);
    }
  }

  @Test
  void embedsAllDocsInOrder() {
    FakeEmbeddingModel model = new FakeEmbeddingModel(3);
    LangChain4jEmbedder e = new LangChain4jEmbedder(model);
    List<float[]> vectors =
        e.embedAll(
            List.of(
                IngestDoc.text("a", "hi"), IngestDoc.text("b", "hello"), IngestDoc.text("c", "x")));
    assertThat(vectors).hasSize(3);
    assertThat(vectors.get(0)[0]).isEqualTo(2f);
    assertThat(vectors.get(1)[0]).isEqualTo(5f);
    assertThat(vectors.get(2)[0]).isEqualTo(1f);
    assertThat(model.seenBatches).hasSize(1);
    assertThat(model.seenBatches.get(0)).containsExactly("hi", "hello", "x");
  }

  @Test
  void exposesNameAndDimension() {
    FakeEmbeddingModel model = new FakeEmbeddingModel(384);
    LangChain4jEmbedder e = new LangChain4jEmbedder(model);
    assertThat(e.dimension()).isEqualTo(384);
    assertThat(e.name()).isEqualTo("langchain4j:FakeEmbeddingModel");
  }

  @Test
  void honoursExplicitNameAndDimension() {
    FakeEmbeddingModel model = new FakeEmbeddingModel(384);
    LangChain4jEmbedder e = new LangChain4jEmbedder(model, 384, "minilm");
    assertThat(e.name()).isEqualTo("minilm");
    assertThat(e.dimension()).isEqualTo(384);
  }

  @Test
  void emptyListShortCircuits() {
    FakeEmbeddingModel model = new FakeEmbeddingModel(3);
    LangChain4jEmbedder e = new LangChain4jEmbedder(model);
    assertThat(e.embedAll(List.of())).isEmpty();
    assertThat(model.seenBatches).isEmpty();
  }

  @Test
  void rejectsDocWithoutText() {
    FakeEmbeddingModel model = new FakeEmbeddingModel(3);
    LangChain4jEmbedder e = new LangChain4jEmbedder(model);
    IngestDoc blob =
        new IngestDoc("blob1", null, new byte[] {1, 2}, "application/octet-stream", null, null);
    assertThatThrownBy(() -> e.embedAll(List.of(blob)))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("blob1")
        .hasMessageContaining("text");
  }

  @Test
  void detectsDimensionMismatchFromModel() {
    EmbeddingModel weirdModel =
        new EmbeddingModel() {
          @Override
          public int dimension() {
            return 3;
          }

          @Override
          public Response<List<Embedding>> embedAll(List<TextSegment> segments) {
            return Response.from(List.of(Embedding.from(new float[] {1f, 2f})));
          }
        };
    LangChain4jEmbedder e = new LangChain4jEmbedder(weirdModel);
    assertThatThrownBy(() -> e.embedAll(List.of(IngestDoc.text("a", "x"))))
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("vector of length 2");
  }

  @Test
  void rejectsBadConstructorArgs() {
    FakeEmbeddingModel model = new FakeEmbeddingModel(3);
    assertThatThrownBy(() -> new LangChain4jEmbedder(model, 0, "n"))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> new LangChain4jEmbedder(model, 3, " "))
        .isInstanceOf(IllegalArgumentException.class);
  }
}
