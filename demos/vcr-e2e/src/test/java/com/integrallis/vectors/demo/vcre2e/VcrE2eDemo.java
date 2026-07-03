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
package com.integrallis.vectors.demo.vcre2e;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.vcr.VCRMode;
import com.integrallis.vectors.vcr.VCRModel;
import com.integrallis.vectors.vcr.junit5.VCRTest;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

/**
 * End-to-end demonstration of the VCR test harness.
 *
 * <p>The {@link VCRTest} annotation wires the JUnit 5 extension (from {@code vectors-vcr-junit5})
 * that finds all {@link VCRModel} fields on the test instance, and wraps them with the appropriate
 * recorder for their type through the {@code ModelWrapperProvider} service-loader SPI.
 *
 * <p>Because {@code vectors-vcr-langchain4j} is on the classpath, this LangChain4j {@link
 * EmbeddingModel} field is wrapped with {@code VCREmbeddingModel}. The committed cassettes are
 * replayed in strict playback mode, so the default build never rewrites source-tree fixtures or
 * calls the real model.
 *
 * <p>Run:
 *
 * <pre>
 *   ./gradlew :demos:vcr-e2e:test   # strict playback; committed cassettes are required
 * </pre>
 */
@VCRTest(mode = VCRMode.PLAYBACK, dataDir = "src/test/resources/vcr-data")
public class VcrE2eDemo {

  static final AtomicInteger realEmbedderCalls = new AtomicInteger();

  @VCRModel EmbeddingModel embedder = new FakeBackend();

  @Test
  void embedsSingleText() {
    realEmbedderCalls.set(0);
    Response<Embedding> r = embedder.embed("what is HNSW?");
    assertThat(r.content().vector()).hasSize(4);
    assertThat(realEmbedderCalls).hasValue(0);
  }

  @Test
  void embedsBatchOfTexts() {
    realEmbedderCalls.set(0);
    Response<List<Embedding>> r =
        embedder.embedAll(List.of(TextSegment.from("hello"), TextSegment.from("world")));
    assertThat(r.content()).hasSize(2);
    assertThat(r.content().get(0).vector()).hasSize(4);
    assertThat(realEmbedderCalls).hasValue(0);
  }

  /** Minimal LangChain4j {@code EmbeddingModel} standing in for an external provider. */
  static final class FakeBackend implements EmbeddingModel {
    @Override
    public Response<Embedding> embed(String text) {
      realEmbedderCalls.incrementAndGet();
      float[] v = new float[] {(float) text.length(), 42f, 7f, 3f};
      return Response.from(Embedding.from(v));
    }

    @Override
    public Response<Embedding> embed(TextSegment segment) {
      return embed(segment.text());
    }

    @Override
    public Response<List<Embedding>> embedAll(List<TextSegment> segments) {
      List<Embedding> list = segments.stream().map(s -> embed(s).content()).toList();
      return Response.from(list);
    }
  }
}
