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
package com.integrallis.vectors.core;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * {@link Documents} is the fluent batch builder that reduces the verbosity of assembling a list of
 * {@link Document}s. It is-a {@link List} of {@code Document}, so it drops straight into {@code
 * VectorCollection.addAll(...)} with no wrapper.
 */
class DocumentsTest {

  private static float[] v(float a, float b) {
    return new float[] {a, b};
  }

  @Test
  void isAListOfDocument() {
    Documents docs = Documents.of("a", v(1, 0), "hello");
    assertThat(docs).isInstanceOf(List.class);
    assertThat((List<Document>) docs).hasSize(1);
  }

  @Test
  void seedFactoryThenFluentAddsBuildInOrder() {
    Documents docs = Documents.of("a", v(1, 0), "hello").add("b", v(0, 1), "bye").add("c", v(1, 1));

    assertThat(docs).hasSize(3);
    assertThat(docs.get(0).id()).isEqualTo("a");
    assertThat(docs.get(0).text()).isEqualTo("hello");
    assertThat(docs.get(1).id()).isEqualTo("b");
    assertThat(docs.get(1).text()).isEqualTo("bye");
    assertThat(docs.get(2).id()).isEqualTo("c");
    assertThat(docs.get(2).text()).as("id+vector overload leaves text null").isNull();
  }

  @Test
  void emptyFactoryThenAdds() {
    Documents docs = Documents.of().add("a", v(1, 0)).add("b", v(0, 1), "bye");
    assertThat(docs).hasSize(2);
    assertThat(docs.get(0).text()).isNull();
    assertThat(docs.get(1).text()).isEqualTo("bye");
  }

  @Test
  void varargsFactoryWrapsExistingDocuments() {
    Documents docs = Documents.of(Document.of("a", v(1, 0)), Document.of("b", v(0, 1), "bye"));
    assertThat(docs).hasSize(2);
    assertThat(docs.get(1).text()).isEqualTo("bye");
  }

  @Test
  void andAppendsPrebuiltDocuments() {
    Documents docs =
        Documents.of("a", v(1, 0)).and(Document.of("b", v(0, 1)), Document.of("c", v(1, 1)));
    assertThat(docs).hasSize(3);
    assertThat(docs.get(2).id()).isEqualTo("c");
  }

  @Test
  void iteratesInInsertionOrder() {
    Documents docs = Documents.of("a", v(1, 0)).add("b", v(0, 1));
    StringBuilder ids = new StringBuilder();
    for (Document d : docs) ids.append(d.id());
    assertThat(ids.toString()).isEqualTo("ab");
  }
}
