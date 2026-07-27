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

import java.util.AbstractList;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Fluent batch builder for {@link Document}s. Collapses the verbose {@code
 * List.of(Document.of(...), Document.of(...))} form into a chain:
 *
 * <pre>{@code
 * collection.addAll(
 *     Documents.of("a", embedding1, "hello world")
 *              .add("b", embedding2, "goodbye world"));
 * }</pre>
 *
 * <p>{@code Documents} <b>is-a</b> {@code List<Document>} (via {@link AbstractList}), so it feeds
 * {@code VectorCollection.addAll(Collection)} directly with no {@code build()} call and no wrapper.
 * Each fluent {@link #add(String, float[], String) add} constructs a {@link Document} record
 * internally, so the call site never repeats {@code Document.of}.
 *
 * <p>The builder mutates in place and is not thread-safe; construct it on one thread and hand the
 * finished list off.
 */
public final class Documents extends AbstractList<Document> {

  private final List<Document> docs;

  private Documents(List<Document> docs) {
    this.docs = docs;
  }

  /** An empty accumulator to start a chain from. */
  public static Documents of() {
    return new Documents(new ArrayList<>());
  }

  /** Seeds the chain with a single id + vector document. */
  public static Documents of(String id, float[] vector) {
    return of().add(id, vector);
  }

  /** Seeds the chain with a single id + vector + text document. */
  public static Documents of(String id, float[] vector, String text) {
    return of().add(id, vector, text);
  }

  /** Wraps existing {@link Document} instances in a fluent, appendable builder. */
  public static Documents of(Document... docs) {
    Documents d = of();
    Collections.addAll(d.docs, docs);
    return d;
  }

  /** Appends an id + vector document (text left null) and returns {@code this} for chaining. */
  public Documents add(String id, float[] vector) {
    docs.add(Document.of(id, vector));
    return this;
  }

  /** Appends an id + vector + text document and returns {@code this} for chaining. */
  public Documents add(String id, float[] vector, String text) {
    docs.add(Document.of(id, vector, text));
    return this;
  }

  /**
   * Appends pre-built {@link Document} instances (e.g. carrying metadata). Named {@code and} rather
   * than {@code add} because {@code List.add(E)} must return {@code boolean}, which would preclude
   * fluent chaining.
   */
  public Documents and(Document... more) {
    Collections.addAll(docs, more);
    return this;
  }

  @Override
  public Document get(int index) {
    return docs.get(index);
  }

  @Override
  public int size() {
    return docs.size();
  }
}
