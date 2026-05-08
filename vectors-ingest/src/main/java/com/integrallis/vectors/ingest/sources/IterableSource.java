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
package com.integrallis.vectors.ingest.sources;

import com.integrallis.vectors.ingest.IngestDoc;
import com.integrallis.vectors.ingest.IngestSource;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import java.util.OptionalLong;

/**
 * Wraps any {@link Iterable} of {@link IngestDoc} as an {@link IngestSource}. The wrapped iterable
 * must be re-iterable when {@code startOffset > 0}; this implementation skips leading docs by
 * count.
 */
public final class IterableSource implements IngestSource {

  private final String name;
  private final Iterable<IngestDoc> docs;
  private final long startOffset;
  private final OptionalLong estimatedSize;

  /** Creates a source named {@code name} starting at offset {@code 0}. */
  public static IterableSource of(String name, Iterable<IngestDoc> docs) {
    long size = (docs instanceof Collection<?> c) ? c.size() : -1L;
    return new IterableSource(
        name, docs, 0L, size >= 0 ? OptionalLong.of(size) : OptionalLong.empty());
  }

  /** Creates a source that resumes at {@code startOffset}. */
  public static IterableSource resuming(String name, Iterable<IngestDoc> docs, long startOffset) {
    long size = (docs instanceof Collection<?> c) ? c.size() : -1L;
    return new IterableSource(
        name, docs, startOffset, size >= 0 ? OptionalLong.of(size) : OptionalLong.empty());
  }

  /** Snapshots an immutable view of {@code docs} eagerly. */
  public static IterableSource copyOf(String name, List<IngestDoc> docs) {
    List<IngestDoc> copy = List.copyOf(docs);
    return new IterableSource(name, copy, 0L, OptionalLong.of(copy.size()));
  }

  private IterableSource(
      String name, Iterable<IngestDoc> docs, long startOffset, OptionalLong estimatedSize) {
    Objects.requireNonNull(name, "name");
    if (name.isBlank()) {
      throw new IllegalArgumentException("name must be non-blank");
    }
    Objects.requireNonNull(docs, "docs");
    if (startOffset < 0) {
      throw new IllegalArgumentException("startOffset must be >= 0");
    }
    this.name = name;
    this.docs = docs;
    this.startOffset = startOffset;
    this.estimatedSize = estimatedSize;
  }

  @Override
  public String name() {
    return name;
  }

  @Override
  public OptionalLong estimatedSize() {
    return estimatedSize;
  }

  @Override
  public long startOffset() {
    return startOffset;
  }

  @Override
  public Iterator<IngestDoc> iterator() {
    Iterator<IngestDoc> it = docs.iterator();
    long skip = startOffset;
    while (skip > 0 && it.hasNext()) {
      it.next();
      skip--;
    }
    return it;
  }
}
