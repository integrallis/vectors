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
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.OptionalLong;
import java.util.stream.Stream;

/**
 * Walks a directory and emits one {@link IngestDoc} per regular file. Hidden files (starting with
 * {@code .}) are skipped. The doc id is the file path relative to the root, with forward slashes;
 * the payload is the file's bytes (text or blob, depending on the configured MIME).
 *
 * <p>Iteration order is the lexicographic ordering of relative paths so resume-by-offset is
 * deterministic.
 */
public final class DirectorySource implements IngestSource {

  private final String name;
  private final Path root;
  private final long startOffset;
  private final String mime;
  private final boolean asText;

  public DirectorySource(String name, Path root) {
    this(name, root, 0L, "text/plain", true);
  }

  public DirectorySource(String name, Path root, long startOffset, String mime, boolean asText) {
    Objects.requireNonNull(name, "name");
    if (name.isBlank()) {
      throw new IllegalArgumentException("name must be non-blank");
    }
    Objects.requireNonNull(root, "root");
    Objects.requireNonNull(mime, "mime");
    if (startOffset < 0) {
      throw new IllegalArgumentException("startOffset must be >= 0");
    }
    this.name = name;
    this.root = root;
    this.startOffset = startOffset;
    this.mime = mime;
    this.asText = asText;
  }

  @Override
  public String name() {
    return name;
  }

  @Override
  public OptionalLong estimatedSize() {
    try {
      return OptionalLong.of(listFiles().size());
    } catch (IOException e) {
      return OptionalLong.empty();
    }
  }

  @Override
  public long startOffset() {
    return startOffset;
  }

  @Override
  public Iterator<IngestDoc> iterator() {
    List<Path> files;
    try {
      files = listFiles();
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
    if (startOffset >= files.size()) {
      return Collections.emptyIterator();
    }
    return new Iter(files.subList((int) startOffset, files.size()).iterator());
  }

  private List<Path> listFiles() throws IOException {
    if (!Files.exists(root)) {
      return List.of();
    }
    try (Stream<Path> walk = Files.walk(root)) {
      List<Path> all = new ArrayList<>();
      walk.filter(Files::isRegularFile)
          .filter(p -> !p.getFileName().toString().startsWith("."))
          .forEach(all::add);
      all.sort((a, b) -> root.relativize(a).toString().compareTo(root.relativize(b).toString()));
      return all;
    }
  }

  private final class Iter implements Iterator<IngestDoc> {
    private final Iterator<Path> inner;

    Iter(Iterator<Path> inner) {
      this.inner = inner;
    }

    @Override
    public boolean hasNext() {
      return inner.hasNext();
    }

    @Override
    public IngestDoc next() {
      if (!inner.hasNext()) throw new NoSuchElementException();
      Path p = inner.next();
      String id = root.relativize(p).toString().replace('\\', '/');
      try {
        byte[] bytes = Files.readAllBytes(p);
        if (asText) {
          return new IngestDoc(
              id, new String(bytes, StandardCharsets.UTF_8), null, mime, null, null);
        }
        return new IngestDoc(id, null, bytes, mime, null, null);
      } catch (IOException e) {
        throw new UncheckedIOException(e);
      }
    }
  }
}
