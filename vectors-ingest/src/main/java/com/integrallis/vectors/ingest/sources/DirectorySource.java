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

  /**
   * Default per-file size cap: 64 MiB. Each file is loaded whole into heap as one {@link
   * IngestDoc}, so an unbounded read of a multi-GB file would OOM the JVM (and no amount of
   * horizontal worker-partitioning helps — a single file always lands whole on one worker; only
   * streaming or pre-chunking raises the single-file ceiling). 64 MiB matches TurboPuffer's
   * max-document limit, and is far above any sane text-embedding unit (embedding models cap at a
   * few thousand tokens), so it rejects only pathological inputs. Raise it via the {@code
   * maxFileBytes} constructor parameter for large-blob ingests, or pre-chunk the input.
   */
  public static final long DEFAULT_MAX_FILE_BYTES = 64L * 1024 * 1024;

  private final String name;
  private final Path root;
  private final long startOffset;
  private final String mime;
  private final boolean asText;
  private final long maxFileBytes;
  // Snapshot of the walked+sorted file list, computed once and reused. Without it, estimatedSize()
  // and iterator() each full-walk+sort the tree (and every re-iteration for resume walks again).
  // Snapshotting also pins a stable file set/order across re-iterations, which is exactly what
  // resume-by-offset relies on. Volatile with a benign compute race (the walk is idempotent).
  private volatile List<Path> cachedFiles;

  public DirectorySource(String name, Path root) {
    this(name, root, 0L, "text/plain", true);
  }

  public DirectorySource(String name, Path root, long startOffset, String mime, boolean asText) {
    this(name, root, startOffset, mime, asText, DEFAULT_MAX_FILE_BYTES);
  }

  public DirectorySource(
      String name, Path root, long startOffset, String mime, boolean asText, long maxFileBytes) {
    Objects.requireNonNull(name, "name");
    if (name.isBlank()) {
      throw new IllegalArgumentException("name must be non-blank");
    }
    Objects.requireNonNull(root, "root");
    Objects.requireNonNull(mime, "mime");
    if (startOffset < 0) {
      throw new IllegalArgumentException("startOffset must be >= 0");
    }
    if (maxFileBytes <= 0) {
      throw new IllegalArgumentException("maxFileBytes must be > 0: " + maxFileBytes);
    }
    this.name = name;
    this.root = root;
    this.startOffset = startOffset;
    this.mime = mime;
    this.asText = asText;
    this.maxFileBytes = maxFileBytes;
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
    List<Path> cached = cachedFiles;
    if (cached != null) {
      return cached;
    }
    List<Path> snapshot;
    if (!Files.exists(root)) {
      snapshot = List.of();
    } else {
      try (Stream<Path> walk = Files.walk(root)) {
        List<Path> all = new ArrayList<>();
        walk.filter(Files::isRegularFile)
            .filter(p -> !p.getFileName().toString().startsWith("."))
            .forEach(all::add);
        all.sort((a, b) -> root.relativize(a).toString().compareTo(root.relativize(b).toString()));
        snapshot = List.copyOf(all);
      }
    }
    cachedFiles = snapshot;
    return snapshot;
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
        // Stat before reading: a whole-file readAllBytes on a multi-GB file would OOM the heap.
        // Reject oversized files with a clear, actionable error instead.
        long size = Files.size(p);
        if (size > maxFileBytes) {
          throw new IOException(
              "file '"
                  + id
                  + "' is "
                  + size
                  + " bytes, exceeding the per-file cap of "
                  + maxFileBytes
                  + " bytes; raise it via the DirectorySource maxFileBytes parameter or pre-chunk"
                  + " the file into embeddable segments");
        }
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
