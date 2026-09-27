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
package com.integrallis.vectors.spring.batch;

import com.integrallis.vectors.db.storage.FileFormat;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import org.springframework.batch.item.ItemReader;

/**
 * Emits one collection root per read, for a step that migrates many collections.
 *
 * <p>A directory counts as a collection root when it contains generation directories. Pointing this
 * at a parent — the shape a Studio data directory or a per-tenant layout already has — yields the
 * collections inside it; pointing it at a single collection yields that one.
 *
 * <p>The scan happens once, on the first read, so a directory added while the step is running is
 * not picked up half way through. A batch that saw a moving target would report counts nobody could
 * reconcile afterwards.
 */
public class CollectionRootItemReader implements ItemReader<Path> {

  private final List<Path> roots = new ArrayList<>();
  private final Path base;
  private Iterator<Path> cursor;

  /**
   * Creates a reader over one base directory.
   *
   * @param base a collection root, or a directory containing collection roots
   */
  public CollectionRootItemReader(Path base) {
    this.base = Objects.requireNonNull(base, "base");
  }

  @Override
  public Path read() {
    if (cursor == null) {
      scan();
      cursor = roots.iterator();
    }
    // null is Spring Batch's end-of-input signal, not an error.
    return cursor.hasNext() ? cursor.next() : null;
  }

  /**
   * The roots this reader found, available after the first {@link #read()}.
   *
   * @return an unmodifiable view, empty before the scan has run
   */
  public List<Path> discoveredRoots() {
    return List.copyOf(roots);
  }

  private void scan() {
    try {
      if (!Files.isDirectory(base)) {
        return;
      }
      if (hasGenerations(base)) {
        roots.add(base);
        return;
      }
      try (var stream = Files.list(base)) {
        for (Path child : stream.filter(Files::isDirectory).sorted().toList()) {
          if (hasGenerations(child)) {
            roots.add(child);
          }
        }
      }
    } catch (IOException e) {
      throw new UncheckedIOException("cannot scan " + base + " for collections", e);
    }
  }

  private static boolean hasGenerations(Path path) throws IOException {
    try (var stream = Files.list(path)) {
      return stream
          .filter(Files::isDirectory)
          .anyMatch(p -> p.getFileName().toString().startsWith(FileFormat.GENERATION_DIR_PREFIX));
    }
  }
}
