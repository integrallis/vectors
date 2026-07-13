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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.vectors.ingest.IngestDoc;
import java.io.Closeable;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class JsonlSourceTest {

  @Test
  void readsSimpleTextDocs(@TempDir Path tmp) throws IOException {
    Path file = tmp.resolve("docs.jsonl");
    Files.writeString(
        file,
        """
        {"id":"a","text":"alpha"}
        {"id":"b","text":"beta","mime":"text/plain"}
        {"id":"c","text":"gamma","attrs":{"topic":"x"}}
        """,
        StandardCharsets.UTF_8);
    JsonlSource src = new JsonlSource("docs", file);
    List<IngestDoc> docs = new ArrayList<>();
    src.forEach(docs::add);
    assertThat(docs).hasSize(3);
    assertThat(docs.get(0).id()).isEqualTo("a");
    assertThat(docs.get(0).text()).isEqualTo("alpha");
    assertThat(docs.get(2).attrs()).containsEntry("topic", "x");
  }

  @Test
  void readsPrecomputedVector(@TempDir Path tmp) throws IOException {
    Path file = tmp.resolve("docs.jsonl");
    Files.writeString(
        file,
        "{\"id\":\"v1\",\"text\":\"x\",\"precomputedVector\":[0.1,0.2,0.3]}\n",
        StandardCharsets.UTF_8);
    List<IngestDoc> docs = new ArrayList<>();
    new JsonlSource("v", file).forEach(docs::add);
    assertThat(docs).hasSize(1);
    assertThat(docs.get(0).precomputedVector()).containsExactly(0.1f, 0.2f, 0.3f);
  }

  @Test
  void skipsBlankLines(@TempDir Path tmp) throws IOException {
    Path file = tmp.resolve("docs.jsonl");
    Files.writeString(
        file,
        "{\"id\":\"a\",\"text\":\"x\"}\n\n   \n{\"id\":\"b\",\"text\":\"y\"}\n",
        StandardCharsets.UTF_8);
    List<IngestDoc> docs = new ArrayList<>();
    new JsonlSource("d", file).forEach(docs::add);
    assertThat(docs).extracting(IngestDoc::id).containsExactly("a", "b");
  }

  @Test
  void resumesFromOffset(@TempDir Path tmp) throws IOException {
    Path file = tmp.resolve("docs.jsonl");
    Files.writeString(
        file,
        "{\"id\":\"a\",\"text\":\"1\"}\n"
            + "{\"id\":\"b\",\"text\":\"2\"}\n"
            + "{\"id\":\"c\",\"text\":\"3\"}\n",
        StandardCharsets.UTF_8);
    JsonlSource src = new JsonlSource("d", file, 2L, false, n -> {});
    List<IngestDoc> docs = new ArrayList<>();
    src.forEach(docs::add);
    assertThat(docs).extracting(IngestDoc::id).containsExactly("c");
  }

  @Test
  void malformedLineFailsByDefault(@TempDir Path tmp) throws IOException {
    Path file = tmp.resolve("docs.jsonl");
    Files.writeString(
        file,
        "{\"id\":\"a\",\"text\":\"x\"}\nnot-json\n{\"id\":\"b\",\"text\":\"y\"}\n",
        StandardCharsets.UTF_8);
    JsonlSource src = new JsonlSource("d", file);
    assertThatThrownBy(
            () -> {
              for (IngestDoc d : src) {
                // exhaust
                d.id();
              }
            })
        .isInstanceOf(UncheckedIOException.class)
        .hasMessageContaining("malformed JSONL at line 2");
  }

  @Test
  void malformedSkippedWhenRequested(@TempDir Path tmp) throws IOException {
    Path file = tmp.resolve("docs.jsonl");
    Files.writeString(
        file,
        "{\"id\":\"a\",\"text\":\"x\"}\nnot-json\n{\"id\":\"b\",\"text\":\"y\"}\n",
        StandardCharsets.UTF_8);
    AtomicLong observed = new AtomicLong(-1);
    JsonlSource src = new JsonlSource("d", file, 0L, true, observed::set);
    List<IngestDoc> docs = new ArrayList<>();
    src.forEach(docs::add);
    assertThat(docs).extracting(IngestDoc::id).containsExactly("a", "b");
    assertThat(observed.get()).isEqualTo(2L);
  }

  @Test
  void earlyCloseReleasesFileDescriptor(@TempDir Path tmp) throws Exception {
    // Regression: the reader used to close only on natural EOF or IOException; abandoning iteration
    // after reading a prefix leaked one FD. The iterator is now Closeable and close() releases the
    // underlying reader deterministically. We reflect the private reader to prove the FD is gone
    // (a read on a closed BufferedReader throws "Stream closed") — an FD count is not portable.
    Path file = tmp.resolve("docs.jsonl");
    Files.writeString(
        file,
        "{\"id\":\"a\",\"text\":\"x\"}\n{\"id\":\"b\",\"text\":\"y\"}\n{\"id\":\"c\",\"text\":\"z\"}\n",
        StandardCharsets.UTF_8);
    JsonlSource src = new JsonlSource("d", file);

    Iterator<IngestDoc> it = src.iterator();
    assertThat(it).isInstanceOf(Closeable.class);
    assertThat(it.next().id()).isEqualTo("a"); // consume only a prefix, then abandon

    java.lang.reflect.Field readerField = it.getClass().getDeclaredField("reader");
    readerField.setAccessible(true);
    java.io.BufferedReader reader = (java.io.BufferedReader) readerField.get(it);

    ((Closeable) it).close();

    assertThatThrownBy(reader::read)
        .as("reader must be closed after Iter.close()")
        .isInstanceOf(IOException.class)
        .hasMessageContaining("Stream closed");
  }

  @Test
  void closeIsIdempotentAndStopsIteration(@TempDir Path tmp) throws Exception {
    Path file = tmp.resolve("docs.jsonl");
    Files.writeString(
        file,
        "{\"id\":\"a\",\"text\":\"x\"}\n{\"id\":\"b\",\"text\":\"y\"}\n",
        StandardCharsets.UTF_8);
    JsonlSource src = new JsonlSource("d", file);

    Iterator<IngestDoc> it = src.iterator();
    assertThat(it.next().id()).isEqualTo("a"); // consume the buffered doc so `next` is cleared

    Closeable c = (Closeable) it;
    c.close();
    c.close(); // second close must not throw

    assertThat(it.hasNext()).as("closed iterator yields nothing further").isFalse();
    assertThatThrownBy(it::next).isInstanceOf(java.util.NoSuchElementException.class);
  }
}
