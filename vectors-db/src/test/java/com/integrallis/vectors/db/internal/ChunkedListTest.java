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
package com.integrallis.vectors.db.internal;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatExceptionOfType;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class ChunkedListTest {

  @Test
  void emptyIsEmpty() {
    ChunkedList<String> list = ChunkedList.empty();
    assertThat(list.size()).isZero();
    assertThat(list.isEmpty()).isTrue();
  }

  @Test
  void singleAppendRoundTrip() {
    ChunkedList<String> list = ChunkedList.<String>empty().append("hello");
    assertThat(list.size()).isEqualTo(1);
    assertThat(list.isEmpty()).isFalse();
    assertThat(list.get(0)).isEqualTo("hello");
  }

  @Test
  void appendDoesNotMutateSource() {
    ChunkedList<String> source = ChunkedList.<String>empty().append("a");
    ChunkedList<String> extended = source.append("b");
    assertThat(source.size()).isEqualTo(1);
    assertThat(source.get(0)).isEqualTo("a");
    assertThat(extended.size()).isEqualTo(2);
    assertThat(extended.get(0)).isEqualTo("a");
    assertThat(extended.get(1)).isEqualTo("b");
  }

  @Test
  void getOutOfBoundsThrows() {
    ChunkedList<String> list = ChunkedList.<String>empty().append("a");
    assertThatExceptionOfType(IndexOutOfBoundsException.class).isThrownBy(() -> list.get(-1));
    assertThatExceptionOfType(IndexOutOfBoundsException.class).isThrownBy(() -> list.get(1));
    assertThatExceptionOfType(IndexOutOfBoundsException.class)
        .isThrownBy(() -> ChunkedList.empty().get(0));
  }

  @Test
  void crossesChunkBoundaryCleanly() {
    // CHUNK_SIZE = 1024; cross it and then some.
    ChunkedList.Builder<Integer> b = ChunkedList.<Integer>empty().toBuilder();
    int n = ChunkedList.CHUNK_SIZE + 250;
    for (int i = 0; i < n; i++) {
      b.append(i);
    }
    ChunkedList<Integer> list = b.build();
    assertThat(list.size()).isEqualTo(n);
    for (int i = 0; i < n; i++) {
      assertThat(list.get(i)).isEqualTo(i);
    }
  }

  @Test
  void crossesMultipleChunkBoundaries() {
    ChunkedList.Builder<Integer> b = ChunkedList.<Integer>empty().toBuilder();
    int n = ChunkedList.CHUNK_SIZE * 5 + 7;
    for (int i = 0; i < n; i++) {
      b.append(i);
    }
    ChunkedList<Integer> list = b.build();
    assertThat(list.size()).isEqualTo(n);
    // Spot check a few across chunk boundaries.
    assertThat(list.get(0)).isZero();
    assertThat(list.get(ChunkedList.CHUNK_SIZE - 1)).isEqualTo(ChunkedList.CHUNK_SIZE - 1);
    assertThat(list.get(ChunkedList.CHUNK_SIZE)).isEqualTo(ChunkedList.CHUNK_SIZE);
    assertThat(list.get(ChunkedList.CHUNK_SIZE * 3)).isEqualTo(ChunkedList.CHUNK_SIZE * 3);
    assertThat(list.get(n - 1)).isEqualTo(n - 1);
  }

  @Test
  void toBuilderProducesStructuralSharingAndBuilderDoesNotMutateSource() {
    ChunkedList.Builder<Integer> a = ChunkedList.<Integer>empty().toBuilder();
    for (int i = 0; i < 2000; i++) {
      a.append(i);
    }
    ChunkedList<Integer> source = a.build();
    // Builder from a populated source.
    ChunkedList.Builder<Integer> b = source.toBuilder();
    for (int i = 0; i < 500; i++) {
      b.append(2000 + i);
    }
    ChunkedList<Integer> extended = b.build();

    // Source unchanged.
    assertThat(source.size()).isEqualTo(2000);
    for (int i = 0; i < 2000; i++) {
      assertThat(source.get(i)).isEqualTo(i);
    }
    // Extended contains all source entries plus new ones.
    assertThat(extended.size()).isEqualTo(2500);
    for (int i = 0; i < 2500; i++) {
      assertThat(extended.get(i)).isEqualTo(i);
    }
  }

  @Test
  void setReplacesAtIndexWithoutMutatingSource() {
    ChunkedList.Builder<Integer> b = ChunkedList.<Integer>empty().toBuilder();
    for (int i = 0; i < 1500; i++) {
      b.append(i);
    }
    ChunkedList<Integer> source = b.build();

    ChunkedList<Integer> modified = source.set(1000, -1);
    assertThat(source.get(1000)).isEqualTo(1000);
    assertThat(modified.get(1000)).isEqualTo(-1);
    // Other indices unchanged in both.
    assertThat(source.get(999)).isEqualTo(999);
    assertThat(modified.get(999)).isEqualTo(999);
    assertThat(source.get(1001)).isEqualTo(1001);
    assertThat(modified.get(1001)).isEqualTo(1001);
  }

  @Test
  void chunkBoundaryAppendOnFullTail() {
    // Build exactly to chunk boundary, then snapshot and continue appending. Exercises the
    // "tail is full → allocate fresh chunk" branch in toBuilder().
    ChunkedList.Builder<Integer> b = ChunkedList.<Integer>empty().toBuilder();
    for (int i = 0; i < ChunkedList.CHUNK_SIZE; i++) {
      b.append(i);
    }
    ChunkedList<Integer> source = b.build();
    assertThat(source.size()).isEqualTo(ChunkedList.CHUNK_SIZE);

    ChunkedList.Builder<Integer> b2 = source.toBuilder();
    b2.append(ChunkedList.CHUNK_SIZE);
    ChunkedList<Integer> extended = b2.build();

    assertThat(extended.size()).isEqualTo(ChunkedList.CHUNK_SIZE + 1);
    assertThat(extended.get(ChunkedList.CHUNK_SIZE - 1)).isEqualTo(ChunkedList.CHUNK_SIZE - 1);
    assertThat(extended.get(ChunkedList.CHUNK_SIZE)).isEqualTo(ChunkedList.CHUNK_SIZE);
    // Source is independent.
    assertThat(source.size()).isEqualTo(ChunkedList.CHUNK_SIZE);
  }

  @Test
  void nullValuesArePermitted() {
    ChunkedList<String> list =
        ChunkedList.<String>empty().toBuilder().append("a").append(null).append("c").build();
    assertThat(list.size()).isEqualTo(3);
    assertThat(list.get(0)).isEqualTo("a");
    assertThat(list.get(1)).isNull();
    assertThat(list.get(2)).isEqualTo("c");
  }
}
