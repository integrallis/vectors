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
package com.integrallis.vectors.ingest;

import static org.assertj.core.api.Assertions.assertThat;

import java.time.Duration;
import java.util.Optional;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class BatchAccumulatorTest {

  private static EmbeddedDoc d(long offset, String text) {
    return new EmbeddedDoc(IngestDoc.text("id-" + offset, text), new float[] {1f, 2f}, offset);
  }

  @Test
  void flushesWhenMaxDocsHit() {
    BatchAccumulator acc =
        new BatchAccumulator(new BatchPolicy(2, 1024L * 1024L, Duration.ofSeconds(1)));
    assertThat(acc.add(d(0, "a"))).isEmpty();
    Optional<Batch> b = acc.add(d(1, "b"));
    assertThat(b).isPresent();
    assertThat(b.get().batchId()).isEqualTo(0L);
    assertThat(b.get().size()).isEqualTo(2);
    assertThat(b.get().lastSourceOffset()).isEqualTo(1L);
    assertThat(acc.pendingDocs()).isZero();
  }

  @Test
  void flushesWhenMaxBytesHit() {
    BatchAccumulator acc =
        new BatchAccumulator(new BatchPolicy(1000, 4L /* bytes */, Duration.ofSeconds(1)));
    // "ab" → 2 bytes, "cd" → 2 bytes → total 4 ≥ 4 → flush
    assertThat(acc.add(d(0, "ab"))).isEmpty();
    Optional<Batch> b = acc.add(d(1, "cd"));
    assertThat(b).isPresent();
    assertThat(b.get().size()).isEqualTo(2);
  }

  @Test
  void timedFlush() {
    BatchAccumulator acc =
        new BatchAccumulator(new BatchPolicy(1000, 1024L * 1024L, Duration.ofMillis(50)));
    long t0 = System.nanoTime();
    acc.add(d(0, "x"));
    assertThat(acc.flushIfTimedOut(t0 + Duration.ofMillis(10).toNanos())).isEmpty();
    Optional<Batch> b = acc.flushIfTimedOut(t0 + Duration.ofMillis(75).toNanos());
    assertThat(b).isPresent();
    assertThat(b.get().size()).isEqualTo(1);
  }

  @Test
  void drainReturnsRemainingThenEmpty() {
    BatchAccumulator acc =
        new BatchAccumulator(new BatchPolicy(1000, 1024L, Duration.ofSeconds(1)));
    acc.add(d(0, "x"));
    Optional<Batch> first = acc.drain();
    assertThat(first).isPresent();
    assertThat(first.get().size()).isEqualTo(1);
    assertThat(acc.drain()).isEmpty();
  }

  @Test
  void emptyDrainAndTimedFlushAreNoOps() {
    BatchAccumulator acc = new BatchAccumulator(new BatchPolicy(1000, 1024L, Duration.ofMillis(1)));
    assertThat(acc.drain()).isEmpty();
    assertThat(acc.flushIfTimedOut(System.nanoTime() + 1_000_000_000L)).isEmpty();
  }

  @Test
  void batchIdsAreMonotonic() {
    BatchAccumulator acc = new BatchAccumulator(new BatchPolicy(1, 1024L, Duration.ofSeconds(1)));
    assertThat(acc.add(d(0, "x")).orElseThrow().batchId()).isEqualTo(0L);
    assertThat(acc.add(d(1, "y")).orElseThrow().batchId()).isEqualTo(1L);
    assertThat(acc.add(d(2, "z")).orElseThrow().batchId()).isEqualTo(2L);
  }

  @Test
  void pendingCountersTrackBuffer() {
    BatchAccumulator acc =
        new BatchAccumulator(new BatchPolicy(10, 1024L * 1024L, Duration.ofSeconds(1)));
    acc.add(d(0, "abc"));
    acc.add(d(1, "defg"));
    assertThat(acc.pendingDocs()).isEqualTo(2);
    assertThat(acc.pendingBytes()).isEqualTo(7L);
    acc.drain();
    assertThat(acc.pendingDocs()).isZero();
    assertThat(acc.pendingBytes()).isZero();
  }

  @Test
  void sizeOfCountsTextAndBlob() {
    long s = BatchAccumulator.sizeOf(IngestDoc.text("a", "abcd"));
    assertThat(s).isEqualTo(4L);
    IngestDoc bin = new IngestDoc("b", null, new byte[] {1, 2, 3, 4, 5}, "x", null, null);
    assertThat(BatchAccumulator.sizeOf(bin)).isEqualTo(5L);
    IngestDoc both = new IngestDoc("c", "ab", new byte[] {1, 2}, "x", null, null);
    assertThat(BatchAccumulator.sizeOf(both)).isEqualTo(4L);
  }
}
