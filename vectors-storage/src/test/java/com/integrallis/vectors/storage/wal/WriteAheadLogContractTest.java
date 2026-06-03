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
package com.integrallis.vectors.storage.wal;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.vectors.storage.backend.HeapStorageBackend;
import com.integrallis.vectors.storage.backend.LocalFileStorageBackend;
import com.integrallis.vectors.storage.backend.StorageBackend;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.stream.Stream;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Contract test for {@link BackendWriteAheadLog} parameterised across {@link HeapStorageBackend}
 * and {@link LocalFileStorageBackend}. Validates the §6.2 / §16.2 invariants from {@code
 * vectors-distributed-design.md}: monotone seq numbers, group commit, segment rollover, CRC-checked
 * replay, indexed/unindexed tail tracking.
 */
@Tag("unit")
class WriteAheadLogContractTest {

  @TempDir Path tmp;

  static Stream<String> backends() {
    return Stream.of("heap", "local");
  }

  private StorageBackend backend(String type) throws IOException {
    return switch (type) {
      case "heap" -> new HeapStorageBackend();
      case "local" -> new LocalFileStorageBackend(Files.createTempDirectory(null));
      default -> throw new IllegalArgumentException(type);
    };
  }

  private BackendWriteAheadLog wal(StorageBackend b) throws IOException {
    return new BackendWriteAheadLog(b, "ns", Duration.ofMillis(50), 512 * 1024 * 1024);
  }

  // ─── core append/replay ───────────────────────────────────────────────────

  @ParameterizedTest
  @MethodSource("backends")
  void appendThenReadRecoversEntry(String type) throws IOException {
    StorageBackend b = backend(type);
    try (BackendWriteAheadLog w = wal(b)) {
      long seq = w.append("hello".getBytes());
      assertThat(seq).isEqualTo(0L);
      try (Stream<WriteAheadLog.WalEntry> s = w.readFrom(0)) {
        List<WriteAheadLog.WalEntry> all = s.toList();
        assertThat(all).hasSize(1);
        assertThat(all.get(0).sequenceNumber()).isEqualTo(0L);
        assertThat(all.get(0).data()).isEqualTo("hello".getBytes());
      }
    }
  }

  @ParameterizedTest
  @MethodSource("backends")
  void sequenceNumbersStrictlyMonotonic(String type) throws IOException {
    StorageBackend b = backend(type);
    try (BackendWriteAheadLog w = wal(b)) {
      long s0 = w.append(new byte[] {1});
      long s1 = w.append(new byte[] {2});
      long s2 = w.append(new byte[] {3});
      assertThat(s0).isEqualTo(0L);
      assertThat(s1).isEqualTo(1L);
      assertThat(s2).isEqualTo(2L);
      assertThat(w.lastSequenceNumber()).isEqualTo(2L);
    }
  }

  @ParameterizedTest
  @MethodSource("backends")
  void readFromMidSeqSkipsEarlierEntries(String type) throws IOException {
    StorageBackend b = backend(type);
    try (BackendWriteAheadLog w = wal(b)) {
      for (int i = 0; i < 10; i++) w.append(new byte[] {(byte) i});
      try (Stream<WriteAheadLog.WalEntry> s = w.readFrom(7)) {
        List<WriteAheadLog.WalEntry> tail = s.toList();
        assertThat(tail).extracting(e -> e.sequenceNumber()).containsExactly(7L, 8L, 9L);
      }
    }
  }

  @ParameterizedTest
  @MethodSource("backends")
  void emptyLogReturnsMinusOneSequence(String type) throws IOException {
    StorageBackend b = backend(type);
    try (BackendWriteAheadLog w = wal(b)) {
      assertThat(w.lastSequenceNumber()).isEqualTo(-1L);
      try (Stream<WriteAheadLog.WalEntry> s = w.readFrom(0)) {
        assertThat(s.toList()).isEmpty();
      }
    }
  }

  @ParameterizedTest
  @MethodSource("backends")
  void reopenedWalRestoresSequenceAndEntries(String type) throws IOException {
    StorageBackend b = backend(type);
    try (BackendWriteAheadLog w = wal(b)) {
      w.append("a".getBytes());
      w.append("b".getBytes());
    }
    try (BackendWriteAheadLog w = wal(b)) {
      assertThat(w.lastSequenceNumber()).isEqualTo(1L);
      long seq = w.append("c".getBytes());
      assertThat(seq).isEqualTo(2L);
      try (Stream<WriteAheadLog.WalEntry> s = w.readFrom(0)) {
        assertThat(s.toList()).extracting(e -> new String(e.data())).containsExactly("a", "b", "c");
      }
    }
  }

  @ParameterizedTest
  @MethodSource("backends")
  void reopenedWalRecoversDurableActiveSegment(String type) throws IOException {
    StorageBackend b = backend(type);
    BackendWriteAheadLog writer = wal(b);
    try {
      writer.append("a".getBytes());
      writer.append("b".getBytes());

      try (BackendWriteAheadLog recovered = wal(b)) {
        assertThat(recovered.lastSequenceNumber()).isEqualTo(1L);
        assertThat(recovered.append("c".getBytes())).isEqualTo(2L);
        try (Stream<WriteAheadLog.WalEntry> s = recovered.readFrom(0)) {
          assertThat(s.toList())
              .extracting(e -> new String(e.data()))
              .containsExactly("a", "b", "c");
        }
      }
    } finally {
      writer.close();
    }
  }

  // ─── segment rollover & integrity ─────────────────────────────────────────

  @ParameterizedTest
  @MethodSource("backends")
  void segmentRolloverCreatesNewSegment(String type) throws IOException {
    StorageBackend b = backend(type);
    // Each frame is 4 + payloadLen + 4 = 12 bytes for a 4-byte payload; cap at 32 forces
    // ≤ 2 frames per segment, so 5 appends produce ≥ 3 segments.
    try (BackendWriteAheadLog w = new BackendWriteAheadLog(b, "ns", Duration.ofMillis(50), 32)) {
      for (int i = 0; i < 5; i++) {
        byte[] payload = new byte[] {(byte) i, (byte) i, (byte) i, (byte) i};
        w.append(payload);
      }
    }
    List<String> wals = b.list("ns/wal/");
    assertThat(wals.size()).isGreaterThanOrEqualTo(3);
  }

  @ParameterizedTest
  @MethodSource("backends")
  void crcCorruptionDetectedOnRead(String type) throws IOException {
    StorageBackend b = backend(type);
    // Force the entry into a closed segment by capping each segment at one frame.
    try (BackendWriteAheadLog w = new BackendWriteAheadLog(b, "ns", Duration.ofMillis(50), 16)) {
      w.append(new byte[] {1, 1, 1, 1});
      w.append(new byte[] {2, 2, 2, 2});
    }
    // Corrupt the second segment so replay must deliver the first valid entry before failing.
    String secondSegKey = b.list("ns/wal/").stream().sorted().skip(1).findFirst().orElseThrow();
    byte[] data = b.get(secondSegKey);
    data[data.length - 1] ^= (byte) 0xFF;
    b.put(secondSegKey, data);

    try (BackendWriteAheadLog w = new BackendWriteAheadLog(b, "ns", Duration.ofMillis(50), 16)) {
      try (Stream<WriteAheadLog.WalEntry> s = w.readFrom(0)) {
        Iterator<WriteAheadLog.WalEntry> iterator = s.iterator();
        WriteAheadLog.WalEntry first = iterator.next();
        assertThat(first.sequenceNumber()).isEqualTo(0L);
        assertThat(first.data()).isEqualTo(new byte[] {1, 1, 1, 1});

        assertThatThrownBy(iterator::hasNext)
            .isInstanceOf(UncheckedIOException.class)
            .hasCauseInstanceOf(IOException.class)
            .hasRootCauseMessage("CRC mismatch in WAL segment ns/wal/000000000002.log at seq 1");
      }
    }
  }

  // ─── group commit ─────────────────────────────────────────────────────────

  @ParameterizedTest
  @MethodSource("backends")
  void groupCommit_concurrentAppendsCoalesceIntoOneObject(String type) throws Exception {
    StorageBackend b = backend(type);
    int n = 500;
    try (BackendWriteAheadLog w =
        new BackendWriteAheadLog(b, "ns", Duration.ofMillis(200), 512 * 1024 * 1024)) {
      CountDownLatch ready = new CountDownLatch(n);
      CountDownLatch go = new CountDownLatch(1);
      try (var pool = Executors.newVirtualThreadPerTaskExecutor()) {
        for (int i = 0; i < n; i++) {
          final int id = i;
          pool.submit(
              () -> {
                ready.countDown();
                try {
                  go.await();
                  w.append(("e-" + id).getBytes());
                } catch (Exception e) {
                  throw new RuntimeException(e);
                }
              });
        }
        ready.await(5, TimeUnit.SECONDS);
        go.countDown();
      } // close awaits all virtual threads
      assertThat(w.lastSequenceNumber()).isEqualTo(n - 1L);
      // All n appends should have coalesced into a single PUT against the active segment.
      assertThat(w.putCount()).isEqualTo(1L);
    }
  }

  @ParameterizedTest
  @MethodSource("backends")
  void groupCommit_overflowingBatchTriggersEarlyFlush(String type) throws IOException {
    StorageBackend b = backend(type);
    // 32-byte cap, 4-byte payload → 12-byte frames → 2 frames/segment.
    try (BackendWriteAheadLog w = new BackendWriteAheadLog(b, "ns", Duration.ofMillis(200), 32)) {
      for (int i = 0; i < 6; i++) w.append(new byte[] {(byte) i, 0, 0, 0});
    }
    // 6 appends × 12-byte frames = 72 bytes total → ≥ 3 segments under the 32-byte cap.
    assertThat(b.list("ns/wal/").size()).isGreaterThanOrEqualTo(3);
  }

  @ParameterizedTest
  @MethodSource("backends")
  void scheduledCommitFailureReleasesWaitingAppender(String type) throws Exception {
    StorageBackend b = new FailingWalSegmentPutBackend(backend(type));
    BackendWriteAheadLog w =
        new BackendWriteAheadLog(b, "ns", Duration.ofMillis(10), 512 * 1024 * 1024);
    try (var pool = Executors.newSingleThreadExecutor()) {
      Future<Long> append = pool.submit(() -> w.append(new byte[] {1, 2, 3}));

      assertThatThrownBy(() -> append.get(5, TimeUnit.SECONDS))
          .isInstanceOf(ExecutionException.class)
          .hasCauseInstanceOf(IOException.class)
          .hasRootCauseMessage("backend put failed: ns/wal/000000000001.log");

      assertThatThrownBy(() -> w.append(new byte[] {4, 5, 6}))
          .isInstanceOf(IOException.class)
          .hasMessage("WAL commit failed")
          .hasRootCauseMessage("backend put failed: ns/wal/000000000001.log");
    } finally {
      w.close();
    }
  }

  // ─── unindexed tail / markIndexed ─────────────────────────────────────────

  @ParameterizedTest
  @MethodSource("backends")
  void unindexedTailSeqs_includesActiveAndClosedUnmarked(String type) throws IOException {
    StorageBackend b = backend(type);
    try (BackendWriteAheadLog w = new BackendWriteAheadLog(b, "ns", Duration.ofMillis(50), 16)) {
      // 12-byte frames, 16-byte cap → 1 frame per segment. Three appends → 2 closed + 1 active.
      w.append(new byte[] {1, 2, 3, 4});
      w.append(new byte[] {5, 6, 7, 8});
      w.append(new byte[] {9, 0, 1, 2});
      long[] tail = w.unindexedTailSeqs();
      // 3 ranges × 2 longs each = 6 longs covering seqs 0,0 / 1,1 / 2,2.
      assertThat(tail).containsExactly(0L, 0L, 1L, 1L, 2L, 2L);
    }
  }

  @ParameterizedTest
  @MethodSource("backends")
  void markIndexed_movesSegmentOutOfTail(String type) throws IOException {
    StorageBackend b = backend(type);
    try (BackendWriteAheadLog w = new BackendWriteAheadLog(b, "ns", Duration.ofMillis(50), 16)) {
      w.append(new byte[] {1, 2, 3, 4});
      w.append(new byte[] {5, 6, 7, 8});
      w.append(new byte[] {9, 0, 1, 2});
      w.markIndexed(0, 1); // covers the first two closed segments
      long[] tail = w.unindexedTailSeqs();
      assertThat(tail).containsExactly(2L, 2L);
    }
  }

  @ParameterizedTest
  @MethodSource("backends")
  void markIndexedCompactsManifestPrefix(String type) throws IOException {
    StorageBackend b = backend(type);
    try (BackendWriteAheadLog w = new BackendWriteAheadLog(b, "ns", Duration.ofMillis(50), 16)) {
      w.append(new byte[] {1, 1, 1, 1});
      w.append(new byte[] {2, 2, 2, 2});
      w.append(new byte[] {3, 3, 3, 3});
      w.append(new byte[] {4, 4, 4, 4});
      w.markIndexed(0, 1);
    }

    String manifest = new String(b.get("ns/manifest.json"), StandardCharsets.UTF_8);
    assertThat(manifest).contains("\"compactedSegmentIndex\":2");
    assertThat(manifest).contains("\"compactedThroughSeq\":1");
    assertThat(manifest).doesNotContain("\"firstSeq\":0");
    assertThat(manifest).doesNotContain("\"firstSeq\":1");

    try (BackendWriteAheadLog reopened =
        new BackendWriteAheadLog(b, "ns", Duration.ofMillis(50), 16)) {
      assertThat(reopened.lastSequenceNumber()).isEqualTo(3L);
      assertThat(reopened.unindexedTailSeqs()).containsExactly(2L, 2L, 3L, 3L);
      assertThat(reopened.append(new byte[] {5, 5, 5, 5})).isEqualTo(4L);

      assertThatThrownBy(() -> reopened.readFrom(0))
          .isInstanceOf(IOException.class)
          .hasMessage("WAL entries compacted through seq 1");
      try (Stream<WriteAheadLog.WalEntry> s = reopened.readFrom(2)) {
        assertThat(s.toList())
            .extracting(WriteAheadLog.WalEntry::sequenceNumber)
            .containsExactly(2L, 3L, 4L);
      }
    }
  }

  private static final class FailingWalSegmentPutBackend implements StorageBackend {
    private final StorageBackend delegate;

    FailingWalSegmentPutBackend(StorageBackend delegate) {
      this.delegate = delegate;
    }

    @Override
    public void put(String key, byte[] value) throws IOException {
      if (key.contains("/wal/")) {
        throw new IOException("backend put failed: " + key);
      }
      delegate.put(key, value);
    }

    @Override
    public byte[] get(String key) throws IOException {
      return delegate.get(key);
    }

    @Override
    public StoredValue getWithEtag(String key) throws IOException {
      return delegate.getWithEtag(key);
    }

    @Override
    public byte[] getRange(String key, long offset, int length) throws IOException {
      return delegate.getRange(key, offset, length);
    }

    @Override
    public List<String> list(String prefix) throws IOException {
      return delegate.list(prefix);
    }

    @Override
    public void delete(String key) throws IOException {
      delegate.delete(key);
    }

    @Override
    public ConditionalPutResult conditionalPut(String key, byte[] value, String expectedEtag)
        throws IOException {
      return delegate.conditionalPut(key, value, expectedEtag);
    }
  }
}
