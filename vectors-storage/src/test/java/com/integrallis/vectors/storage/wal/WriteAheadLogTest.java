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

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class WriteAheadLogTest {

  @TempDir Path tmp;

  @Test
  void appendReturnsMonotonicallyIncreasingSeqs() throws IOException {
    try (SegmentedWriteAheadLog wal = new SegmentedWriteAheadLog(tmp)) {
      long s0 = wal.append("a".getBytes());
      long s1 = wal.append("b".getBytes());
      long s2 = wal.append("c".getBytes());
      assertThat(s0).isLessThan(s1);
      assertThat(s1).isLessThan(s2);
    }
  }

  @Test
  void replayAfterSealDeliversAllEntries() throws IOException {
    List<String> replayed = new ArrayList<>();
    try (SegmentedWriteAheadLog wal = new SegmentedWriteAheadLog(tmp)) {
      wal.append("entry-0".getBytes());
      wal.append("entry-1".getBytes());
      wal.seal();
      wal.append("entry-2".getBytes());
      wal.seal(); // seal again so entry-2 is in a sealed segment
    }

    try (SegmentedWriteAheadLog wal = new SegmentedWriteAheadLog(tmp)) {
      wal.replay(b -> replayed.add(new String(b)));
    }

    assertThat(replayed).containsExactly("entry-0", "entry-1", "entry-2");
  }

  @Test
  void openSegmentNotReplayed() throws IOException {
    List<String> replayed = new ArrayList<>();
    try (SegmentedWriteAheadLog wal = new SegmentedWriteAheadLog(tmp)) {
      wal.append("sealed".getBytes());
      wal.seal();
      wal.append("open".getBytes()); // in current open segment
      wal.replay(b -> replayed.add(new String(b)));
    }

    // Only the sealed segment's entries are replayed
    assertThat(replayed).containsExactly("sealed");
  }

  @Test
  void nextSeqIncrementsAfterEachAppend() throws IOException {
    try (SegmentedWriteAheadLog wal = new SegmentedWriteAheadLog(tmp)) {
      assertThat(wal.nextSeq()).isEqualTo(0L);
      wal.append("x".getBytes());
      assertThat(wal.nextSeq()).isEqualTo(1L);
      wal.append("y".getBytes());
      assertThat(wal.nextSeq()).isEqualTo(2L);
    }
  }

  @Test
  void reopenedWalContinuesSeqAfterExisting() throws IOException {
    try (SegmentedWriteAheadLog wal = new SegmentedWriteAheadLog(tmp)) {
      wal.append("first".getBytes());
      wal.seal();
    }
    try (SegmentedWriteAheadLog wal = new SegmentedWriteAheadLog(tmp)) {
      long seq = wal.append("second".getBytes());
      assertThat(seq).isEqualTo(1L); // continues after the 1 sealed entry
    }
  }

  @Test
  void crcCorruptionDetectedDuringReplay() throws IOException {
    try (SegmentedWriteAheadLog wal = new SegmentedWriteAheadLog(tmp)) {
      wal.append("payload".getBytes());
      wal.seal();
    }
    // Corrupt the segment file
    Path seg = Files.list(tmp).filter(p -> p.toString().endsWith(".seg")).findFirst().orElseThrow();
    byte[] bytes = Files.readAllBytes(seg);
    bytes[bytes.length - 1] = (byte) (bytes[bytes.length - 1] ^ 0xFF); // flip a CRC byte
    Files.write(seg, bytes);

    try (SegmentedWriteAheadLog wal = new SegmentedWriteAheadLog(tmp)) {
      assertThatThrownBy(() -> wal.replay(b -> {})).isInstanceOf(IOException.class);
    }
  }

  @Test
  void largeBatchAppend_allReplayed() throws IOException {
    int n = 1000;
    try (SegmentedWriteAheadLog wal = new SegmentedWriteAheadLog(tmp)) {
      for (int i = 0; i < n; i++) wal.append(("entry-" + i).getBytes());
      wal.seal();
    }

    List<String> replayed = new ArrayList<>();
    try (SegmentedWriteAheadLog wal = new SegmentedWriteAheadLog(tmp)) {
      wal.replay(b -> replayed.add(new String(b)));
    }

    assertThat(replayed).hasSize(n);
    for (int i = 0; i < n; i++) assertThat(replayed.get(i)).isEqualTo("entry-" + i);
  }

  @Test
  void emptyWalReplayIsNoOp() throws IOException {
    List<String> replayed = new ArrayList<>();
    try (SegmentedWriteAheadLog wal = new SegmentedWriteAheadLog(tmp)) {
      wal.replay(b -> replayed.add(new String(b)));
    }
    assertThat(replayed).isEmpty();
  }
}
