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
package com.integrallis.vectors.db.storage;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatIOException;
import static org.assertj.core.api.Assertions.assertThatIllegalArgumentException;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Random;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class FileFormatTest {

  @Nested
  class Magic {

    @Test
    void manifestMagicSpellsVdbvOnDisk() {
      ByteBuffer buf = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
      buf.putInt(FileFormat.MAGIC_MANIFEST);
      buf.flip();
      assertThat(buf.get()).isEqualTo((byte) 'V');
      assertThat(buf.get()).isEqualTo((byte) 'D');
      assertThat(buf.get()).isEqualTo((byte) 'B');
      assertThat(buf.get()).isEqualTo((byte) 'V');
    }

    @Test
    void idmapMagicSpellsVidpOnDisk() {
      ByteBuffer buf = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
      buf.putInt(FileFormat.MAGIC_IDMAP);
      buf.flip();
      assertThat(buf.get()).isEqualTo((byte) 'V');
      assertThat(buf.get()).isEqualTo((byte) 'I');
      assertThat(buf.get()).isEqualTo((byte) 'D');
      assertThat(buf.get()).isEqualTo((byte) 'P');
    }

    @Test
    void metadataMagicSpellsVmdbOnDisk() {
      ByteBuffer buf = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
      buf.putInt(FileFormat.MAGIC_METADATA);
      buf.flip();
      assertThat(buf.get()).isEqualTo((byte) 'V');
      assertThat(buf.get()).isEqualTo((byte) 'M');
      assertThat(buf.get()).isEqualTo((byte) 'D');
      assertThat(buf.get()).isEqualTo((byte) 'B');
    }
  }

  @Nested
  class GenerationDirNames {

    @Test
    void formatsZeroPaddedGenerationDir() {
      assertThat(FileFormat.generationDirName(0L)).isEqualTo("gen-0000000000000000");
      assertThat(FileFormat.generationDirName(1L)).isEqualTo("gen-0000000000000001");
      assertThat(FileFormat.generationDirName(42L)).isEqualTo("gen-0000000000000042");
      assertThat(FileFormat.generationDirName(FileFormat.MAX_GENERATION_NUMBER))
          .isEqualTo("gen-9999999999999999");
    }

    @Test
    void formatsZeroPaddedTmpDir() {
      assertThat(FileFormat.generationTmpDirName(0L)).isEqualTo(".gen-0000000000000000.tmp");
      assertThat(FileFormat.generationTmpDirName(42L)).isEqualTo(".gen-0000000000000042.tmp");
    }

    @Test
    void negativeGenerationNumberThrows() {
      assertThatIllegalArgumentException().isThrownBy(() -> FileFormat.generationDirName(-1L));
      assertThatIllegalArgumentException().isThrownBy(() -> FileFormat.generationTmpDirName(-1L));
    }

    @Test
    void generationNumberAboveSixteenDigitsThrows() {
      long tooBig = FileFormat.MAX_GENERATION_NUMBER + 1L;
      assertThatIllegalArgumentException()
          .isThrownBy(() -> FileFormat.generationDirName(tooBig))
          .withMessageContaining("exceeds");
      assertThatIllegalArgumentException()
          .isThrownBy(() -> FileFormat.generationTmpDirName(tooBig))
          .withMessageContaining("exceeds");
    }

    @Test
    void parseRoundTrip() {
      for (long g : new long[] {0L, 1L, 42L, FileFormat.MAX_GENERATION_NUMBER}) {
        assertThat(FileFormat.parseGenerationDirName(FileFormat.generationDirName(g))).isEqualTo(g);
      }
    }

    @Test
    void parseRejectsInvalidNames() {
      assertThat(FileFormat.parseGenerationDirName(null)).isEqualTo(-1L);
      assertThat(FileFormat.parseGenerationDirName("")).isEqualTo(-1L);
      assertThat(FileFormat.parseGenerationDirName("gen-42")).isEqualTo(-1L); // not 16 digits
      assertThat(FileFormat.parseGenerationDirName("gen-00000042")).isEqualTo(-1L); // was 8-digit
      assertThat(FileFormat.parseGenerationDirName("gen-000000000000004x")).isEqualTo(-1L);
      assertThat(FileFormat.parseGenerationDirName(".gen-0000000000000042.tmp")).isEqualTo(-1L);
      assertThat(FileFormat.parseGenerationDirName("foo-0000000000000001")).isEqualTo(-1L);
    }

    @Test
    void lexicographicSortMatchesNumericSort() {
      // Critical regression: must include boundary values where 8-digit formatting would have
      // silently sorted incorrectly. 99_999_999 and 100_000_000 straddle the old 8-digit boundary.
      String a = FileFormat.generationDirName(5L);
      String b = FileFormat.generationDirName(10L);
      String c = FileFormat.generationDirName(999L);
      String d = FileFormat.generationDirName(99_999_999L);
      String e = FileFormat.generationDirName(100_000_000L);
      String f = FileFormat.generationDirName(FileFormat.MAX_GENERATION_NUMBER);
      assertThat(a.compareTo(b)).isNegative();
      assertThat(b.compareTo(c)).isNegative();
      assertThat(c.compareTo(d)).isNegative();
      assertThat(d.compareTo(e)).isNegative();
      assertThat(e.compareTo(f)).isNegative();
    }
  }

  @Nested
  class Varint {

    @Test
    void singleByteValues() throws IOException {
      for (int v : new int[] {0, 1, 42, 126, 127}) {
        assertThat(FileFormat.varIntSize(v)).isEqualTo(1);
        ByteBuffer buf = ByteBuffer.allocate(5);
        FileFormat.writeVarInt(buf, v);
        buf.flip();
        assertThat(buf.remaining()).isEqualTo(1);
        assertThat(FileFormat.readVarInt(buf)).isEqualTo(v);
      }
    }

    @Test
    void twoByteValues() throws IOException {
      for (int v : new int[] {128, 255, 1000, 16383}) {
        assertThat(FileFormat.varIntSize(v)).isEqualTo(2);
        ByteBuffer buf = ByteBuffer.allocate(5);
        FileFormat.writeVarInt(buf, v);
        buf.flip();
        assertThat(buf.remaining()).isEqualTo(2);
        assertThat(FileFormat.readVarInt(buf)).isEqualTo(v);
      }
    }

    @Test
    void multiByteValues() throws IOException {
      // 16384 = 2^14 → 3 bytes; 2_097_152 = 2^21 → 4 bytes; MAX_VALUE → 5 bytes.
      int[] cases = {16384, 2_097_152, Integer.MAX_VALUE};
      int[] expectedSizes = {3, 4, 5};
      for (int i = 0; i < cases.length; i++) {
        assertThat(FileFormat.varIntSize(cases[i])).isEqualTo(expectedSizes[i]);
        ByteBuffer buf = ByteBuffer.allocate(5);
        FileFormat.writeVarInt(buf, cases[i]);
        buf.flip();
        assertThat(buf.remaining()).isEqualTo(expectedSizes[i]);
        assertThat(FileFormat.readVarInt(buf)).isEqualTo(cases[i]);
      }
    }

    @Test
    void randomRoundTrip() throws IOException {
      Random rng = new Random(42L);
      for (int i = 0; i < 10_000; i++) {
        int v = rng.nextInt(Integer.MAX_VALUE);
        ByteBuffer buf = ByteBuffer.allocate(5);
        int written = FileFormat.writeVarInt(buf, v);
        assertThat(written).isEqualTo(FileFormat.varIntSize(v));
        buf.flip();
        assertThat(FileFormat.readVarInt(buf)).isEqualTo(v);
      }
    }

    @Test
    void negativeValuesRejected() {
      ByteBuffer buf = ByteBuffer.allocate(5);
      assertThatIllegalArgumentException().isThrownBy(() -> FileFormat.writeVarInt(buf, -1));
      assertThatIllegalArgumentException().isThrownBy(() -> FileFormat.varIntSize(-1));
    }

    @Test
    void truncatedVarintInBufferThrows() {
      ByteBuffer buf = ByteBuffer.allocate(5);
      buf.put((byte) 0x80); // continuation bit set, but no follow-up byte
      buf.flip();
      assertThatIOException().isThrownBy(() -> FileFormat.readVarInt(buf));
    }

    @Test
    void moreThanFiveContinuationBytesThrows() {
      // Note: this does NOT test LEB128 canonical-form enforcement (readVarInt silently accepts
      // a non-canonical 0x80 0x00 as zero, matching Protobuf). It only pins the 5-byte limit
      // that bounds read cost and prevents attacker-controlled loops.
      ByteBuffer buf = ByteBuffer.allocate(6);
      for (int i = 0; i < 5; i++) {
        buf.put((byte) 0x80);
      }
      buf.put((byte) 0x01);
      buf.flip();
      assertThatIOException().isThrownBy(() -> FileFormat.readVarInt(buf));
    }

    @Test
    void memorySegmentReadRoundTrip() throws IOException {
      try (Arena arena = Arena.ofConfined()) {
        MemorySegment seg = arena.allocate(16);
        ByteBuffer view = seg.asByteBuffer();
        // Write three varints of different widths back-to-back.
        FileFormat.writeVarInt(view, 42);
        FileFormat.writeVarInt(view, 16383);
        FileFormat.writeVarInt(view, 1_000_000);

        long offset = 0;
        long r1 = FileFormat.readVarIntFromSegment(seg, offset);
        assertThat((int) r1).isEqualTo(42);
        offset += (int) (r1 >>> 32);
        long r2 = FileFormat.readVarIntFromSegment(seg, offset);
        assertThat((int) r2).isEqualTo(16383);
        offset += (int) (r2 >>> 32);
        long r3 = FileFormat.readVarIntFromSegment(seg, offset);
        assertThat((int) r3).isEqualTo(1_000_000);
      }
    }

    @Test
    void segmentVarintOutOfBoundsThrows() {
      try (Arena arena = Arena.ofConfined()) {
        MemorySegment seg = arena.allocate(1);
        seg.set(java.lang.foreign.ValueLayout.JAVA_BYTE, 0, (byte) 0x80);
        assertThatIOException().isThrownBy(() -> FileFormat.readVarIntFromSegment(seg, 0L));
      }
    }
  }
}
