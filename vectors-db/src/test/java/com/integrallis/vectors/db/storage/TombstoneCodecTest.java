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

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.BitSet;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class TombstoneCodecTest {

  @Nested
  class RoundTrip {

    @Test
    void emptyBitSetEncodesToEmptyArray() {
      BitSet empty = new BitSet();
      byte[] encoded = TombstoneCodec.encode(empty, 100);
      assertThat(encoded).isEmpty();
    }

    @Test
    void emptyBytesDecodeToEmptyBitSet() throws IOException {
      BitSet decoded = TombstoneCodec.decode(new byte[0]);
      assertThat(decoded.isEmpty()).isTrue();
    }

    @Test
    void nullDecodeToEmptyBitSet() throws IOException {
      BitSet decoded = TombstoneCodec.decode(null);
      assertThat(decoded.isEmpty()).isTrue();
    }

    @Test
    void singleBitRoundTrips() throws IOException {
      BitSet bs = new BitSet();
      bs.set(7);
      byte[] encoded = TombstoneCodec.encode(bs, 64);

      assertThat(encoded.length).isEqualTo(TombstoneCodec.HEADER_SIZE + Long.BYTES);

      BitSet decoded = TombstoneCodec.decode(encoded);
      assertThat(decoded).isEqualTo(bs);
      assertThat(decoded.get(7)).isTrue();
      assertThat(decoded.cardinality()).isEqualTo(1);
    }

    @Test
    void sparseBitsRoundTrip() throws IOException {
      BitSet bs = new BitSet();
      bs.set(0);
      bs.set(63);
      bs.set(64);
      bs.set(127);
      bs.set(999);

      byte[] encoded = TombstoneCodec.encode(bs, 1000);
      BitSet decoded = TombstoneCodec.decode(encoded);

      assertThat(decoded).isEqualTo(bs);
      assertThat(decoded.cardinality()).isEqualTo(5);
      assertThat(decoded.get(0)).isTrue();
      assertThat(decoded.get(63)).isTrue();
      assertThat(decoded.get(64)).isTrue();
      assertThat(decoded.get(127)).isTrue();
      assertThat(decoded.get(999)).isTrue();
      assertThat(decoded.get(1)).isFalse();
      assertThat(decoded.get(998)).isFalse();
    }

    @Test
    void denseBitsRoundTrip() throws IOException {
      BitSet bs = new BitSet();
      for (int i = 0; i < 256; i++) {
        bs.set(i);
      }

      byte[] encoded = TombstoneCodec.encode(bs, 256);
      BitSet decoded = TombstoneCodec.decode(encoded);

      assertThat(decoded).isEqualTo(bs);
      assertThat(decoded.cardinality()).isEqualTo(256);
    }

    @Test
    void headerContainsMagicAndVersionAndPhysicalCount() {
      BitSet bs = new BitSet();
      bs.set(0);

      byte[] encoded = TombstoneCodec.encode(bs, 42);
      ByteBuffer buf = ByteBuffer.wrap(encoded).order(ByteOrder.LITTLE_ENDIAN);

      assertThat(buf.getInt()).isEqualTo(FileFormat.MAGIC_TOMBSTONES);
      assertThat(buf.getInt()).isEqualTo(FileFormat.VERSION_TOMBSTONES);
      assertThat(buf.getInt()).isEqualTo(42); // physicalCount
      assertThat(buf.getInt()).isEqualTo(0); // reserved
    }
  }

  @Nested
  class Validation {

    @Test
    void truncatedHeaderRejected() {
      byte[] tooShort = new byte[TombstoneCodec.HEADER_SIZE - 1];
      assertThatIOException()
          .isThrownBy(() -> TombstoneCodec.decode(tooShort))
          .withMessageContaining("truncated");
    }

    @Test
    void wrongMagicRejected() {
      BitSet bs = new BitSet();
      bs.set(0);
      byte[] encoded = TombstoneCodec.encode(bs, 10);
      encoded[0] ^= (byte) 0xFF;
      assertThatIOException()
          .isThrownBy(() -> TombstoneCodec.decode(encoded))
          .withMessageContaining("magic");
    }

    @Test
    void wrongVersionRejected() {
      BitSet bs = new BitSet();
      bs.set(0);
      byte[] encoded = TombstoneCodec.encode(bs, 10);
      ByteBuffer buf = ByteBuffer.wrap(encoded).order(ByteOrder.LITTLE_ENDIAN);
      buf.putInt(4, 999);
      assertThatIOException()
          .isThrownBy(() -> TombstoneCodec.decode(encoded))
          .withMessageContaining("version");
    }

    @Test
    void nonZeroReservedRejected() {
      BitSet bs = new BitSet();
      bs.set(0);
      byte[] encoded = TombstoneCodec.encode(bs, 10);
      ByteBuffer buf = ByteBuffer.wrap(encoded).order(ByteOrder.LITTLE_ENDIAN);
      buf.putInt(12, 1);
      assertThatIOException()
          .isThrownBy(() -> TombstoneCodec.decode(encoded))
          .withMessageContaining("reserved");
    }

    @Test
    void bodySizeNotMultipleOf8Rejected() {
      // Construct a valid header followed by 5 bytes of body (not a multiple of 8).
      byte[] bad = new byte[TombstoneCodec.HEADER_SIZE + 5];
      ByteBuffer buf = ByteBuffer.wrap(bad).order(ByteOrder.LITTLE_ENDIAN);
      buf.putInt(FileFormat.MAGIC_TOMBSTONES);
      buf.putInt(FileFormat.VERSION_TOMBSTONES);
      buf.putInt(10);
      buf.putInt(0);
      assertThatIOException()
          .isThrownBy(() -> TombstoneCodec.decode(bad))
          .withMessageContaining("multiple of 8");
    }
  }
}
