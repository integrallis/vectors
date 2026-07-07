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

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.QuantizerKind;
import com.integrallis.vectors.db.VectorCollectionConfig;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class ManifestTest {

  private static Manifest sample() {
    VectorCollectionConfig config =
        new VectorCollectionConfig(
            128,
            SimilarityFunction.DOT_PRODUCT,
            IndexType.FLAT,
            QuantizerKind.NONE,
            Integer.MAX_VALUE);
    return Manifest.build(
        config,
        /* generationNumber */ 42L,
        /* liveCount */ 10_000L,
        /* vectorsBinLength */ 5_120_000L,
        /* vectorsBinCrc32 */ 0xDEADBEEFL,
        /* metadataBinLength */ 250_000L,
        /* metadataBinCrc32 */ 0x12345678L,
        /* idmapBinLength */ 45_000L,
        /* idmapBinCrc32 */ 0x87654321L,
        /* graphBinLength */ 131_072L,
        /* graphBinCrc32 */ 0xCAFEBABEL,
        /* quantizedBinLength */ 0L,
        /* quantizedBinCrc32 */ 0L);
  }

  private static Manifest sampleWithTombstones() {
    VectorCollectionConfig config =
        new VectorCollectionConfig(
            128,
            SimilarityFunction.DOT_PRODUCT,
            IndexType.FLAT,
            QuantizerKind.NONE,
            Integer.MAX_VALUE);
    return Manifest.buildWithTombstones(
        config,
        42L,
        10_000L,
        5_120_000L,
        0xDEADBEEFL,
        250_000L,
        0x12345678L,
        45_000L,
        0x87654321L,
        131_072L,
        0xCAFEBABEL,
        0L,
        0L,
        5L,
        128L,
        0xABCDEF01L);
  }

  @Nested
  class Encoding {

    @Test
    void toBytesProducesExactHeaderSize() {
      byte[] encoded = sample().toBytes();
      assertThat(encoded).hasSize(Manifest.HEADER_SIZE);
    }

    @Test
    void headerStartsWithMagicAndVersion() {
      byte[] encoded = sample().toBytes();
      ByteBuffer buf = ByteBuffer.wrap(encoded).order(ByteOrder.LITTLE_ENDIAN);
      assertThat(buf.getInt()).isEqualTo(FileFormat.MAGIC_MANIFEST);
      assertThat(buf.getInt()).isEqualTo(FileFormat.VERSION_MANIFEST);
      assertThat(buf.getInt()).isEqualTo(Manifest.HEADER_SIZE);
      assertThat(buf.getInt()).isEqualTo(0); // flags
    }
  }

  @Nested
  class RoundTrip {

    @Test
    void allFieldsSurviveRoundTrip() throws IOException {
      Manifest original = sample();
      byte[] encoded = original.toBytes();
      Manifest decoded = Manifest.fromBytes(encoded);
      assertThat(decoded.dimension()).isEqualTo(128);
      assertThat(decoded.metric()).isEqualTo(SimilarityFunction.DOT_PRODUCT);
      assertThat(decoded.indexType()).isEqualTo(IndexType.FLAT);
      assertThat(decoded.quantizerKind()).isEqualTo(QuantizerKind.NONE);
      assertThat(decoded.generationNumber()).isEqualTo(42L);
      assertThat(decoded.liveCount()).isEqualTo(10_000L);
      assertThat(decoded.vectorsBinLength()).isEqualTo(5_120_000L);
      assertThat(decoded.vectorsBinCrc32()).isEqualTo(0xDEADBEEFL);
      assertThat(decoded.metadataBinLength()).isEqualTo(250_000L);
      assertThat(decoded.metadataBinCrc32()).isEqualTo(0x12345678L);
      assertThat(decoded.idmapBinLength()).isEqualTo(45_000L);
      assertThat(decoded.idmapBinCrc32()).isEqualTo(0x87654321L);
      assertThat(decoded.graphBinLength()).isEqualTo(131_072L);
      assertThat(decoded.graphBinCrc32()).isEqualTo(0xCAFEBABEL);
      assertThat(decoded.quantizedBinLength()).isEqualTo(0L);
      assertThat(decoded.quantizedBinCrc32()).isEqualTo(0L);
      assertThat(decoded.tombstoneCount()).isEqualTo(0L);
      assertThat(decoded.tombstonesBinLength()).isEqualTo(0L);
      assertThat(decoded.tombstonesBinCrc32()).isEqualTo(0L);
      assertThat(decoded.createdEpochMillis()).isPositive();
    }

    @Test
    void tombstoneFieldsSurviveRoundTrip() throws IOException {
      Manifest original = sampleWithTombstones();
      byte[] encoded = original.toBytes();
      Manifest decoded = Manifest.fromBytes(encoded);
      assertThat(decoded.tombstoneCount()).isEqualTo(5L);
      assertThat(decoded.tombstonesBinLength()).isEqualTo(128L);
      assertThat(decoded.tombstonesBinCrc32()).isEqualTo(0xABCDEF01L);
      // liveCount is the non-tombstoned count.
      assertThat(decoded.liveCount()).isEqualTo(10_000L);
    }

    @Test
    void fileRoundTrip(@TempDir Path tmp) throws IOException {
      Path file = tmp.resolve("manifest.bin");
      Manifest original = sample();
      original.writeTo(file);
      assertThat(Files.size(file)).isEqualTo(Manifest.HEADER_SIZE);
      Manifest decoded = Manifest.readFrom(file);
      assertThat(decoded.vectorsBinCrc32()).isEqualTo(original.vectorsBinCrc32());
      assertThat(decoded.createdEpochMillis()).isEqualTo(original.createdEpochMillis());
    }

    @Test
    void allMetricsRoundTrip() throws IOException {
      for (SimilarityFunction metric : SimilarityFunction.values()) {
        VectorCollectionConfig config =
            new VectorCollectionConfig(
                64, metric, IndexType.FLAT, QuantizerKind.NONE, Integer.MAX_VALUE);
        Manifest m = Manifest.build(config, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L);
        assertThat(Manifest.fromBytes(m.toBytes()).metric()).isEqualTo(metric);
      }
    }

    @Test
    void clockInjectableBuildPreservesExplicitTimestamp() throws IOException {
      VectorCollectionConfig config =
          new VectorCollectionConfig(
              32, SimilarityFunction.COSINE, IndexType.FLAT, QuantizerKind.NONE, Integer.MAX_VALUE);
      long fixedClock = 1_700_000_000_000L;
      Manifest m =
          Manifest.build(config, 7L, fixedClock, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L);
      assertThat(m.createdEpochMillis()).isEqualTo(fixedClock);
      Manifest decoded = Manifest.fromBytes(m.toBytes());
      assertThat(decoded.createdEpochMillis()).isEqualTo(fixedClock);
    }
  }

  @Nested
  class Validation {

    @Test
    void truncatedBytesRejected() {
      byte[] tooShort = new byte[Manifest.HEADER_SIZE - 1];
      assertThatIOException().isThrownBy(() -> Manifest.fromBytes(tooShort));
    }

    @Test
    void wrongMagicRejected() {
      byte[] encoded = sample().toBytes();
      encoded[0] ^= (byte) 0xFF;
      assertThatIOException()
          .isThrownBy(() -> Manifest.fromBytes(encoded))
          .withMessageContaining("magic");
    }

    @Test
    void wrongVersionRejected() {
      byte[] encoded = sample().toBytes();
      ByteBuffer buf = ByteBuffer.wrap(encoded).order(ByteOrder.LITTLE_ENDIAN);
      buf.putInt(4, 999);
      long selfCrc = Checksums.ofBytes(encoded, 0, Manifest.SELF_CRC_OFFSET);
      buf.putInt(Manifest.SELF_CRC_OFFSET, (int) selfCrc);
      assertThatIOException()
          .isThrownBy(() -> Manifest.fromBytes(encoded))
          .withMessageContaining("version");
    }

    @Test
    void wrongHeaderLengthRejected() {
      byte[] encoded = sample().toBytes();
      ByteBuffer buf = ByteBuffer.wrap(encoded).order(ByteOrder.LITTLE_ENDIAN);
      buf.putInt(8, 999);
      long selfCrc = Checksums.ofBytes(encoded, 0, Manifest.SELF_CRC_OFFSET);
      buf.putInt(Manifest.SELF_CRC_OFFSET, (int) selfCrc);
      assertThatIOException()
          .isThrownBy(() -> Manifest.fromBytes(encoded))
          .withMessageContaining("header length");
    }

    @Test
    void unknownFlagsRejected() {
      byte[] encoded = sample().toBytes();
      ByteBuffer buf = ByteBuffer.wrap(encoded).order(ByteOrder.LITTLE_ENDIAN);
      // 0x1 (bit 0 = vectorsNormalized, #A) is a defined flag now; use an undefined bit to trip
      // the "unknown flags" rejection.
      buf.putInt(12, 0x2);
      long selfCrc = Checksums.ofBytes(encoded, 0, Manifest.SELF_CRC_OFFSET);
      buf.putInt(Manifest.SELF_CRC_OFFSET, (int) selfCrc);
      assertThatIOException()
          .isThrownBy(() -> Manifest.fromBytes(encoded))
          .withMessageContaining("flags");
    }

    @Test
    void corruptPayloadTripsSelfCrc() {
      byte[] encoded = sample().toBytes();
      encoded[16] ^= (byte) 0x01;
      assertThatIOException()
          .isThrownBy(() -> Manifest.fromBytes(encoded))
          .withMessageContaining("self-CRC");
    }

    @Test
    void corruptSelfCrcRejected() {
      byte[] encoded = sample().toBytes();
      encoded[Manifest.SELF_CRC_OFFSET] ^= (byte) 0xFF;
      assertThatIOException()
          .isThrownBy(() -> Manifest.fromBytes(encoded))
          .withMessageContaining("self-CRC");
    }

    @Test
    void outOfRangeMetricOrdinalRejected() {
      byte[] encoded = sample().toBytes();
      ByteBuffer buf = ByteBuffer.wrap(encoded).order(ByteOrder.LITTLE_ENDIAN);
      buf.putInt(20, 9999);
      long selfCrc = Checksums.ofBytes(encoded, 0, Manifest.SELF_CRC_OFFSET);
      buf.putInt(Manifest.SELF_CRC_OFFSET, (int) selfCrc);
      assertThatIOException()
          .isThrownBy(() -> Manifest.fromBytes(encoded))
          .withMessageContaining("metric");
    }
  }

  @Nested
  class ConstructorValidation {

    @Test
    void negativeDimensionRejected() {
      assertThatIllegalArgumentException()
          .isThrownBy(
              () ->
                  new Manifest(
                      -1,
                      SimilarityFunction.DOT_PRODUCT,
                      IndexType.FLAT,
                      QuantizerKind.NONE,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L));
    }

    @Test
    void negativeGenerationNumberRejected() {
      assertThatIllegalArgumentException()
          .isThrownBy(
              () ->
                  new Manifest(
                      64,
                      SimilarityFunction.DOT_PRODUCT,
                      IndexType.FLAT,
                      QuantizerKind.NONE,
                      -1L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L));
    }

    @Test
    void negativeCreatedEpochMillisRejected() {
      assertThatIllegalArgumentException()
          .isThrownBy(
              () ->
                  new Manifest(
                      64,
                      SimilarityFunction.DOT_PRODUCT,
                      IndexType.FLAT,
                      QuantizerKind.NONE,
                      0L,
                      -1L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L));
    }

    @Test
    void negativeLiveCountRejected() {
      assertThatIllegalArgumentException()
          .isThrownBy(
              () ->
                  new Manifest(
                      64,
                      SimilarityFunction.DOT_PRODUCT,
                      IndexType.FLAT,
                      QuantizerKind.NONE,
                      0L,
                      0L,
                      -1L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L));
    }

    @Test
    void negativeGraphBinLengthRejected() {
      assertThatIllegalArgumentException()
          .isThrownBy(
              () ->
                  new Manifest(
                      64,
                      SimilarityFunction.DOT_PRODUCT,
                      IndexType.FLAT,
                      QuantizerKind.NONE,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      -1L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L));
    }

    @Test
    void negativeQuantizedBinLengthRejected() {
      assertThatIllegalArgumentException()
          .isThrownBy(
              () ->
                  new Manifest(
                      64,
                      SimilarityFunction.DOT_PRODUCT,
                      IndexType.FLAT,
                      QuantizerKind.NONE,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      -1L,
                      0L,
                      0L,
                      0L,
                      0L));
    }

    @Test
    void negativeTombstoneCountRejected() {
      assertThatIllegalArgumentException()
          .isThrownBy(
              () ->
                  new Manifest(
                      64,
                      SimilarityFunction.DOT_PRODUCT,
                      IndexType.FLAT,
                      QuantizerKind.NONE,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      -1L,
                      0L,
                      0L));
    }

    @Test
    void negativeTombstonesBinLengthRejected() {
      assertThatIllegalArgumentException()
          .isThrownBy(
              () ->
                  new Manifest(
                      64,
                      SimilarityFunction.DOT_PRODUCT,
                      IndexType.FLAT,
                      QuantizerKind.NONE,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      -1L,
                      0L));
    }

    /**
     * Cross-field invariant: {@code tombstoneCount == 0} ⟺ {@code tombstonesBinLength == 0}.
     * Pre-fix audit T2.7 noted that the constructor only checked individual fields for
     * non-negativity, so an inverted manifest claiming "1234 tombstones in 0 bytes" passed
     * validation. The check has to live in the constructor because that's the gate every
     * persisted/derived manifest flows through (build*, fromBytes, direct construction).
     */
    @Test
    void positiveTombstoneCountWithZeroBytesRejected() {
      assertThatIllegalArgumentException()
          .isThrownBy(
              () ->
                  new Manifest(
                      64,
                      SimilarityFunction.DOT_PRODUCT,
                      IndexType.FLAT,
                      QuantizerKind.NONE,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      5L, // tombstoneCount > 0 but tombstonesBinLength == 0 → inconsistent
                      0L,
                      0L))
          .withMessageContaining("tombstone");
    }

    @Test
    void positiveTombstoneBytesWithZeroCountRejected() {
      assertThatIllegalArgumentException()
          .isThrownBy(
              () ->
                  new Manifest(
                      64,
                      SimilarityFunction.DOT_PRODUCT,
                      IndexType.FLAT,
                      QuantizerKind.NONE,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L, // tombstoneCount == 0 but tombstonesBinLength > 0 → inconsistent
                      128L,
                      0L))
          .withMessageContaining("tombstone");
    }

    @Test
    void zeroTombstoneBytesWithNonZeroCrcRejected() {
      assertThatIllegalArgumentException()
          .isThrownBy(
              () ->
                  new Manifest(
                      64,
                      SimilarityFunction.DOT_PRODUCT,
                      IndexType.FLAT,
                      QuantizerKind.NONE,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0L,
                      0xDEADBEEFL)) // tombstonesBinLength == 0 but CRC is set → inconsistent
          .withMessageContaining("tombstone");
    }
  }
}
