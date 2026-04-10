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
        /* idmapBinCrc32 */ 0x87654321L);
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
      // createdEpochMillis is populated at build() time; just assert non-zero.
      assertThat(decoded.createdEpochMillis()).isPositive();
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
        Manifest m = Manifest.build(config, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L);
        assertThat(Manifest.fromBytes(m.toBytes()).metric()).isEqualTo(metric);
      }
    }

    @Test
    void clockInjectableBuildPreservesExplicitTimestamp() throws IOException {
      VectorCollectionConfig config =
          new VectorCollectionConfig(
              32, SimilarityFunction.COSINE, IndexType.FLAT, QuantizerKind.NONE, Integer.MAX_VALUE);
      long fixedClock = 1_700_000_000_000L;
      Manifest m = Manifest.build(config, 7L, fixedClock, 0L, 0L, 0L, 0L, 0L, 0L, 0L);
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
      // Corrupt byte 0 of the magic.
      encoded[0] ^= (byte) 0xFF;
      assertThatIOException()
          .isThrownBy(() -> Manifest.fromBytes(encoded))
          .withMessageContaining("magic");
    }

    @Test
    void wrongVersionRejected() {
      byte[] encoded = sample().toBytes();
      // Corrupt the version word (offset 4-7).
      ByteBuffer buf = ByteBuffer.wrap(encoded).order(ByteOrder.LITTLE_ENDIAN);
      buf.putInt(4, 999);
      // Re-compute the self-CRC so the version error is what trips, not the CRC.
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
    void nonZeroFlagsRejected() {
      byte[] encoded = sample().toBytes();
      ByteBuffer buf = ByteBuffer.wrap(encoded).order(ByteOrder.LITTLE_ENDIAN);
      buf.putInt(12, 0x1);
      long selfCrc = Checksums.ofBytes(encoded, 0, Manifest.SELF_CRC_OFFSET);
      buf.putInt(Manifest.SELF_CRC_OFFSET, (int) selfCrc);
      assertThatIOException()
          .isThrownBy(() -> Manifest.fromBytes(encoded))
          .withMessageContaining("flags");
    }

    @Test
    void corruptPayloadTripsSelfCrc() {
      byte[] encoded = sample().toBytes();
      // Flip a byte in the dimension field (offset 16) without recomputing the CRC.
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
                      0L));
    }
  }
}
