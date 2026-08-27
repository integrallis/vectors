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
package com.integrallis.vectors.core;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Base64;
import java.util.HexFormat;
import java.util.Map;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

class RotatedCodebookMatrixTest {

  private static final int FIXTURE_MAGIC = 0x43514658;
  private static final float REFERENCE_TOLERANCE = 1.0e-4f;
  private static final Map<String, String> FIXTURE_SHA256 =
      Map.of(
          "needle-cq2-v7bd8a63.b64",
          "122e46e23f4fd44d7ecc29a8ccfda69389e5bc42134863b448ecbbcddf03ef1b",
          "needle-cq3-v7bd8a63.b64",
          "e405f999513b34b19f4b1988f80ee38e5caf1f7be811b1ce030eb0b0ad954926",
          "needle-cq4-v7bd8a63.b64",
          "ec16ec3ee81abb0cf3e31176ad7bbf43f607d54b161ad148ec85f0c9212ff3b5",
          "needle-ternary-v7bd8a63.b64",
          "6b53746785cfbbfee07ab5caf9bbbd999e3641251cf9d02f6731925624ce6cb3");

  @ParameterizedTest(name = "{0}")
  @MethodSource("needleFixtures")
  void matchesFixturesGeneratedByNeedleReferenceCode(String name, String resource)
      throws IOException {
    Fixture fixture = readFixture(resource);
    RotatedCodebookMatrix matrix = fixture.matrix();
    float[] output = new float[fixture.rows()];

    matrix.multiply(matrix.prepare(fixture.input()), output);

    assertThat(output).containsExactly(fixture.expected(), within(REFERENCE_TOLERANCE));
  }

  @Test
  void preparedActivationCanBeReusedByCompatibleMatrices() throws IOException {
    Fixture fixture = readFixture("needle-cq2-v7bd8a63.b64");
    RotatedCodebookMatrix first = fixture.matrix();
    RotatedCodebookMatrix second = fixture.matrix();
    RotatedCodebookMatrix.PreparedActivation activation = first.prepare(fixture.input());
    float[] firstOutput = new float[fixture.rows()];
    float[] secondOutput = new float[fixture.rows()];

    first.multiply(activation, firstOutput);
    second.multiply(activation, secondOutput);

    assertThat(secondOutput).containsExactly(firstOutput);
    assertThat(first.accepts(activation)).isTrue();
    assertThat(second.accepts(activation)).isTrue();
  }

  @ParameterizedTest(name = "decode {0}")
  @MethodSource("needleFixtures")
  void decodedRowsReproduceNeedleReferenceDotProducts(String name, String resource)
      throws IOException {
    Fixture fixture = readFixture(resource);
    RotatedCodebookMatrix matrix = fixture.matrix();
    float[] row = new float[fixture.columns()];

    for (int index = 0; index < fixture.rows(); index++) {
      matrix.decodeRow(index, row);
      float dot = VectorUtil.dotProduct(row, fixture.input());
      assertThat(dot).isCloseTo(fixture.expected()[index], within(REFERENCE_TOLERANCE));
    }
  }

  @Test
  void rowSliceSharesStorageAndMatchesTheFullMatrix() throws IOException {
    Fixture fixture = readFixture("needle-cq3-v7bd8a63.b64");
    RotatedCodebookMatrix matrix = fixture.matrix();
    RotatedCodebookMatrix slice = matrix.rowSlice(1, 2);
    float[] complete = new float[fixture.rows()];
    float[] selected = new float[2];

    matrix.multiply(matrix.prepare(fixture.input()), complete);
    slice.multiply(slice.prepare(fixture.input()), selected);

    assertThat(slice.rows()).isEqualTo(2);
    assertThat(selected).containsExactly(complete[1], complete[2]);
    assertThatThrownBy(() -> matrix.rowSlice(-1, 1)).isInstanceOf(IndexOutOfBoundsException.class);
    assertThatThrownBy(() -> matrix.rowSlice(2, 2)).isInstanceOf(IndexOutOfBoundsException.class);
  }

  @Test
  void readsCodesAndNormsFromMappedFileSlices(@TempDir Path directory) throws IOException {
    Fixture fixture = readFixture("needle-cq4-v7bd8a63.b64");
    byte[] storage = new byte[fixture.packed().length + fixture.norms().length];
    System.arraycopy(fixture.packed(), 0, storage, 0, fixture.packed().length);
    System.arraycopy(fixture.norms(), 0, storage, fixture.packed().length, fixture.norms().length);
    Path path = directory.resolve("matrix.bin");
    Files.write(path, storage);

    try (Arena arena = Arena.ofConfined();
        FileChannel channel = FileChannel.open(path)) {
      MemorySegment mapped = channel.map(FileChannel.MapMode.READ_ONLY, 0, storage.length, arena);
      RotatedCodebookMatrix matrix =
          RotatedCodebookMatrix.of(
              mapped.asSlice(0, fixture.packed().length),
              mapped.asSlice(fixture.packed().length, fixture.norms().length),
              fixture.rows(),
              fixture.columns(),
              fixture.groupSize(),
              fixture.encoding(),
              fixture.codebook());
      float[] output = new float[fixture.rows()];

      matrix.multiply(matrix.prepare(fixture.input()), output);

      assertThat(output).containsExactly(fixture.expected(), within(REFERENCE_TOLERANCE));
    }
  }

  @Test
  void rejectsInvalidGeometryAndStorageBeforeInference() throws IOException {
    Fixture fixture = readFixture("needle-cq4-v7bd8a63.b64");

    assertThatThrownBy(
            () ->
                RotatedCodebookMatrix.of(
                    MemorySegment.ofArray(fixture.packed()),
                    MemorySegment.ofArray(fixture.norms()),
                    0,
                    fixture.columns(),
                    fixture.groupSize(),
                    fixture.encoding(),
                    fixture.codebook()))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("rows");
    assertThatThrownBy(
            () ->
                RotatedCodebookMatrix.of(
                    MemorySegment.ofArray(fixture.packed()),
                    MemorySegment.ofArray(fixture.norms()),
                    fixture.rows(),
                    fixture.columns(),
                    96,
                    fixture.encoding(),
                    fixture.codebook()))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("power of two");
    assertThatThrownBy(
            () ->
                RotatedCodebookMatrix.of(
                    MemorySegment.ofArray(new byte[fixture.packed().length - 1]),
                    MemorySegment.ofArray(fixture.norms()),
                    fixture.rows(),
                    fixture.columns(),
                    fixture.groupSize(),
                    fixture.encoding(),
                    fixture.codebook()))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("packed codes");
    assertThatThrownBy(
            () ->
                RotatedCodebookMatrix.of(
                    MemorySegment.ofArray(fixture.packed()),
                    MemorySegment.ofArray(new byte[fixture.norms().length - 1]),
                    fixture.rows(),
                    fixture.columns(),
                    fixture.groupSize(),
                    fixture.encoding(),
                    fixture.codebook()))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("norms");
    assertThatThrownBy(
            () ->
                RotatedCodebookMatrix.of(
                    MemorySegment.ofArray(fixture.packed()),
                    MemorySegment.ofArray(fixture.norms()),
                    fixture.rows(),
                    fixture.columns(),
                    fixture.groupSize(),
                    fixture.encoding(),
                    new float[15]))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("codebook");
  }

  @Test
  void rejectsIncompatibleInputsOutputsAndPreparedActivations() throws IOException {
    Fixture cq2 = readFixture("needle-cq2-v7bd8a63.b64");
    Fixture cq4 = readFixture("needle-cq4-v7bd8a63.b64");
    RotatedCodebookMatrix cq2Matrix = cq2.matrix();
    RotatedCodebookMatrix cq4Matrix = cq4.matrix();
    RotatedCodebookMatrix.PreparedActivation cq2Activation = cq2Matrix.prepare(cq2.input());

    assertThatThrownBy(() -> cq2Matrix.prepare(new float[cq2.columns() - 1]))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("input");
    assertThatThrownBy(
            () -> cq2Matrix.multiply(cq2Matrix.prepare(cq2.input()), new float[cq2.rows() - 1]))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("output");
    assertThatThrownBy(() -> cq4Matrix.multiply(cq2Activation, new float[cq4.rows()]))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("incompatible");
    assertThat(cq4Matrix.accepts(cq2Activation)).isFalse();
  }

  @Test
  void rejectsInvalidTernaryCrumbsWhenRead() throws IOException {
    Fixture fixture = readFixture("needle-ternary-v7bd8a63.b64");
    byte[] invalid = fixture.packed().clone();
    invalid[0] = (byte) ((invalid[0] & 0xFC) | 0x02);
    RotatedCodebookMatrix matrix =
        RotatedCodebookMatrix.of(
            MemorySegment.ofArray(invalid),
            MemorySegment.ofArray(fixture.norms()),
            fixture.rows(),
            fixture.columns(),
            fixture.groupSize(),
            fixture.encoding(),
            fixture.codebook());

    assertThatThrownBy(
            () -> matrix.multiply(matrix.prepare(fixture.input()), new float[fixture.rows()]))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("ternary crumb");
  }

  private static Stream<Arguments> needleFixtures() {
    return Stream.of(
        Arguments.of("CQ2", "needle-cq2-v7bd8a63.b64"),
        Arguments.of("CQ3", "needle-cq3-v7bd8a63.b64"),
        Arguments.of("CQ4", "needle-cq4-v7bd8a63.b64"),
        Arguments.of("ternary", "needle-ternary-v7bd8a63.b64"));
  }

  private static Fixture readFixture(String name) throws IOException {
    try (var stream = RotatedCodebookMatrixTest.class.getResourceAsStream(name)) {
      if (stream == null) {
        throw new IOException("Missing reference fixture " + name);
      }
      byte[] bytes = Base64.getMimeDecoder().decode(stream.readAllBytes());
      String checksum = HexFormat.of().formatHex(sha256(bytes));
      if (!checksum.equals(FIXTURE_SHA256.get(name))) {
        throw new IOException("Reference fixture checksum changed for " + name);
      }
      ByteBuffer data = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
      if (data.getInt() != FIXTURE_MAGIC || data.getInt() != 1) {
        throw new IOException("Invalid reference fixture " + name);
      }
      int rows = data.getInt();
      int columns = data.getInt();
      int groupSize = data.getInt();
      int recordBits = data.getInt();
      int codebookLength = data.getInt();
      int packedLength = data.getInt();
      int normCount = data.getInt();
      float[] codebook = new float[codebookLength];
      for (int index = 0; index < codebook.length; index++) {
        codebook[index] = data.getFloat();
      }
      byte[] packed = new byte[packedLength];
      data.get(packed);
      byte[] norms = new byte[Math.multiplyExact(normCount, Short.BYTES)];
      data.get(norms);
      float[] input = new float[columns];
      for (int index = 0; index < input.length; index++) {
        input[index] = data.getFloat();
      }
      float[] expected = new float[rows];
      for (int index = 0; index < expected.length; index++) {
        expected[index] = data.getFloat();
      }
      if (data.hasRemaining()) {
        throw new IOException("Trailing bytes in reference fixture " + name);
      }
      return new Fixture(
          rows, columns, groupSize, encoding(recordBits), codebook, packed, norms, input, expected);
    }
  }

  private static byte[] sha256(byte[] bytes) throws IOException {
    try {
      return MessageDigest.getInstance("SHA-256").digest(bytes);
    } catch (NoSuchAlgorithmException exception) {
      throw new IOException("SHA-256 is unavailable", exception);
    }
  }

  private static RotatedCodebookMatrix.Encoding encoding(int recordBits) throws IOException {
    return switch (recordBits) {
      case 2 -> RotatedCodebookMatrix.Encoding.CQ2;
      case 3 -> RotatedCodebookMatrix.Encoding.CQ3;
      case 4 -> RotatedCodebookMatrix.Encoding.CQ4;
      case 5 -> RotatedCodebookMatrix.Encoding.TERNARY;
      default -> throw new IOException("Unsupported reference record bits " + recordBits);
    };
  }

  private record Fixture(
      int rows,
      int columns,
      int groupSize,
      RotatedCodebookMatrix.Encoding encoding,
      float[] codebook,
      byte[] packed,
      byte[] norms,
      float[] input,
      float[] expected) {

    private RotatedCodebookMatrix matrix() {
      return RotatedCodebookMatrix.of(
          MemorySegment.ofArray(packed),
          MemorySegment.ofArray(norms),
          rows,
          columns,
          groupSize,
          encoding,
          codebook);
    }
  }
}
