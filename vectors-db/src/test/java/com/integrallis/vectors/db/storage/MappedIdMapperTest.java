package com.integrallis.vectors.db.storage;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatExceptionOfType;
import static org.assertj.core.api.Assertions.assertThatIOException;

import com.integrallis.vectors.db.id.IdMapper;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class MappedIdMapperTest {

  private static List<String> sequentialIds(int n) {
    List<String> ids = new ArrayList<>(n);
    for (int i = 0; i < n; i++) {
      ids.add("doc-" + i);
    }
    return ids;
  }

  @Nested
  class RoundTrip {

    @Test
    void emptyMap(@TempDir Path tmp) throws IOException {
      Path file = tmp.resolve("idmap.bin");
      MappedIdMapper.Writer.writeTo(file, List.of());
      try (Arena arena = Arena.ofConfined()) {
        MappedIdMapper m = MappedIdMapper.open(file, arena);
        assertThat(m.size()).isEqualTo(0);
        assertThat(m.contains("anything")).isFalse();
        assertThat(m.ordinalOf("anything")).isEqualTo(-1);
      }
    }

    @Test
    void singleId(@TempDir Path tmp) throws IOException {
      Path file = tmp.resolve("idmap.bin");
      MappedIdMapper.Writer.writeTo(file, List.of("the-only-id"));
      try (Arena arena = Arena.ofConfined()) {
        MappedIdMapper m = MappedIdMapper.open(file, arena);
        assertThat(m.size()).isEqualTo(1);
        assertThat(m.contains("the-only-id")).isTrue();
        assertThat(m.ordinalOf("the-only-id")).isEqualTo(0);
        assertThat(m.idOf(0)).isEqualTo("the-only-id");
      }
    }

    @Test
    void oneHundredIds(@TempDir Path tmp) throws IOException {
      Path file = tmp.resolve("idmap.bin");
      List<String> ids = sequentialIds(100);
      MappedIdMapper.Writer.writeTo(file, ids);
      try (Arena arena = Arena.ofConfined()) {
        MappedIdMapper m = MappedIdMapper.open(file, arena);
        assertThat(m.size()).isEqualTo(100);
        for (int i = 0; i < 100; i++) {
          assertThat(m.idOf(i)).isEqualTo(ids.get(i));
          assertThat(m.ordinalOf(ids.get(i))).isEqualTo(i);
          assertThat(m.contains(ids.get(i))).isTrue();
        }
      }
    }

    @Test
    void tenThousandIds(@TempDir Path tmp) throws IOException {
      Path file = tmp.resolve("idmap.bin");
      List<String> ids = sequentialIds(10_000);
      MappedIdMapper.Writer.writeTo(file, ids);
      try (Arena arena = Arena.ofConfined()) {
        MappedIdMapper m = MappedIdMapper.open(file, arena);
        assertThat(m.size()).isEqualTo(10_000);
        // Spot-check every 97th ordinal to keep the test fast while still exercising the heap.
        for (int i = 0; i < 10_000; i += 97) {
          assertThat(m.idOf(i)).isEqualTo(ids.get(i));
          assertThat(m.ordinalOf(ids.get(i))).isEqualTo(i);
        }
        // Unknown id.
        assertThat(m.ordinalOf("not-present")).isEqualTo(-1);
      }
    }

    @Test
    void variableLengthUtf8Ids(@TempDir Path tmp) throws IOException {
      Path file = tmp.resolve("idmap.bin");
      List<String> ids =
          List.of(
              "a",
              "ab",
              "xyz",
              // 127-byte ascii — max single-byte varint length
              "x".repeat(127),
              // 128-byte ascii — varint flips to 2 bytes
              "y".repeat(128),
              // multi-byte UTF-8
              "日本語のID",
              // emoji (4-byte UTF-8 code points)
              "id-\uD83D\uDE80\uD83C\uDF89");
      MappedIdMapper.Writer.writeTo(file, ids);
      try (Arena arena = Arena.ofConfined()) {
        MappedIdMapper m = MappedIdMapper.open(file, arena);
        assertThat(m.size()).isEqualTo(ids.size());
        for (int i = 0; i < ids.size(); i++) {
          assertThat(m.idOf(i)).isEqualTo(ids.get(i));
          assertThat(m.ordinalOf(ids.get(i))).isEqualTo(i);
        }
      }
    }
  }

  @Nested
  class Validation {

    @Test
    void missingFileThrows(@TempDir Path tmp) {
      Path file = tmp.resolve("does-not-exist.bin");
      try (Arena arena = Arena.ofConfined()) {
        assertThatIOException().isThrownBy(() -> MappedIdMapper.open(file, arena));
      }
    }

    @Test
    void truncatedHeaderThrows(@TempDir Path tmp) throws IOException {
      Path file = tmp.resolve("idmap.bin");
      Files.write(file, new byte[8]); // shorter than the 16-byte header
      try (Arena arena = Arena.ofConfined()) {
        assertThatIOException()
            .isThrownBy(() -> MappedIdMapper.open(file, arena))
            .withMessageContaining("truncated");
      }
    }

    @Test
    void wrongMagicThrows(@TempDir Path tmp) throws IOException {
      Path file = tmp.resolve("idmap.bin");
      MappedIdMapper.Writer.writeTo(file, List.of("a", "b"));
      byte[] bytes = Files.readAllBytes(file);
      // Corrupt the magic word.
      bytes[0] ^= (byte) 0xFF;
      Files.write(file, bytes);
      try (Arena arena = Arena.ofConfined()) {
        assertThatIOException()
            .isThrownBy(() -> MappedIdMapper.open(file, arena))
            .withMessageContaining("magic");
      }
    }

    @Test
    void wrongVersionThrows(@TempDir Path tmp) throws IOException {
      Path file = tmp.resolve("idmap.bin");
      MappedIdMapper.Writer.writeTo(file, List.of("a", "b"));
      byte[] bytes = Files.readAllBytes(file);
      ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).putInt(4, 999);
      Files.write(file, bytes);
      try (Arena arena = Arena.ofConfined()) {
        assertThatIOException()
            .isThrownBy(() -> MappedIdMapper.open(file, arena))
            .withMessageContaining("version");
      }
    }

    @Test
    void negativeCountThrows(@TempDir Path tmp) throws IOException {
      Path file = tmp.resolve("idmap.bin");
      MappedIdMapper.Writer.writeTo(file, List.of("a"));
      byte[] bytes = Files.readAllBytes(file);
      ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).putInt(8, -1);
      Files.write(file, bytes);
      try (Arena arena = Arena.ofConfined()) {
        assertThatIOException()
            .isThrownBy(() -> MappedIdMapper.open(file, arena))
            .withMessageContaining("count");
      }
    }

    @Test
    void declaredSizeLargerThanFileThrows(@TempDir Path tmp) throws IOException {
      Path file = tmp.resolve("idmap.bin");
      MappedIdMapper.Writer.writeTo(file, List.of("a", "b"));
      byte[] bytes = Files.readAllBytes(file);
      // Pump heap byte length up so expected > file size.
      ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).putInt(12, 1_000_000);
      Files.write(file, bytes);
      try (Arena arena = Arena.ofConfined()) {
        assertThatIOException()
            .isThrownBy(() -> MappedIdMapper.open(file, arena))
            .withMessageContaining("truncated");
      }
    }

    @Test
    void nullIdInWriterThrows(@TempDir Path tmp) {
      Path file = tmp.resolve("idmap.bin");
      List<String> idsWithNull = new ArrayList<>();
      idsWithNull.add("a");
      idsWithNull.add(null);
      assertThatIOException()
          .isThrownBy(() -> MappedIdMapper.Writer.writeTo(file, idsWithNull))
          .withMessageContaining("null");
    }
  }

  @Nested
  class ReadOnly {

    @Test
    void putThrowsUnsupported(@TempDir Path tmp) throws IOException {
      Path file = tmp.resolve("idmap.bin");
      MappedIdMapper.Writer.writeTo(file, List.of("x"));
      try (Arena arena = Arena.ofConfined()) {
        IdMapper m = MappedIdMapper.open(file, arena);
        assertThatExceptionOfType(UnsupportedOperationException.class).isThrownBy(() -> m.put("y"));
      }
    }

    @Test
    void idOfNegativeOrdinalThrows(@TempDir Path tmp) throws IOException {
      Path file = tmp.resolve("idmap.bin");
      MappedIdMapper.Writer.writeTo(file, List.of("x"));
      try (Arena arena = Arena.ofConfined()) {
        MappedIdMapper m = MappedIdMapper.open(file, arena);
        assertThatExceptionOfType(IndexOutOfBoundsException.class).isThrownBy(() -> m.idOf(-1));
        assertThatExceptionOfType(IndexOutOfBoundsException.class).isThrownBy(() -> m.idOf(1));
      }
    }
  }

  @Nested
  class Reopen {

    @Test
    void reopenedMapperSeesSameIds(@TempDir Path tmp) throws IOException {
      Path file = tmp.resolve("idmap.bin");
      List<String> ids = sequentialIds(500);
      MappedIdMapper.Writer.writeTo(file, ids);

      // First open: read every id.
      try (Arena a1 = Arena.ofConfined()) {
        MappedIdMapper m1 = MappedIdMapper.open(file, a1);
        for (int i = 0; i < ids.size(); i++) {
          assertThat(m1.idOf(i)).isEqualTo(ids.get(i));
        }
      }

      // Second independent open: same file, same answers.
      try (Arena a2 = Arena.ofConfined()) {
        MappedIdMapper m2 = MappedIdMapper.open(file, a2);
        assertThat(m2.size()).isEqualTo(ids.size());
        for (int i = 0; i < ids.size(); i++) {
          assertThat(m2.idOf(i)).isEqualTo(ids.get(i));
          assertThat(m2.ordinalOf(ids.get(i))).isEqualTo(i);
        }
      }
    }
  }
}
