package com.integrallis.vectors.core;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

/** Tests for {@link VectorEncoding}. */
class VectorEncodingTest {

  @Test
  void float32_bytesPerDimension() {
    assertThat(VectorEncoding.FLOAT32.bytesPerDimension()).isEqualTo(4);
  }

  @Test
  void int8_bytesPerDimension() {
    assertThat(VectorEncoding.INT8.bytesPerDimension()).isEqualTo(1);
  }

  @Test
  void binary_bytesPerDimension() {
    assertThat(VectorEncoding.BINARY.bytesPerDimension()).isZero();
  }

  @Test
  void float32_vectorByteSize() {
    assertThat(VectorEncoding.FLOAT32.vectorByteSize(128)).isEqualTo(512);
    assertThat(VectorEncoding.FLOAT32.vectorByteSize(3)).isEqualTo(12);
    assertThat(VectorEncoding.FLOAT32.vectorByteSize(1536)).isEqualTo(6144);
  }

  @Test
  void int8_vectorByteSize() {
    assertThat(VectorEncoding.INT8.vectorByteSize(128)).isEqualTo(128);
    assertThat(VectorEncoding.INT8.vectorByteSize(3)).isEqualTo(3);
  }

  @Test
  void binary_vectorByteSize() {
    // 64 bits = 1 long = 8 bytes
    assertThat(VectorEncoding.BINARY.vectorByteSize(64)).isEqualTo(8);
    // 65 bits = 2 longs = 16 bytes
    assertThat(VectorEncoding.BINARY.vectorByteSize(65)).isEqualTo(16);
    // 1 bit = 1 long = 8 bytes (minimum)
    assertThat(VectorEncoding.BINARY.vectorByteSize(1)).isEqualTo(8);
    // 128 bits = 2 longs = 16 bytes
    assertThat(VectorEncoding.BINARY.vectorByteSize(128)).isEqualTo(16);
  }
}
