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

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Tests for {@link VectorEncoding}. */
@Tag("unit")
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
