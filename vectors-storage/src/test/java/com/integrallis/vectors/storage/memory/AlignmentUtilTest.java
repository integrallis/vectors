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
package com.integrallis.vectors.storage.memory;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/** Tests for {@link AlignmentUtil}. */
@Tag("unit")
class AlignmentUtilTest {

  @Test
  void pageSize_isPositivePowerOfTwo() {
    assertThat(AlignmentUtil.PAGE_SIZE).isGreaterThan(0);
    assertThat(AlignmentUtil.PAGE_SIZE & (AlignmentUtil.PAGE_SIZE - 1)).isZero();
  }

  @Test
  void vectorAlignment_is64() {
    assertThat(AlignmentUtil.VECTOR_ALIGNMENT).isEqualTo(64);
  }

  @ParameterizedTest
  @CsvSource({
    "0, 64, 0",
    "1, 64, 64",
    "63, 64, 64",
    "64, 64, 64",
    "65, 64, 128",
    "127, 64, 128",
    "128, 64, 128",
    "0, 4096, 0",
    "1, 4096, 4096",
    "4095, 4096, 4096",
    "4096, 4096, 4096",
    "4097, 4096, 8192",
    "512, 4, 512",
    "513, 4, 516",
  })
  void alignUp_correctResults(long value, int alignment, long expected) {
    assertThat(AlignmentUtil.alignUp(value, alignment)).isEqualTo(expected);
  }

  @Test
  void alignUp_nonPowerOfTwo_throws() {
    assertThatThrownBy(() -> AlignmentUtil.alignUp(100, 3))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void alignUp_zeroAlignment_throws() {
    assertThatThrownBy(() -> AlignmentUtil.alignUp(100, 0))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void alignUp_negativeAlignment_throws() {
    assertThatThrownBy(() -> AlignmentUtil.alignUp(100, -1))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @ParameterizedTest
  @CsvSource({
    "0, 64, 0",
    "1, 64, 63",
    "63, 64, 1",
    "64, 64, 0",
    "65, 64, 63",
    "100, 4, 0",
    "101, 4, 3",
  })
  void paddingFor_correctResults(long offset, int alignment, int expected) {
    assertThat(AlignmentUtil.paddingFor(offset, alignment)).isEqualTo(expected);
  }

  @ParameterizedTest
  @CsvSource({
    "0, 64, true",
    "64, 64, true",
    "128, 64, true",
    "63, 64, false",
    "65, 64, false",
    "4096, 4096, true",
    "4097, 4096, false",
  })
  void isAligned_correctResults(long value, int alignment, boolean expected) {
    assertThat(AlignmentUtil.isAligned(value, alignment)).isEqualTo(expected);
  }
}
