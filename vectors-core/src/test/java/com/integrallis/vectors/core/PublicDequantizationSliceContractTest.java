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

import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.stream.Stream;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/** Public decode boundaries reject invalid slices before touching caller-owned output. */
class PublicDequantizationSliceContractTest {
  @FunctionalInterface
  interface Decode {
    void apply(MemorySegment weights, long offset, float[] out, int start, int length);
  }

  static Stream<Arguments> requests() {
    Stream.Builder<Arguments> result = Stream.builder();
    Decode[] decoders = {
      VectorUtil::ggufQ4_KDequantize,
      VectorUtil::ggufQ5_KDequantize,
      VectorUtil::ggufQ5_0Dequantize,
      VectorUtil::ggufQ6_KDequantize
    };
    String[] names = {"Q4_K", "Q5_K", "Q5_0", "Q6_K"};
    int[] block = {256, 256, 32, 256}, bytes = {144, 176, 22, 210};
    for (int f = 0; f < 4; f++)
      for (String failure :
          new String[] {
            "negative-weight-offset", "overflow-weight-end", "short-weight",
                "negative-output-offset",
            "negative-length", "output-end", "unaligned-length", "zero-at-end"
          }) result.add(Arguments.of(names[f], decoders[f], block[f], bytes[f], failure));
    return result.build();
  }

  @ParameterizedTest(name = "{0}: {4}")
  @MethodSource("requests")
  void invalidOrEmptyRangesPreserveBothBuffers(
      String format, Decode decode, int block, int blockBytes, String request) {
    byte[] packed = new byte[blockBytes + 7];
    Arrays.fill(packed, (byte) 0x5a);
    MemorySegment weights = MemorySegment.ofArray(packed).asReadOnly();
    float[] out = new float[block + 4];
    Arrays.fill(out, 12345.25f);
    long offset = 7;
    int start = 2, length = block;
    switch (request) {
      case "negative-weight-offset" -> offset = -1;
      case "overflow-weight-end" -> offset = Long.MAX_VALUE - 1;
      case "short-weight" -> offset = 8;
      case "negative-output-offset" -> start = -1;
      case "negative-length" -> length = -1;
      case "output-end" -> start = out.length - block + 1;
      case "unaligned-length" -> length = block - 1;
      case "zero-at-end" -> {
        offset = packed.length;
        start = out.length;
        length = 0;
      }
      default -> throw new AssertionError(request);
    }
    long byteOffset = offset;
    int outOffset = start, count = length;
    byte[] before = packed.clone();
    if (request.equals("zero-at-end")) decode.apply(weights, byteOffset, out, outOffset, count);
    else
      assertThatThrownBy(() -> decode.apply(weights, byteOffset, out, outOffset, count))
          .isInstanceOf(IllegalArgumentException.class);
    assertThat(out).containsOnly(12345.25f);
    assertThat(packed).containsExactly(before);
  }
}
