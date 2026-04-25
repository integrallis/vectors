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
package com.integrallis.vectors.storage;

import static org.assertj.core.api.Assertions.assertThat;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Tests for {@link StorageLayouts}. */
@Tag("unit")
class StorageLayoutsTest {

  @Test
  void layouts_areLittleEndian() {
    assertThat(StorageLayouts.FLOAT_LE.order()).isEqualTo(ByteOrder.LITTLE_ENDIAN);
    assertThat(StorageLayouts.INT_LE.order()).isEqualTo(ByteOrder.LITTLE_ENDIAN);
    assertThat(StorageLayouts.LONG_LE.order()).isEqualTo(ByteOrder.LITTLE_ENDIAN);
    assertThat(StorageLayouts.SHORT_LE.order()).isEqualTo(ByteOrder.LITTLE_ENDIAN);
  }

  @Test
  void float_roundTrip() {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment seg = arena.allocate(4);
      seg.set(StorageLayouts.FLOAT_LE, 0, 3.14f);
      assertThat(seg.get(StorageLayouts.FLOAT_LE, 0)).isEqualTo(3.14f);
    }
  }

  @Test
  void int_roundTrip() {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment seg = arena.allocate(4);
      seg.set(StorageLayouts.INT_LE, 0, 42);
      assertThat(seg.get(StorageLayouts.INT_LE, 0)).isEqualTo(42);
    }
  }

  @Test
  void long_roundTrip() {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment seg = arena.allocate(8);
      seg.set(StorageLayouts.LONG_LE, 0, Long.MAX_VALUE);
      assertThat(seg.get(StorageLayouts.LONG_LE, 0)).isEqualTo(Long.MAX_VALUE);
    }
  }

  @Test
  void short_roundTrip() {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment seg = arena.allocate(2);
      seg.set(StorageLayouts.SHORT_LE, 0, (short) 1024);
      assertThat(seg.get(StorageLayouts.SHORT_LE, 0)).isEqualTo((short) 1024);
    }
  }

  @Test
  void byte_roundTrip() {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment seg = arena.allocate(1);
      seg.set(StorageLayouts.BYTE, 0, (byte) 0x7F);
      assertThat(seg.get(StorageLayouts.BYTE, 0)).isEqualTo((byte) 0x7F);
    }
  }
}
