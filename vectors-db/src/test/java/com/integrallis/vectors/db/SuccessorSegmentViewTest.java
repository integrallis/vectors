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
package com.integrallis.vectors.db;

import static org.junit.jupiter.api.Assertions.*;

import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.db.storage.MemorySegmentVectors;
import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class SuccessorSegmentViewTest {
  @TempDir Path directory;

  @Test
  void appendViewExposesIndependentReadOnlySegmentsForMappedAndStagedRows() throws Exception {
    float[] first = {1, 2, 3};
    float[] second = {4, 5, 6};
    Path file = directory.resolve("vectors.bin");
    new SuccessorVectors(null, 0, List.of(Document.of("first", first)), 3).writeTo(file);
    try (Arena arena = Arena.ofShared()) {
      var mapped = MemorySegmentVectors.open(file, 1, 3, arena);
      var view =
          new SuccessorVectors(mapped, 1, List.of(Document.of("second", second)), 3).asVectors();
      assertTrue(
          view.supportsSegments(), "append must use the existing segment batch-scoring path");
      var a = view.vectorSegment(0);
      var b = view.vectorSegment(1);
      assertEquals(12, a.byteSize());
      assertEquals(12, b.byteSize());
      assertTrue(a.isReadOnly());
      assertTrue(b.isReadOnly());
      for (int d = 0; d < 3; d++) {
        assertEquals(first[d], a.getAtIndex(ValueLayout.JAVA_FLOAT, d));
        assertEquals(second[d], b.getAtIndex(ValueLayout.JAVA_FLOAT, d));
      }
      assertThrows(IndexOutOfBoundsException.class, () -> view.vectorSegment(-1));
      assertThrows(IndexOutOfBoundsException.class, () -> view.vectorSegment(2));
    }
  }
}
