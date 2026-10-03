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

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.db.storage.MemorySegmentVectors;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class SuccessorVectorOwnershipTest {
  @Test
  void stagedRowsAreStableWhileCarriedRowsReuseScratch(@TempDir Path directory) throws Exception {
    float[] a = {1, 2, 3}, b = {4, 5, 6}, c = {7, 8, 9};
    var staged =
        new SuccessorVectors(null, 0, List.of(Document.of("a", a), Document.of("b", b)), 3);
    Path file = directory.resolve("vectors.bin");
    staged.writeTo(file);
    try (Arena arena = Arena.ofConfined()) {
      var carried = MemorySegmentVectors.open(file, 2, 3, arena);
      var mixed = new SuccessorVectors(carried, 2, List.of(Document.of("c", c)), 3).asVectors();
      assertThat(mixed.sharesReturnBuffer()).isTrue();
      assertThat(mixed.sharesReturnBuffer(0)).isTrue();
      assertThat(mixed.sharesReturnBuffer(1)).isTrue();
      assertThat(mixed.sharesReturnBuffer(2)).isFalse();
      float[] stable = mixed.getVector(2);
      float[] scratch = mixed.getVector(0);
      assertThat(mixed.getVector(1)).isSameAs(scratch).containsExactly(b);
      assertThat(stable).containsExactly(c);
    }
    assertThat(staged.asVectors().sharesReturnBuffer(0)).isFalse();
  }
}
