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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.integrallis.vectors.db.storage.FileFormat;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Pins the assumption {@code StoredRecipeAnchor} depends on.
 *
 * <p>It selects the newest generation by <b>lexicographic</b> directory name, which is only the
 * newest generation because {@link FileFormat#generationDirName(long)} zero-pads to a fixed width.
 * If that padding were ever dropped or narrowed, {@code gen-9} would sort after {@code gen-10} and
 * the anchor would be read from an older manifest — silently, producing a stale or absent recipe
 * rather than an error.
 *
 * <p>This test exists because that assumption lives in one class and the naming it depends on lives
 * in another. It fails loudly if they drift apart.
 */
class StoredRecipeAnchorOrderingTest {

  @Test
  @DisplayName("generation names are fixed-width, so lexicographic order is numeric order")
  void lexicographicOrderMatchesNumericOrder() {
    long[] generations = {
      0L, 1L, 2L, 9L, 10L, 11L, 99L, 100L, 1_000L, 999_999L, FileFormat.MAX_GENERATION_NUMBER
    };
    String previous = null;
    for (long generation : generations) {
      String name = FileFormat.generationDirName(generation);
      if (previous != null) {
        assertTrue(
            previous.compareTo(name) < 0,
            "lexicographic order broke between "
                + previous
                + " and "
                + name
                + ": StoredRecipeAnchor would then read the wrong manifest");
      }
      previous = name;
    }
  }

  @Test
  @DisplayName("a generation number that would break the sort is refused, not formatted")
  void oversizedGenerationIsRefused() {
    // Found by writing this test: FileFormat already guards the invariant rather than trusting it.
    // A 17-digit name would sort BEFORE every 16-digit one, so the guard is what actually protects
    // StoredRecipeAnchor -- the padding alone would not.
    assertThrows(
        IllegalArgumentException.class,
        () -> FileFormat.generationDirName(FileFormat.MAX_GENERATION_NUMBER + 1));
    assertThrows(IllegalArgumentException.class, () -> FileFormat.generationDirName(-1L));
  }

  @Test
  @DisplayName("the padding width is pinned, because the ordering depends on it")
  void paddingWidthIsPinned() {
    // 16 digits holds Long.MAX_VALUE (19 digits) only partially -- the format widens rather than
    // truncates beyond that, which is why the test above includes MAX_VALUE explicitly.
    assertEquals(
        FileFormat.GENERATION_DIR_PREFIX + "0000000000000000", FileFormat.generationDirName(0L));
    assertEquals(
        FileFormat.GENERATION_DIR_PREFIX + "0000000000000001", FileFormat.generationDirName(1L));
    assertEquals(
        FileFormat.GENERATION_DIR_PREFIX + "0000000000000042", FileFormat.generationDirName(42L));
    assertEquals(
        16,
        FileFormat.generationDirName(7L).length() - FileFormat.GENERATION_DIR_PREFIX.length(),
        "a narrower field would make gen-9 sort after gen-10");
  }

  @Test
  @DisplayName("the single-digit and double-digit boundary is the one that would break first")
  void theBoundaryThatWouldBreakFirst() {
    // Without padding this is the smallest failing case: "gen-9" > "gen-10" lexicographically.
    assertTrue(
        FileFormat.generationDirName(9L).compareTo(FileFormat.generationDirName(10L)) < 0,
        "gen-9 must sort before gen-10");
    assertTrue(
        "gen-9".compareTo("gen-10") > 0,
        "this documents WHY the padding is required: unpadded names sort wrongly");
  }
}
