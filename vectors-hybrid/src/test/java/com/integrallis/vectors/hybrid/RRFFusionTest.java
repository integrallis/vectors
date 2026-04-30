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
package com.integrallis.vectors.hybrid;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class RRFFusionTest {

  @Test
  void rankBasedScoring() {
    var fusion = new RRFFusion();
    // doc-a is rank 1 in both lists, doc-b rank 2 in list 1, doc-c rank 2 in list 2
    List<ScoredId> list1 = List.of(new ScoredId("doc-a", 0.9f), new ScoredId("doc-b", 0.8f));
    List<ScoredId> list2 = List.of(new ScoredId("doc-a", 0.7f), new ScoredId("doc-c", 0.6f));

    List<ScoredId> fused = fusion.fuse(List.of(list1, list2), 3);

    assertThat(fused).hasSize(3);
    // doc-a appears in both lists at rank 1: score = 2 * 1/(60+1) = 2/61
    assertThat(fused.get(0).id()).isEqualTo("doc-a");
    assertThat(fused.get(0).score())
        .isCloseTo(2.0f / 61, org.assertj.core.data.Offset.offset(1e-6f));
    // doc-b and doc-c each appear once at rank 2: score = 1/(60+2) = 1/62
    assertThat(fused.get(1).score())
        .isCloseTo(1.0f / 62, org.assertj.core.data.Offset.offset(1e-6f));
  }

  @Test
  void disjointResultSets() {
    var fusion = new RRFFusion();
    List<ScoredId> list1 = List.of(new ScoredId("a", 1.0f), new ScoredId("b", 0.5f));
    List<ScoredId> list2 = List.of(new ScoredId("c", 1.0f), new ScoredId("d", 0.5f));

    List<ScoredId> fused = fusion.fuse(List.of(list1, list2), 4);

    assertThat(fused).hasSize(4);
    // All 4 docs have equal RRF scores at their respective ranks; a, c at rank 1, b, d at rank 2
    assertThat(fused.get(0).score())
        .isCloseTo(1.0f / 61, org.assertj.core.data.Offset.offset(1e-6f));
    assertThat(fused.get(2).score())
        .isCloseTo(1.0f / 62, org.assertj.core.data.Offset.offset(1e-6f));
  }

  @Test
  void kLimitsResults() {
    var fusion = new RRFFusion();
    List<ScoredId> list1 =
        List.of(new ScoredId("a", 1.0f), new ScoredId("b", 0.5f), new ScoredId("c", 0.3f));

    List<ScoredId> fused = fusion.fuse(List.of(list1), 2);

    assertThat(fused).hasSize(2);
  }

  @Test
  void kZeroReturnsEmpty() {
    var fusion = new RRFFusion();
    List<ScoredId> fused = fusion.fuse(List.of(List.of(new ScoredId("a", 1.0f))), 0);
    assertThat(fused).isEmpty();
  }

  @Test
  void emptyResultLists() {
    var fusion = new RRFFusion();
    List<ScoredId> fused = fusion.fuse(List.of(List.of(), List.of()), 5);
    assertThat(fused).isEmpty();
  }

  @Test
  void invalidSmoothingKThrows() {
    assertThatThrownBy(() -> new RRFFusion(0)).isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> new RRFFusion(-1)).isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void tieBreakingByIdForDeterminism() {
    var fusion = new RRFFusion();
    // Both docs appear once at rank 1 — same RRF score, tie-break by ID
    List<ScoredId> list1 = List.of(new ScoredId("z", 1.0f));
    List<ScoredId> list2 = List.of(new ScoredId("a", 1.0f));

    List<ScoredId> fused = fusion.fuse(List.of(list1, list2), 2);

    assertThat(fused).hasSize(2);
    assertThat(fused.get(0).id()).isEqualTo("a");
    assertThat(fused.get(1).id()).isEqualTo("z");
  }
}
