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
package com.integrallis.vectors.studio.distributed.search;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import com.integrallis.vectors.studio.core.search.SearchHit;
import com.integrallis.vectors.studio.sidecart.TextSearchHit;
import java.util.List;
import java.util.function.Function;
import org.junit.jupiter.api.Test;

class RrfFuserTest {

  private static SearchHit dense(String id) {
    return new SearchHit(id, 0.9, null, null, null);
  }

  private static TextSearchHit sparse(String id) {
    return new TextSearchHit(id, 0.5);
  }

  private static final Function<String, SearchHit> NULL_HYDRATOR =
      id -> new SearchHit(id, 0d, null, null, null);

  @Test
  void overlappingIdRankedFirst() {
    List<SearchHit> dense = List.of(dense("a"), dense("b"), dense("c"));
    List<TextSearchHit> sparse = List.of(sparse("c"), sparse("a"), sparse("d"));
    List<SearchHit> fused = RrfFuser.fuse(dense, sparse, 4, 60, NULL_HYDRATOR);
    // Both 'a' and 'c' appear in both lists. 'a' is rank 0 dense + rank 1 sparse;
    // 'c' is rank 2 dense + rank 0 sparse. Their RRF aggregates are equal.
    // 'b' (only dense, rank 1) and 'd' (only sparse, rank 2) follow.
    assertThat(fused).extracting(SearchHit::id).startsWith("a", "c");
    assertThat(fused).hasSize(4);
  }

  @Test
  void rrfFormulaMatchesPaperDefinition() {
    List<SearchHit> dense = List.of(dense("x"));
    List<TextSearchHit> sparse = List.of(sparse("x"));
    List<SearchHit> fused = RrfFuser.fuse(dense, sparse, 1, 60, NULL_HYDRATOR);
    // x is rank 0 in both => 1/(60+1) + 1/(60+1) = 2/61.
    assertThat(fused).hasSize(1);
    assertThat(fused.get(0).score()).isCloseTo(2.0 / 61.0, within(1e-12));
  }

  @Test
  void uniqueSparseIdsAreHydrated() {
    List<SearchHit> dense = List.of(dense("a"));
    List<TextSearchHit> sparse = List.of(sparse("b"));
    Function<String, SearchHit> hydrator =
        id -> new SearchHit(id, 0d, new float[] {1f}, "txt-" + id, null);
    List<SearchHit> fused = RrfFuser.fuse(dense, sparse, 5, 60, hydrator);
    SearchHit b = fused.stream().filter(h -> "b".equals(h.id())).findFirst().orElseThrow();
    assertThat(b.text()).isEqualTo("txt-b");
    assertThat(b.vector()).containsExactly(1f);
  }

  @Test
  void emptySparseLeavesDenseRankingIntact() {
    List<SearchHit> dense = List.of(dense("a"), dense("b"), dense("c"));
    List<SearchHit> fused = RrfFuser.fuse(dense, List.of(), 3, 60, NULL_HYDRATOR);
    assertThat(fused).extracting(SearchHit::id).containsExactly("a", "b", "c");
  }

  @Test
  void emptyDenseUsesSparseOrdering() {
    List<TextSearchHit> sparse = List.of(sparse("x"), sparse("y"), sparse("z"));
    List<SearchHit> fused = RrfFuser.fuse(List.of(), sparse, 3, 60, NULL_HYDRATOR);
    assertThat(fused).extracting(SearchHit::id).containsExactly("x", "y", "z");
  }

  @Test
  void zeroKReturnsEmpty() {
    List<SearchHit> fused = RrfFuser.fuse(List.of(dense("a")), List.of(sparse("b")), 0, 60, null);
    assertThat(fused).isEmpty();
  }
}
