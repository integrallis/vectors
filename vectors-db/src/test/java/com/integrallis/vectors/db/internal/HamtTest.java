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
package com.integrallis.vectors.db.internal;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.TreeSet;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class HamtTest {

  @Test
  void emptyIsEmpty() {
    Hamt<Integer, Integer> h = Hamt.empty();
    assertThat(h.size()).isZero();
    assertThat(h.isEmpty()).isTrue();
    assertThat(h.get(1)).isNull();
    assertThat(h.containsKey(1)).isFalse();
  }

  @Test
  void singleEntryRoundTrip() {
    Hamt<String, Integer> h = Hamt.<String, Integer>empty().put("a", 1);
    assertThat(h.size()).isEqualTo(1);
    assertThat(h.isEmpty()).isFalse();
    assertThat(h.get("a")).isEqualTo(1);
    assertThat(h.get("absent")).isNull();
    assertThat(h.containsKey("a")).isTrue();
    assertThat(h.containsKey("absent")).isFalse();
  }

  @Test
  void putReturnsNewInstanceWhenValueChanges() {
    Hamt<String, Integer> base = Hamt.<String, Integer>empty().put("a", 1);
    Hamt<String, Integer> changed = base.put("a", 2);
    assertThat(changed).isNotSameAs(base);
    assertThat(changed.get("a")).isEqualTo(2);
    assertThat(base.get("a")).isEqualTo(1);
  }

  @Test
  void samePutReturnsSelfWhenValueUnchanged() {
    // The Vavr port didn't have this optimization; we added it. Verify that idempotent
    // puts don't allocate a new spine.
    Hamt<String, Integer> base = Hamt.<String, Integer>empty().put("a", 1).put("b", 2);
    Hamt<String, Integer> sameAgain = base.put("a", 1);
    assertThat(sameAgain).isSameAs(base);
  }

  @Test
  void putReplacesExistingKey() {
    Hamt<String, Integer> h = Hamt.<String, Integer>empty().put("a", 1).put("a", 2).put("a", 3);
    assertThat(h.size()).isEqualTo(1);
    assertThat(h.get("a")).isEqualTo(3);
  }

  @Test
  void removeFromEmpty() {
    Hamt<Integer, Integer> h = Hamt.<Integer, Integer>empty().remove(1);
    assertThat(h.size()).isZero();
    assertThat(h.isEmpty()).isTrue();
  }

  @Test
  void removeUnknownKey() {
    Hamt<Integer, Integer> h = Hamt.<Integer, Integer>empty().put(1, 2);
    Hamt<Integer, Integer> stillThere = h.remove(3);
    assertThat(stillThere).isSameAs(h);
    assertThat(stillThere.size()).isEqualTo(1);
    Hamt<Integer, Integer> gone = stillThere.remove(1);
    assertThat(gone.size()).isZero();
    assertThat(gone.get(1)).isNull();
  }

  @Test
  void nullKeyAndValueArePermitted() {
    Hamt<Integer, Integer> h = Hamt.<Integer, Integer>empty().put(null, 7);
    assertThat(h.containsKey(null)).isTrue();
    assertThat(h.get(null)).isEqualTo(7);
  }

  @Test
  void hashCollisionSpawnsLeafList() {
    // Two distinct keys with intentionally colliding hashCodes — the trie must keep both via
    // a LeafList rather than overwriting one.
    Hamt<WeakInteger, Integer> h =
        Hamt.<WeakInteger, Integer>empty().put(new WeakInteger(1), 1).put(new WeakInteger(11), 11);
    assertThat(h.size()).isEqualTo(2);
    assertThat(h.get(new WeakInteger(1))).isEqualTo(1);
    assertThat(h.get(new WeakInteger(11))).isEqualTo(11);
    assertThat(h.get(new WeakInteger(21))).isNull();
  }

  @Test
  void hashCollisionRemoveLeavesOtherEntries() {
    Hamt<WeakInteger, Integer> h =
        Hamt.<WeakInteger, Integer>empty()
            .put(new WeakInteger(1), 1)
            .put(new WeakInteger(11), 11)
            .put(new WeakInteger(21), 21);
    Hamt<WeakInteger, Integer> after = h.remove(new WeakInteger(11));
    assertThat(after.size()).isEqualTo(2);
    assertThat(after.get(new WeakInteger(1))).isEqualTo(1);
    assertThat(after.get(new WeakInteger(11))).isNull();
    assertThat(after.get(new WeakInteger(21))).isEqualTo(21);
  }

  @Test
  void deepestTreeViaPowersOfTwo() {
    // Powers of two keys exercise the depth dimension because hash >>> shift produces a
    // different fragment for each level.
    int n = Integer.SIZE;
    Hamt<Integer, Integer> h = Hamt.empty();
    List<Integer> ints = new ArrayList<>(n);
    for (int i = 0; i < n; i++) {
      int k = 1 << i;
      ints.add(k);
      h = h.put(k, k);
    }
    assertThat(h.size()).isEqualTo(n);
    for (int k : ints) {
      assertThat(h.get(k)).isEqualTo(k);
    }
  }

  @Test
  void randomized5000() {
    runRandomizedTest(5000, /* useWeakHash */ false);
  }

  @Test
  void randomized5000WithWeakHash() {
    runRandomizedTest(5000, /* useWeakHash */ true);
  }

  private void runRandomizedTest(int count, boolean useWeakHash) {
    Random r = new Random(0xC0FFEEL);
    HashMap<Object, Integer> oracle = new HashMap<>();
    for (int i = 0; i < count; i++) {
      Object k = useWeakHash ? new WeakInteger(r.nextInt()) : r.nextInt();
      oracle.put(k, r.nextInt());
    }

    Hamt<Object, Integer> h = Hamt.empty();
    for (var e : oracle.entrySet()) {
      h = h.put(e.getKey(), e.getValue());
    }
    assertThat(h.size()).isEqualTo(oracle.size());
    for (var e : oracle.entrySet()) {
      assertThat(h.get(e.getKey())).isEqualTo(e.getValue());
      assertThat(h.containsKey(e.getKey())).isTrue();
    }

    // Remove everything; the trie must shrink to empty without any pathology.
    for (Object k : new TreeSet<>(oracle.keySet().stream().map(Object::toString).toList())) {
      // TreeSet iteration order is irrelevant; we just want a deterministic removal order.
    }
    for (Object k : new ArrayList<>(oracle.keySet())) {
      h = h.remove(k);
      assertThat(h.containsKey(k)).isFalse();
      assertThat(h.get(k)).isNull();
    }
    assertThat(h.size()).isZero();
    assertThat(h.isEmpty()).isTrue();
  }

  @Test
  void lookupNullAmongManyKeys() {
    // Make sure the null key (hash = 0) survives a deep tree. Vavr had a regression on this once
    // — keeping the test as a guard.
    Hamt<Integer, Integer> h = Hamt.empty();
    for (int i = 0; i < 5000; i++) {
      h = h.put(i, i);
    }
    h = h.put(null, 42);
    assertThat(h.get(0)).isEqualTo(0); // hashCode of Integer 0 is 0 — same fragment as null
    assertThat(h.get(null)).isEqualTo(42);
    assertThat(h.containsKey(null)).isTrue();
  }

  @Test
  void expandsToArrayAndPacksBack() {
    // Cross the MAX_INDEX_NODE = 16 boundary at depth 0, then back below MIN_ARRAY_NODE = 8
    // by removing entries, exercising the IndexedNode -> ArrayNode -> IndexedNode transitions.
    Hamt<Integer, Integer> h = Hamt.empty();
    // 32 distinct fragments at depth 0 — one per slot — by spacing keys widely enough that
    // each falls into its own bucket at the root.
    List<Integer> keys = new ArrayList<>(32);
    for (int i = 0; i < 32; i++) {
      // Use mixed-hash-aware keys: integer hashCode = itself, then mix XORs h>>>16. Powers of 2
      // shifted by 16 bits give distinct depth-0 fragments after mixing.
      int k = (i << 16) ^ i;
      keys.add(k);
      h = h.put(k, i);
    }
    assertThat(h.size()).isEqualTo(32);
    for (int i = 0; i < 32; i++) {
      assertThat(h.get(keys.get(i))).isEqualTo(i);
    }

    // Remove 25 to drop below MIN_ARRAY_NODE — should pack back to Indexed without losing data.
    for (int i = 0; i < 25; i++) {
      h = h.remove(keys.get(i));
    }
    assertThat(h.size()).isEqualTo(7);
    for (int i = 25; i < 32; i++) {
      assertThat(h.get(keys.get(i))).isEqualTo(i);
    }
    for (int i = 0; i < 25; i++) {
      assertThat(h.get(keys.get(i))).isNull();
    }
  }

  /**
   * Integer wrapper with a deliberately weak hashCode that collides keys whose absolute values
   * agree modulo 10. Used to drive the {@link Hamt.LeafList} (collision-chain) code path.
   */
  private static final class WeakInteger {
    final int value;

    WeakInteger(int value) {
      this.value = value;
    }

    @Override
    public boolean equals(Object o) {
      return o instanceof WeakInteger w && w.value == value;
    }

    @Override
    public int hashCode() {
      return Math.abs(value) % 10;
    }

    @Override
    public String toString() {
      return "WeakInteger(" + value + ")";
    }
  }
}
