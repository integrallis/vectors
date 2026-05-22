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

import java.util.Arrays;
import java.util.Objects;

/**
 * Immutable, structurally-sharing persistent hash map (Hash Array Mapped Trie, Bagwell 2001).
 *
 * <p>Every mutation ({@link #put(Object, Object)}, {@link #remove(Object)}) returns a new {@code
 * Hamt} that shares unchanged subtrees with the receiver. This makes the "snapshot a map for the
 * next collection generation" operation O(1) — fundamental for the commit pipeline, which used to
 * pay O(N) per commit for a {@code HashMap.putAll}.
 *
 * <p><b>Internal API.</b> This class is internal to {@code vectors-db}. The internal-dispatch
 * methods {@code get/put/remove(int shift, int keyHash, K key, ...)} are declared on the sealed
 * interface so the leaf, indexed, and array nodes can implement them; they take a {@code shift}
 * parameter that selects the bit fragment of the key's hash and so are not useful from outside the
 * recursion.
 *
 * <p><b>Branching factor.</b> 32-way (5-bit fragments). Depth is {@code ceil(32/5) = 7} for any key
 * set — every read and write is O(7) ≈ O(1) for all practical N. This is the well-established HAMT
 * trade-off (Bagwell, Steindorfer-Vinju, Clojure, Scala, Vavr).
 *
 * <p><b>Threading.</b> Instances are immutable post-construction. Once a {@code Hamt} reference is
 * published, it can be read from any number of threads without synchronization. The commit pipeline
 * publishes successors via a volatile field on the enclosing {@code Generation} record, which gives
 * the happens-before edge readers need.
 *
 * @param <K> key type
 * @param <V> value type
 */
public sealed interface Hamt<K, V>
    permits Hamt.Empty, Hamt.Leaf, Hamt.LeafList, Hamt.Indexed, Hamt.Array {

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  /** Returns the canonical empty Hamt. Constant-time. */
  @SuppressWarnings("unchecked")
  static <K, V> Hamt<K, V> empty() {
    return (Hamt<K, V>) Empty.INSTANCE;
  }

  /**
   * Returns the value associated with {@code key}, or {@code null} if absent. Null keys are
   * permitted (the key's hash is taken as 0, matching {@link Objects#hashCode(Object)}).
   */
  default V get(K key) {
    return getInternal(0, mix(Objects.hashCode(key)), key);
  }

  /** Returns {@code true} if the key is present (even with a null value). */
  default boolean containsKey(K key) {
    return containsKeyInternal(0, mix(Objects.hashCode(key)), key);
  }

  /**
   * Returns a new Hamt with {@code key} bound to {@code value}. If the binding already exists with
   * a value {@link Objects#equals equal} to {@code value}, returns {@code this} (no allocation).
   */
  default Hamt<K, V> put(K key, V value) {
    return putInternal(0, mix(Objects.hashCode(key)), key, value);
  }

  /**
   * Returns a new Hamt with {@code key} absent. If the key was not present, returns {@code this}.
   */
  default Hamt<K, V> remove(K key) {
    return removeInternal(0, mix(Objects.hashCode(key)), key);
  }

  /** Number of bindings. */
  int size();

  /** {@code true} iff size is zero. */
  boolean isEmpty();

  // ---------------------------------------------------------------------------
  // Internal dispatch
  // ---------------------------------------------------------------------------

  /** Returns the value bound to {@code key}, or {@code null}. {@code keyHash} is pre-mixed. */
  V getInternal(int shift, int keyHash, K key);

  /** Whether {@code key} is bound. {@code keyHash} is pre-mixed. */
  boolean containsKeyInternal(int shift, int keyHash, K key);

  /** Path-copy {@code put}. */
  Hamt<K, V> putInternal(int shift, int keyHash, K key, V value);

  /** Path-copy {@code remove}. */
  Hamt<K, V> removeInternal(int shift, int keyHash, K key);

  // ---------------------------------------------------------------------------
  // Static helpers (bit manipulation + array path-copy primitives)
  // ---------------------------------------------------------------------------

  /** Fragment width in bits: 32-way branching gives depth ≤ 7 for any 32-bit hash. */
  int SIZE = 5;

  /**
   * {@code 1 << SIZE = 32}: max children per node, and the sentinel "promote to ArrayNode" size.
   */
  int BUCKET_SIZE = 1 << SIZE;

  /**
   * When an {@link Indexed} node grows past this many children, it converts to an {@link Array}.
   */
  int MAX_INDEX_NODE = BUCKET_SIZE >> 1; // 16

  /**
   * When an {@link Array} node shrinks to this many children, it demotes back to {@link Indexed}.
   */
  int MIN_ARRAY_NODE = BUCKET_SIZE >> 2; // 8

  /** Extracts the {@code SIZE}-bit fragment of {@code hash} at depth {@code shift}. */
  static int hashFragment(int shift, int hash) {
    return (hash >>> shift) & (BUCKET_SIZE - 1);
  }

  /** One-hot bitmap for a fragment in [0, 32). */
  static int toBitmap(int hash) {
    return 1 << hash;
  }

  /** Population count of bits set below {@code bit} in {@code bitmap} — the compact-array index. */
  static int fromBitmap(int bitmap, int bit) {
    return Integer.bitCount(bitmap & (bit - 1));
  }

  /**
   * HashMap-style hash spread: XOR the high 16 bits into the low 16 bits. Mitigates the case where
   * {@link Object#hashCode()} returns values whose entropy lives entirely in the high bits (the
   * 5-bit HAMT fragment at depth 0 would otherwise see only constant bits and force every key into
   * the same subtree). Same primitive as {@code java.util.HashMap.hash}.
   */
  static int mix(int h) {
    return h ^ (h >>> 16);
  }

  /** Returns a copy of {@code arr} with element at {@code index} replaced by {@code newElement}. */
  static Object[] arrayUpdate(Object[] arr, int index, Object newElement) {
    Object[] newArr = Arrays.copyOf(arr, arr.length);
    newArr[index] = newElement;
    return newArr;
  }

  /** Returns a copy of {@code arr} with the element at {@code index} removed. */
  static Object[] arrayRemove(Object[] arr, int index) {
    Object[] newArr = new Object[arr.length - 1];
    System.arraycopy(arr, 0, newArr, 0, index);
    System.arraycopy(arr, index + 1, newArr, index, arr.length - index - 1);
    return newArr;
  }

  /** Returns a copy of {@code arr} with {@code newElem} inserted at {@code index}. */
  static Object[] arrayInsert(Object[] arr, int index, Object newElem) {
    Object[] newArr = new Object[arr.length + 1];
    System.arraycopy(arr, 0, newArr, 0, index);
    newArr[index] = newElem;
    System.arraycopy(arr, index, newArr, index + 1, arr.length - index);
    return newArr;
  }

  // ===========================================================================
  // Node implementations
  // ===========================================================================

  /** Singleton empty node. */
  final class Empty<K, V> implements Hamt<K, V> {
    static final Empty<?, ?> INSTANCE = new Empty<>();

    private Empty() {}

    @Override
    public V getInternal(int shift, int keyHash, K key) {
      return null;
    }

    @Override
    public boolean containsKeyInternal(int shift, int keyHash, K key) {
      return false;
    }

    @Override
    public Hamt<K, V> putInternal(int shift, int keyHash, K key, V value) {
      return new Leaf<>(keyHash, key, value);
    }

    @Override
    public Hamt<K, V> removeInternal(int shift, int keyHash, K key) {
      return this;
    }

    @Override
    public int size() {
      return 0;
    }

    @Override
    public boolean isEmpty() {
      return true;
    }
  }

  /**
   * Single key-value leaf. Modeled as a record so the canonical constructor + accessors are
   * generated for us; we add the trie-dispatch methods.
   */
  record Leaf<K, V>(int hash, K key, V value) implements Hamt<K, V> {

    @Override
    public V getInternal(int shift, int keyHash, K key) {
      return matches(keyHash, key) ? value : null;
    }

    @Override
    public boolean containsKeyInternal(int shift, int keyHash, K key) {
      return matches(keyHash, key);
    }

    @Override
    public Hamt<K, V> putInternal(int shift, int keyHash, K key, V value) {
      if (matches(keyHash, key)) {
        // Same key: if the value is the same too, no path-copy needed.
        return Objects.equals(value, this.value) ? this : new Leaf<>(hash, key, value);
      }
      // Different key: merge with a new singleton at this shift level.
      return mergeLeaves(shift, this, new Leaf<>(keyHash, key, value));
    }

    @Override
    public Hamt<K, V> removeInternal(int shift, int keyHash, K key) {
      return matches(keyHash, key) ? Hamt.empty() : this;
    }

    @Override
    public int size() {
      return 1;
    }

    @Override
    public boolean isEmpty() {
      return false;
    }

    private boolean matches(int keyHash, K otherKey) {
      return keyHash == hash && Objects.equals(otherKey, this.key);
    }
  }

  /**
   * Hash-collision bucket: a non-empty linked list of leaves that all share the same 32-bit hash
   * (so HAMT can't distinguish them by hash fragments alone). Reached only on hash collisions,
   * which {@link #mix} reduces to near-zero in practice for well-behaved hashCodes.
   *
   * <p>Kept as a {@code final class} (not a record) because {@code size} is computed from the tail
   * and records can't carry derived state without going through the canonical constructor.
   */
  final class LeafList<K, V> implements Hamt<K, V> {
    final int hash;
    final K key;
    final V value;
    final int size;

    /**
     * The remainder of the chain. Always a {@link Leaf} or another {@link LeafList} with the same
     * {@code hash}. Never null.
     */
    final Hamt<K, V> tail;

    LeafList(int hash, K key, V value, Hamt<K, V> tail) {
      this.hash = hash;
      this.key = key;
      this.value = value;
      this.size = 1 + tail.size();
      this.tail = tail;
    }

    @Override
    public V getInternal(int shift, int keyHash, K key) {
      if (keyHash != hash) {
        return null;
      }
      // Linear walk through the collision chain. Chains are rare and short.
      Hamt<K, V> node = this;
      while (node instanceof LeafList<K, V> ll) {
        if (Objects.equals(key, ll.key)) {
          return ll.value;
        }
        node = ll.tail;
      }
      if (node instanceof Leaf<K, V> leaf && Objects.equals(key, leaf.key())) {
        return leaf.value();
      }
      return null;
    }

    @Override
    public boolean containsKeyInternal(int shift, int keyHash, K key) {
      return getInternal(shift, keyHash, key) != null || hasKey(keyHash, key);
    }

    /** Slower contains that also recognizes null-valued bindings. */
    private boolean hasKey(int keyHash, K key) {
      if (keyHash != hash) {
        return false;
      }
      Hamt<K, V> node = this;
      while (node instanceof LeafList<K, V> ll) {
        if (Objects.equals(key, ll.key)) {
          return true;
        }
        node = ll.tail;
      }
      return node instanceof Leaf<K, V> leaf && Objects.equals(key, leaf.key());
    }

    @Override
    public Hamt<K, V> putInternal(int shift, int keyHash, K key, V value) {
      if (keyHash == hash) {
        // Same hash bucket: replace the existing entry if present, else prepend.
        Hamt<K, V> filtered = removeFromChain(key);
        if (filtered instanceof LeafList<K, V> || filtered instanceof Leaf<K, V>) {
          return new LeafList<>(hash, key, value, filtered);
        }
        // Filtered chain emptied (impossible since we always had at least 2 entries here),
        // fall through to a single Leaf.
        return new Leaf<>(hash, key, value);
      }
      // Different hash: escalate to a branching Indexed/Array node.
      return mergeLeaves(shift, this, new Leaf<>(keyHash, key, value));
    }

    @Override
    public Hamt<K, V> removeInternal(int shift, int keyHash, K key) {
      if (keyHash != hash) {
        return this;
      }
      return removeFromChain(key);
    }

    /**
     * Returns the chain with {@code k} removed. If the chain shrinks to one element, returns a
     * plain {@link Leaf}; if it would empty entirely, returns {@link Empty} (but that can only
     * happen when called on a 2-element list with both entries matching).
     */
    private Hamt<K, V> removeFromChain(K k) {
      // Walk the chain, accumulating non-matching entries into `keep`.
      Hamt<K, V> keep = Hamt.empty();
      boolean removed = false;
      Hamt<K, V> node = this;
      while (node instanceof LeafList<K, V> ll) {
        if (!removed && Objects.equals(k, ll.key)) {
          removed = true;
        } else {
          keep =
              keep instanceof Empty<K, V>
                  ? new Leaf<>(ll.hash, ll.key, ll.value)
                  : new LeafList<>(ll.hash, ll.key, ll.value, keep);
        }
        node = ll.tail;
      }
      // Final element is a Leaf (chain terminator).
      if (node instanceof Leaf<K, V> leaf) {
        if (!removed && Objects.equals(k, leaf.key())) {
          // The terminator matched: don't add it to keep.
        } else {
          keep =
              keep instanceof Empty<K, V>
                  ? leaf
                  : new LeafList<>(leaf.hash(), leaf.key(), leaf.value(), keep);
        }
      }
      return keep;
    }

    @Override
    public int size() {
      return size;
    }

    @Override
    public boolean isEmpty() {
      return false;
    }
  }

  /**
   * Sparse-bitmap branching node: holds up to {@link #MAX_INDEX_NODE} children indexed by a compact
   * bitmap. Each set bit at position {@code i} means a child exists for fragment {@code i}; that
   * child lives at {@code subNodes[bitCount(bitmap & (bit_i - 1))]}.
   */
  final class Indexed<K, V> implements Hamt<K, V> {
    final int bitmap;
    final int size;
    final Object[] subNodes;

    Indexed(int bitmap, int size, Object[] subNodes) {
      this.bitmap = bitmap;
      this.size = size;
      this.subNodes = subNodes;
    }

    @SuppressWarnings("unchecked")
    @Override
    public V getInternal(int shift, int keyHash, K key) {
      int frag = hashFragment(shift, keyHash);
      int bit = toBitmap(frag);
      if ((bitmap & bit) == 0) {
        return null;
      }
      Hamt<K, V> child = (Hamt<K, V>) subNodes[fromBitmap(bitmap, bit)];
      return child.getInternal(shift + SIZE, keyHash, key);
    }

    @SuppressWarnings("unchecked")
    @Override
    public boolean containsKeyInternal(int shift, int keyHash, K key) {
      int frag = hashFragment(shift, keyHash);
      int bit = toBitmap(frag);
      if ((bitmap & bit) == 0) {
        return false;
      }
      Hamt<K, V> child = (Hamt<K, V>) subNodes[fromBitmap(bitmap, bit)];
      return child.containsKeyInternal(shift + SIZE, keyHash, key);
    }

    @SuppressWarnings("unchecked")
    @Override
    public Hamt<K, V> putInternal(int shift, int keyHash, K key, V value) {
      int frag = hashFragment(shift, keyHash);
      int bit = toBitmap(frag);
      int index = fromBitmap(bitmap, bit);
      boolean exists = (bitmap & bit) != 0;
      Hamt<K, V> existingChild = exists ? (Hamt<K, V>) subNodes[index] : null;
      Hamt<K, V> newChild =
          exists
              ? existingChild.putInternal(shift + SIZE, keyHash, key, value)
              : new Leaf<>(keyHash, key, value);
      if (exists && newChild == existingChild) {
        // Same-value put: no path-copy needed at any level above the leaf.
        return this;
      }
      if (exists) {
        return new Indexed<>(
            bitmap,
            size - existingChild.size() + newChild.size(),
            arrayUpdate(subNodes, index, newChild));
      }
      // New slot: grow OR promote to dense Array if we'd exceed the sparse threshold.
      if (subNodes.length >= MAX_INDEX_NODE) {
        return expandToArray(frag, newChild);
      }
      return new Indexed<>(
          bitmap | bit, size + newChild.size(), arrayInsert(subNodes, index, newChild));
    }

    @SuppressWarnings("unchecked")
    @Override
    public Hamt<K, V> removeInternal(int shift, int keyHash, K key) {
      int frag = hashFragment(shift, keyHash);
      int bit = toBitmap(frag);
      if ((bitmap & bit) == 0) {
        return this;
      }
      int index = fromBitmap(bitmap, bit);
      Hamt<K, V> existingChild = (Hamt<K, V>) subNodes[index];
      Hamt<K, V> newChild = existingChild.removeInternal(shift + SIZE, keyHash, key);
      if (newChild == existingChild) {
        return this;
      }
      boolean removed = newChild.isEmpty();
      if (removed) {
        int newBitmap = bitmap & ~bit;
        if (newBitmap == 0) {
          return Hamt.empty();
        }
        // Collapse to the lone sibling Leaf if only one child remains and it's a leaf.
        if (subNodes.length == 2 && subNodes[index ^ 1] instanceof Leaf<?, ?>) {
          return (Hamt<K, V>) subNodes[index ^ 1];
        }
        return new Indexed<>(newBitmap, size - existingChild.size(), arrayRemove(subNodes, index));
      }
      return new Indexed<>(
          bitmap,
          size - existingChild.size() + newChild.size(),
          arrayUpdate(subNodes, index, newChild));
    }

    /**
     * Promotes this sparse node to a dense {@link Array} node when adding {@code newChild} at
     * {@code frag} would push us past {@link #MAX_INDEX_NODE}.
     */
    private Array<K, V> expandToArray(int frag, Hamt<K, V> newChild) {
      Object[] arr = new Object[BUCKET_SIZE];
      int bit = bitmap;
      int srcIdx = 0;
      int count = 0;
      for (int i = 0; i < BUCKET_SIZE; i++) {
        if ((bit & 1) != 0) {
          arr[i] = subNodes[srcIdx++];
          count++;
        } else if (i == frag) {
          arr[i] = newChild;
          count++;
        } else {
          arr[i] = Hamt.empty();
        }
        bit >>>= 1;
      }
      return new Array<>(count, size + newChild.size(), arr);
    }

    @Override
    public int size() {
      return size;
    }

    @Override
    public boolean isEmpty() {
      return false;
    }
  }

  /**
   * Dense 32-slot branching node. Used when an {@link Indexed} would otherwise exceed {@link
   * #MAX_INDEX_NODE}. Demotes back to {@link Indexed} when shrinking past {@link #MIN_ARRAY_NODE}.
   */
  final class Array<K, V> implements Hamt<K, V> {
    /**
     * Number of non-empty slots. {@code subNodes[i] instanceof Empty} when slot {@code i} is
     * unused.
     */
    final int count;

    final int size;
    final Object[] subNodes; // length is always BUCKET_SIZE = 32

    Array(int count, int size, Object[] subNodes) {
      this.count = count;
      this.size = size;
      this.subNodes = subNodes;
    }

    @SuppressWarnings("unchecked")
    @Override
    public V getInternal(int shift, int keyHash, K key) {
      int frag = hashFragment(shift, keyHash);
      return ((Hamt<K, V>) subNodes[frag]).getInternal(shift + SIZE, keyHash, key);
    }

    @SuppressWarnings("unchecked")
    @Override
    public boolean containsKeyInternal(int shift, int keyHash, K key) {
      int frag = hashFragment(shift, keyHash);
      return ((Hamt<K, V>) subNodes[frag]).containsKeyInternal(shift + SIZE, keyHash, key);
    }

    @SuppressWarnings("unchecked")
    @Override
    public Hamt<K, V> putInternal(int shift, int keyHash, K key, V value) {
      int frag = hashFragment(shift, keyHash);
      Hamt<K, V> existingChild = (Hamt<K, V>) subNodes[frag];
      Hamt<K, V> newChild = existingChild.putInternal(shift + SIZE, keyHash, key, value);
      if (newChild == existingChild) {
        return this;
      }
      boolean wasEmpty = existingChild.isEmpty();
      boolean nowEmpty = newChild.isEmpty();
      if (wasEmpty && !nowEmpty) {
        return new Array<>(
            count + 1, size + newChild.size(), arrayUpdate(subNodes, frag, newChild));
      }
      if (!wasEmpty && nowEmpty) {
        // Removing the last entry of a child during a put? Should not happen — defensive only.
        return new Array<>(
            count - 1, size - existingChild.size(), arrayUpdate(subNodes, frag, newChild));
      }
      return new Array<>(
          count,
          size - existingChild.size() + newChild.size(),
          arrayUpdate(subNodes, frag, newChild));
    }

    @SuppressWarnings("unchecked")
    @Override
    public Hamt<K, V> removeInternal(int shift, int keyHash, K key) {
      int frag = hashFragment(shift, keyHash);
      Hamt<K, V> existingChild = (Hamt<K, V>) subNodes[frag];
      Hamt<K, V> newChild = existingChild.removeInternal(shift + SIZE, keyHash, key);
      if (newChild == existingChild) {
        return this;
      }
      if (existingChild.isEmpty() == newChild.isEmpty()) {
        // Slot occupancy didn't change.
        return new Array<>(
            count,
            size - existingChild.size() + newChild.size(),
            arrayUpdate(subNodes, frag, newChild));
      }
      // Slot transitioned to empty.
      if (count - 1 <= MIN_ARRAY_NODE) {
        return packToIndexed(frag);
      }
      return new Array<>(
          count - 1, size - existingChild.size(), arrayUpdate(subNodes, frag, newChild));
    }

    /** Demotes this dense node back to a sparse {@link Indexed}, excluding {@code excludedFrag}. */
    @SuppressWarnings("unchecked")
    private Indexed<K, V> packToIndexed(int excludedFrag) {
      Object[] packed = new Object[count - 1];
      int bitmap = 0;
      int size = 0;
      int dst = 0;
      for (int i = 0; i < BUCKET_SIZE; i++) {
        Hamt<K, V> child = (Hamt<K, V>) subNodes[i];
        if (i != excludedFrag && !child.isEmpty()) {
          size += child.size();
          packed[dst++] = child;
          bitmap |= (1 << i);
        }
      }
      return new Indexed<>(bitmap, size, packed);
    }

    @Override
    public int size() {
      return size;
    }

    @Override
    public boolean isEmpty() {
      return false;
    }
  }

  // ---------------------------------------------------------------------------
  // Internal helpers (leaf merging)
  // ---------------------------------------------------------------------------

  /**
   * Merges two leaves at the same {@code shift} level. If the full 32-bit hashes are equal, the
   * result is a {@link LeafList}; otherwise the result is an {@link Indexed} node containing both
   * (recursing deeper if their hashes share the fragment at the current level).
   */
  @SuppressWarnings("unchecked")
  static <K, V> Hamt<K, V> mergeLeaves(int shift, Hamt<K, V> leaf1, Leaf<K, V> leaf2) {
    int h1 = leafHash(leaf1);
    int h2 = leaf2.hash();
    if (h1 == h2) {
      // Full hash collision → linked LeafList.
      return new LeafList<>(h1, leaf2.key(), leaf2.value(), leaf1);
    }
    int subH1 = hashFragment(shift, h1);
    int subH2 = hashFragment(shift, h2);
    int newBitmap = toBitmap(subH1) | toBitmap(subH2);
    if (subH1 == subH2) {
      // Same fragment at this depth — recurse one level deeper.
      Hamt<K, V> merged = mergeLeaves(shift + SIZE, leaf1, leaf2);
      return new Indexed<>(newBitmap, merged.size(), new Object[] {merged});
    }
    // Different fragments — pack both into an IndexedNode in sorted order so the bitmap index
    // matches the array layout.
    Object[] children = subH1 < subH2 ? new Object[] {leaf1, leaf2} : new Object[] {leaf2, leaf1};
    return new Indexed<>(newBitmap, leaf1.size() + leaf2.size(), children);
  }

  /** Hash for the head of a leaf (Leaf or LeafList). */
  private static <K, V> int leafHash(Hamt<K, V> leaf) {
    return switch (leaf) {
      case Leaf<K, V> l -> l.hash();
      case LeafList<K, V> ll -> ll.hash;
      default -> throw new IllegalStateException("leafHash called on non-leaf: " + leaf);
    };
  }
}
