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
package com.integrallis.vectors.hnsw;

/**
 * Bounded, worker-owned cache of exact symmetric pair scores during diversity pruning. A cache
 * belongs to one immutable vector source and metric for one build/append operation. Hash collisions
 * only evict entries: the complete ordinal pair is checked before a hit.
 */
final class ExactScoreCache {
  private static final int BITS = 14;
  private final long[] keys = new long[1 << BITS];
  private final float[] scores = new float[1 << BITS];

  static long key(int a, int b) {
    int low = Math.min(a, b);
    int high = Math.max(a, b);
    return ((long) (low + 1) << 32) | ((high + 1L) & 0xffffffffL);
  }

  float get(long key) {
    int slot = slot(key);
    return keys[slot] == key ? scores[slot] : -1f;
  }

  void put(long key, float score) {
    int slot = slot(key);
    keys[slot] = key;
    scores[slot] = score;
  }

  private static int slot(long key) {
    return (int) ((key * 0x9e3779b97f4a7c15L) >>> (64 - BITS));
  }
}
