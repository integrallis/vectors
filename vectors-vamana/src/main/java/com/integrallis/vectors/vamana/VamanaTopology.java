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
package com.integrallis.vectors.vamana;

/**
 * Read-only adjacency view over a Vamana graph: the minimal surface {@link VamanaSearcher} needs to
 * traverse the graph (entry point + per-node neighbour lists). Decoupling the searcher from the
 * concrete heap {@link VamanaGraph} lets a disk-resident implementation page neighbour lists from
 * an mmap'd {@code graph.bin} on demand (see {@code PagedVamanaTopology}) without holding the whole
 * adjacency in heap, while the in-memory {@link VamanaGraph} implements it directly.
 *
 * <p>Implementations need not be thread-safe for mutation (the graph is immutable after build), but
 * concurrent {@link #neighbors} reads from different searcher threads must be safe — each searcher
 * passes its own {@code out} scratch array, so the only shared state read is the immutable
 * topology.
 */
public interface VamanaTopology {

  /** Number of nodes in the graph. */
  int size();

  /** Entry-point node id for beam search (the medoid); {@code -1} iff the graph is empty. */
  int medoid();

  /** Upper bound on any node's neighbour count (the Vamana {@code R} parameter). */
  int maxDegree();

  /**
   * Writes node {@code nodeId}'s neighbour ids into {@code out} in stored order and returns the
   * count. The count is at most {@link #maxDegree()}, so {@code out.length >= maxDegree()} must
   * hold. The stored order is significant: it is the exact iteration order the searcher's beam loop
   * depends on, so heap and paged implementations must agree byte-for-byte.
   *
   * @param nodeId node whose neighbours to read, in {@code [0, size())}
   * @param out caller-owned scratch buffer of length {@code >= maxDegree()}
   * @return the number of neighbour ids written to {@code out}
   */
  int neighbors(int nodeId, int[] out);
}
