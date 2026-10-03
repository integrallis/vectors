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

import java.util.ArrayDeque;
import java.util.BitSet;
import java.util.Deque;

/**
 * Structural health checks on a built HNSW graph.
 *
 * <p>Layer-0 reachability diagnoses fragmentation, but is not by itself a query recall ceiling:
 * upper-layer descent can enter layer 0 at a different node. {@link #unreachableAcrossLayers}
 * follows layer-respecting paths to distinguish nodes the hierarchy cannot possibly reach.
 * Reachability is necessary, not sufficient, for retrieval under a query-dependent greedy walk.
 *
 * <p><b>Diagnosis only.</b> Nothing here repairs a graph. Automatic component-joining is a known
 * trap: Lucene shipped it, found it could take an inordinate amount of time on degenerate input
 * (their reported case was a corpus of identical vectors), and reverted it. Reporting the damage is
 * cheap and safe; repairing it in the build path is neither.
 */
public final class HnswGraphDiagnostics {

  private HnswGraphDiagnostics() {}

  /**
   * What a structural scan of one graph found.
   *
   * @param nodes live nodes in the graph
   * @param unreachableFromEntry nodes not reachable from the entry node by following directed edges
   *     at layer 0; upper layers may still provide a route to these nodes
   * @param danglingEdges edges pointing at ordinals that were never initialised
   * @param selfLoops edges from a node to itself
   * @param minOutDegree smallest layer-0 out-degree over live nodes, or 0 for an empty graph
   * @param maxOutDegree largest layer-0 out-degree
   * @param meanOutDegree mean layer-0 out-degree
   * @param isolatedNodes nodes with no layer-0 out-edges at all
   */
  public record Report(
      int nodes,
      int unreachableFromEntry,
      int danglingEdges,
      int selfLoops,
      int minOutDegree,
      int maxOutDegree,
      double meanOutDegree,
      int isolatedNodes) {

    /** True when every live node can be reached from the entry node at layer 0. */
    public boolean isFullyReachable() {
      return unreachableFromEntry == 0;
    }

    /** True when no structural defect was found. */
    public boolean isHealthy() {
      return isFullyReachable() && danglingEdges == 0 && selfLoops == 0 && isolatedNodes == 0;
    }

    @Override
    public String toString() {
      return "nodes=%,d unreachable=%,d (%.4f%%) dangling=%d selfLoops=%d isolated=%d outDegree[min=%d mean=%.1f max=%d]"
          .formatted(
              nodes,
              unreachableFromEntry,
              nodes == 0 ? 0.0 : 100.0 * unreachableFromEntry / nodes,
              danglingEdges,
              selfLoops,
              isolatedNodes,
              minOutDegree,
              meanOutDegree,
              maxOutDegree);
    }
  }

  /**
   * Scans {@code graph} at layer 0 and reports what it found.
   *
   * <p>Runs a breadth-first traversal from the entry node over directed layer-0 edges, so cost is
   * linear in edges. Safe to call on a graph built with headroom: ordinals at or beyond {@link
   * HnswGraph#size()} are treated as not live.
   */
  public static Report inspect(HnswGraph graph) {
    int live = graph.size();
    if (live == 0) {
      return new Report(0, 0, 0, 0, 0, 0, 0, 0);
    }

    int dangling = 0;
    int selfLoops = 0;
    int isolated = 0;
    int minDegree = Integer.MAX_VALUE;
    int maxDegree = 0;
    long totalDegree = 0;

    for (int node = 0; node < live; node++) {
      NeighborArray neighbors = graph.getNeighbors(node, 0);
      int degree = neighbors == null ? 0 : neighbors.size();
      totalDegree += degree;
      minDegree = Math.min(minDegree, degree);
      maxDegree = Math.max(maxDegree, degree);
      if (degree == 0) {
        isolated++;
      }
      for (int i = 0; i < degree; i++) {
        int target = neighbors.node(i);
        if (target == node) {
          selfLoops++;
        } else if (target < 0 || target >= live || graph.getNeighbors(target, 0) == null) {
          dangling++;
        }
      }
    }

    int reached = reachableFromEntry(graph, live);

    return new Report(
        live,
        live - reached,
        dangling,
        selfLoops,
        minDegree == Integer.MAX_VALUE ? 0 : minDegree,
        maxDegree,
        totalDegree / (double) live,
        isolated);
  }

  /**
   * Counts nodes with no directed path from the entry at maxLevel, allowing downward moves only.
   * This is a necessary reachability condition, not a simulation of query-dependent greedy search.
   */
  public static int unreachableAcrossLayers(HnswGraph graph) {
    int n = graph.size();
    if (n == 0) return 0;
    int entry = graph.entryNode();
    if (entry < 0 || entry >= n) return n;
    BitSet reached = new BitSet(n);
    reached.set(entry);
    for (int layer = graph.maxLevel(); layer >= 0; layer--) {
      Deque<Integer> frontier = new ArrayDeque<>();
      for (int id = reached.nextSetBit(0); id >= 0; id = reached.nextSetBit(id + 1))
        frontier.add(id);
      while (!frontier.isEmpty()) {
        NeighborArray neighbors = graph.getNeighbors(frontier.removeFirst(), layer);
        if (neighbors == null) continue;
        for (int i = 0; i < neighbors.size(); i++) {
          int target = neighbors.node(i);
          if (target < 0 || target >= n || graph.nodeLevel(target) < layer || reached.get(target))
            continue;
          reached.set(target);
          frontier.add(target);
        }
      }
    }
    return n - reached.cardinality();
  }

  /** Counts nodes reachable from the entry node along directed layer-0 edges. */
  private static int reachableFromEntry(HnswGraph graph, int live) {
    int entry = graph.entryNode();
    if (entry < 0 || entry >= live) {
      return 0;
    }
    BitSet seen = new BitSet(live);
    Deque<Integer> frontier = new ArrayDeque<>();
    seen.set(entry);
    frontier.add(entry);
    int reached = 1;

    while (!frontier.isEmpty()) {
      NeighborArray neighbors = graph.getNeighbors(frontier.removeFirst(), 0);
      if (neighbors == null) {
        continue;
      }
      for (int i = 0; i < neighbors.size(); i++) {
        int target = neighbors.node(i);
        if (target < 0 || target >= live || seen.get(target)) {
          continue;
        }
        seen.set(target);
        reached++;
        frontier.add(target);
      }
    }
    return reached;
  }
}
