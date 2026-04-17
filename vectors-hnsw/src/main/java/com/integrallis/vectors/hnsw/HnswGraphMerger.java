package com.integrallis.vectors.hnsw;

import com.integrallis.vectors.core.SimilarityFunction;

/**
 * Merges a compacted generation's HNSW graph into a new graph without full reconstruction.
 *
 * <p>The merge algorithm:
 *
 * <ol>
 *   <li>Initialise a fresh {@link HnswGraph} for the surviving nodes, preserving their original
 *       levels.
 *   <li>Copy and remap all edge lists from the old graph, skipping deleted neighbours.
 *   <li>Track under-connected layer-0 nodes (those that kept fewer than {@code M} neighbours after
 *       remapping) and repair them with a beam-search on the partially-built graph.
 * </ol>
 *
 * <p>The merger never re-trains the graph topology from scratch. Its cost is O(N' · M · d) for the
 * remap phase and O(R · ef · M) for the repair phase where R is the number of repaired nodes. This
 * is significantly cheaper than a full O(N' · log N' · M · d) rebuild when the deletion fraction is
 * small.
 *
 * <p>Used by {@link com.integrallis.vectors.db.index.HnswIndexAdapter#mergeFrom} (in-memory compact
 * path) and directly by the persistent compact path in {@code VectorCollectionImpl}.
 */
public final class HnswGraphMerger {

  private HnswGraphMerger() {}

  /**
   * Produces a new {@link HnswGraph} from {@code old} after removing deleted ordinals.
   *
   * @param old original graph (pre-compaction)
   * @param newVectors dense array of surviving vectors indexed by new ordinal
   * @param oldToNew mapping: {@code oldToNew[oldOrdinal]} = new ordinal, or {@code -1} if deleted
   * @param sim similarity function used when the original graph was built
   * @param maxConnections HNSW M parameter
   * @param efConstruction beam width used for repair searches
   * @return the merged graph, or {@code null} when {@code newVectors} is empty
   */
  public static HnswGraph merge(
      HnswGraph old,
      float[][] newVectors,
      int[] oldToNew,
      SimilarityFunction sim,
      int maxConnections,
      int efConstruction) {

    int newSize = newVectors.length;
    if (newSize == 0) {
      return null;
    }

    // Build reverse mapping: newToOld[newOrdinal] = oldOrdinal.
    int[] newToOld = new int[newSize];
    for (int o = 0; o < oldToNew.length; o++) {
      int n = oldToNew[o];
      if (n >= 0) {
        newToOld[n] = o;
      }
    }

    // Initialise merged graph; assign node levels from the old graph.
    HnswGraph merged = new HnswGraph(newSize, maxConnections);
    int entryNew = 0;
    int entryLevel = -1;
    for (int j = 0; j < newSize; j++) {
      int oldOrd = newToOld[j];
      int lvl = old.nodeLevel(oldOrd);
      merged.initNode(j, lvl);
      if (lvl > entryLevel) {
        entryLevel = lvl;
        entryNew = j;
      }
    }
    merged.setEntryNode(entryNew, Math.max(entryLevel, 0));

    // Remap edges; track under-connected layer-0 nodes for repair.
    // Layer-0 allows up to 2*M neighbours; repair if fewer than M remain after deletion.
    int repairThreshold = maxConnections;
    boolean[] needsRepair = new boolean[newSize];

    for (int j = 0; j < newSize; j++) {
      int oldOrd = newToOld[j];
      int level = merged.nodeLevel(j);
      for (int l = 0; l <= level; l++) {
        NeighborArray oldNbrs = old.getNeighbors(oldOrd, l);
        if (oldNbrs == null) {
          continue;
        }
        NeighborArray newNbrs = merged.getNeighbors(j, l);
        for (int i = 0; i < oldNbrs.size(); i++) {
          int remapped = oldToNew[oldNbrs.node(i)];
          if (remapped >= 0) {
            newNbrs.insert(remapped, oldNbrs.score(i));
          }
        }
        if (l == 0 && newNbrs.size() < repairThreshold) {
          needsRepair[j] = true;
        }
      }
    }

    // Repair under-connected nodes via beam search on the merged graph.
    HnswSearcher searcher = new HnswSearcher(merged, new InMemoryVectors(newVectors), sim);
    int searchK = maxConnections * 2 + 1;
    int searchEf = Math.max(efConstruction, searchK);

    for (int j = 0; j < newSize; j++) {
      if (!needsRepair[j]) {
        continue;
      }
      float[] query = newVectors[j];
      SearchResult result = searcher.search(query, searchK, searchEf);
      int[] ids = result.nodeIds();
      float[] scores = result.scores();
      NeighborArray jNbrs = merged.getNeighbors(j, 0);
      for (int i = 0; i < ids.length; i++) {
        int cand = ids[i];
        if (cand == j) {
          continue;
        }
        jNbrs.insert(cand, scores[i]);
        // Symmetric backlink.
        float backScore = sim.compare(newVectors[cand], query);
        merged.getNeighbors(cand, 0).insert(j, backScore);
      }
    }

    // Trim all layer-0 neighbour arrays to the official maximum (2*M).
    // The NeighborArray is allocated with capacity 2*M+1 (a temporary overflow slot used during
    // normal construction and the repair phase above). Before handing the graph to serialization or
    // search, the slot must be reclaimed so that HnswGraphCodec and the search code see a graph
    // that respects the M contract.
    int maxConn0 = merged.maxConnections0();
    for (int j = 0; j < newSize; j++) {
      NeighborArray na = merged.getNeighbors(j, 0);
      if (na != null) {
        na.trim(maxConn0);
      }
    }

    return merged;
  }
}
