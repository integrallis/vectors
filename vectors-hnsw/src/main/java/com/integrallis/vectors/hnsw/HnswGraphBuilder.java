package com.integrallis.vectors.hnsw;

import com.integrallis.vectors.core.SimilarityFunction;
import java.util.BitSet;

/**
 * Single-threaded HNSW graph builder implementing Algorithm 1 from the original paper.
 *
 * <p>Inserts vectors one at a time, performing greedy descent through upper layers followed by beam
 * search at lower layers with diversity-based neighbor selection.
 *
 * <p>Scratch buffers (BitSet, NodeQueues) are allocated once and reused across insertions.
 */
public final class HnswGraphBuilder {

  private final int maxConnections;
  private final int efConstruction;
  private final RandomAccessVectors vectors;
  private final SimilarityFunction similarityFunction;
  private final HnswGraph graph;
  private final RandomLevelGenerator levelGenerator;

  // Scratch buffers reused across insertions
  private final BitSet visited;
  private final NodeQueue candidates;
  private final NodeQueue results;

  private HnswGraphBuilder(
      int maxConnections,
      int efConstruction,
      RandomAccessVectors vectors,
      SimilarityFunction similarityFunction,
      long seed) {
    this.maxConnections = maxConnections;
    this.efConstruction = efConstruction;
    this.vectors = vectors;
    this.similarityFunction = similarityFunction;
    this.graph = new HnswGraph(vectors.size(), maxConnections);
    this.levelGenerator = new RandomLevelGenerator(maxConnections, seed);

    // Pre-allocate scratch buffers
    this.visited = new BitSet(vectors.size());
    this.candidates = new NodeQueue(efConstruction * 2, false); // max-heap
    this.results = new NodeQueue(efConstruction * 2, true); // min-heap
  }

  /** Creates a builder with a specific seed for deterministic construction. */
  public static HnswGraphBuilder create(
      int maxConnections,
      int efConstruction,
      RandomAccessVectors vectors,
      SimilarityFunction similarityFunction,
      long seed) {
    return new HnswGraphBuilder(maxConnections, efConstruction, vectors, similarityFunction, seed);
  }

  /** Creates a builder with a random seed. */
  public static HnswGraphBuilder create(
      int maxConnections,
      int efConstruction,
      RandomAccessVectors vectors,
      SimilarityFunction similarityFunction) {
    return create(maxConnections, efConstruction, vectors, similarityFunction, System.nanoTime());
  }

  /** Builds the HNSW graph by inserting all vectors sequentially. */
  public HnswGraph build() {
    for (int i = 0; i < vectors.size(); i++) {
      insertNode(i);
    }
    return graph;
  }

  /** Returns the graph (package-private, for testing). */
  HnswGraph getGraph() {
    return graph;
  }

  private void insertNode(int nodeId) {
    int level = levelGenerator.nextLevel();
    graph.initNode(nodeId, level);

    // First node becomes entry point
    if (graph.size() == 1) {
      graph.setEntryNode(nodeId, level);
      return;
    }

    float[] queryVec = vectors.getVector(nodeId);

    int ep = graph.entryNode();
    int epLevel = graph.maxLevel();

    // Phase 1: Greedy descent from epLevel down to level+1
    int currentBest = ep;
    for (int layer = epLevel; layer > level; layer--) {
      currentBest = greedySearch(queryVec, currentBest, layer);
    }

    // Phase 2: Insert at min(level, epLevel) down to 0
    int insertTopLayer = Math.min(level, epLevel);
    int[] entryPoints = new int[] {currentBest};

    for (int layer = insertTopLayer; layer >= 0; layer--) {
      int maxConn = (layer == 0) ? graph.maxConnections0() : maxConnections;

      // Beam search from entry points
      NeighborArray searchResults = searchLayer(queryVec, entryPoints, efConstruction, layer);

      // Diversity-based neighbor selection
      NeighborArray neighbors =
          NeighborSelector.selectDiverse(searchResults, maxConn, vectors, similarityFunction);

      // Forward edges: nodeId → neighbors
      NeighborArray nodeNeighbors = graph.getNeighbors(nodeId, layer);
      nodeNeighbors.copyFrom(neighbors);

      // Reverse edges (backlinks): each neighbor → nodeId
      for (int i = 0; i < neighbors.size(); i++) {
        int neighborId = neighbors.node(i);
        float score = similarityFunction.compare(vectors.getVector(neighborId), queryVec);
        NeighborArray nList = graph.getNeighbors(neighborId, layer);
        nList.insert(nodeId, score);

        // Prune if over limit
        if (nList.size() > maxConn) {
          NeighborArray pruned =
              NeighborSelector.selectDiverse(nList, maxConn, vectors, similarityFunction);
          nList.copyFrom(pruned);
        }
      }

      // Entry points for next (lower) layer = top results from this layer
      entryPoints = topNodes(searchResults, Math.min(efConstruction, searchResults.size()));
    }

    // Update entry point if this node has a higher level
    if (level > epLevel) {
      graph.setEntryNode(nodeId, level);
    }
  }

  /**
   * Greedy search: starting from entryPoint, walk to the neighbor with the best score. Repeat until
   * no improvement.
   */
  private int greedySearch(float[] query, int entryPoint, int layer) {
    int current = entryPoint;
    float currentScore = similarityFunction.compare(query, vectors.getVector(current));

    boolean improved = true;
    while (improved) {
      improved = false;
      NeighborArray neighbors = graph.getNeighbors(current, layer);
      if (neighbors == null) break;

      for (int i = 0; i < neighbors.size(); i++) {
        int neighborId = neighbors.node(i);
        float score = similarityFunction.compare(query, vectors.getVector(neighborId));
        if (score > currentScore) {
          current = neighborId;
          currentScore = score;
          improved = true;
        }
      }
    }
    return current;
  }

  /**
   * Beam search at a given layer. Uses candidate max-heap and result min-heap with visited bitset.
   *
   * @param query the query vector
   * @param entryPoints starting node(s)
   * @param ef beam width
   * @param layer the graph layer to search
   * @return a NeighborArray of results sorted by score descending
   */
  NeighborArray searchLayer(float[] query, int[] entryPoints, int ef, int layer) {
    visited.clear();
    candidates.clear();
    results.clear();

    // Seed with entry points
    for (int ep : entryPoints) {
      if (!visited.get(ep)) {
        visited.set(ep);
        float score = similarityFunction.compare(query, vectors.getVector(ep));
        candidates.add(ep, score);
        results.add(ep, score);
      }
    }

    // Beam search
    while (!candidates.isEmpty()) {
      long topCandidate = candidates.poll();
      float candidateScore = NodeQueue.score(topCandidate);
      int candidateId = NodeQueue.nodeId(topCandidate);

      // Early termination: if best candidate is worse than worst result, stop
      if (results.size() >= ef) {
        float worstResult = NodeQueue.score(results.peek());
        if (candidateScore < worstResult) {
          break;
        }
      }

      NeighborArray neighbors = graph.getNeighbors(candidateId, layer);
      if (neighbors == null) continue;

      for (int i = 0; i < neighbors.size(); i++) {
        int neighborId = neighbors.node(i);
        if (visited.get(neighborId)) continue;
        visited.set(neighborId);

        float score = similarityFunction.compare(query, vectors.getVector(neighborId));

        if (results.size() < ef) {
          candidates.add(neighborId, score);
          results.add(neighborId, score);
        } else {
          float worstResult = NodeQueue.score(results.peek());
          if (score > worstResult) {
            candidates.add(neighborId, score);
            results.poll(); // evict worst
            results.add(neighborId, score);
          }
        }
      }
    }

    // Convert results min-heap to sorted NeighborArray (descending)
    int resultSize = results.size();
    var resultArray = new NeighborArray(resultSize == 0 ? 1 : resultSize);
    // Drain min-heap (worst first) then reverse
    int[] tmpNodes = new int[resultSize];
    float[] tmpScores = new float[resultSize];
    for (int i = resultSize - 1; i >= 0; i--) {
      long entry = results.poll();
      tmpNodes[i] = NodeQueue.nodeId(entry);
      tmpScores[i] = NodeQueue.score(entry);
    }
    for (int i = 0; i < resultSize; i++) {
      resultArray.insert(tmpNodes[i], tmpScores[i]);
    }

    return resultArray;
  }

  /** Extracts the top-n node IDs from a NeighborArray (best-first). */
  private int[] topNodes(NeighborArray array, int n) {
    int count = Math.min(n, array.size());
    int[] result = new int[count];
    for (int i = 0; i < count; i++) {
      result[i] = array.node(i);
    }
    return result;
  }
}
