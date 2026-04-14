package com.integrallis.vectors.ivf;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.core.VectorUtil;
import com.integrallis.vectors.core.cluster.CentroidIndex;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.PriorityQueue;

/**
 * Hierarchical IVF routing index implementing the Buoy / SubBuoy split design.
 *
 * <p>A {@code SubBuoyTree} has a root {@link BuoyIndex} that partitions vectors into K clusters.
 * Any cluster whose size exceeds the {@link ClusterSplitter} threshold is split via bisecting
 * K-Means into two child partitions; those children are stored as leaf {@link ClusterPartition}s
 * inside a lightweight {@code SplitNode}. The resulting tree is at most two levels deep for Phase
 * 2a (deeper splitting is deferred to a future phase).
 *
 * <p>Search algorithm:
 *
 * <ol>
 *   <li>Route the query through the root BuoyIndex to select {@code nprobe} candidate clusters.
 *   <li>For each selected cluster:
 *       <ul>
 *         <li><em>Leaf</em>: brute-force scan the cluster's ordinals.
 *         <li><em>Split node</em>: scan <em>both</em> child partitions (the split already reduced
 *             cluster size by ~½, so total work ≈ original size).
 *       </ul>
 *   <li>Merge all candidate hits into a top-k result.
 * </ol>
 */
public final class SubBuoyTree {

  /** Represents a cluster that has been bisected; holds a centroid router plus two children. */
  private record SplitNode(CentroidIndex subIndex, ClusterPartition left, ClusterPartition right) {}

  private final BuoyIndex rootBuoy;
  private final ClusterPartition[] leafPartitions; // null entry = this cluster was split
  private final SplitNode[] splitNodes; // null entry = this cluster is a leaf
  private final float[][] allVectors;
  private final String[] ids;
  private final SimilarityFunction metric;

  private SubBuoyTree(
      BuoyIndex rootBuoy,
      ClusterPartition[] leafPartitions,
      SplitNode[] splitNodes,
      float[][] allVectors,
      String[] ids,
      SimilarityFunction metric) {
    this.rootBuoy = rootBuoy;
    this.leafPartitions = leafPartitions;
    this.splitNodes = splitNodes;
    this.allVectors = allVectors;
    this.ids = ids;
    this.metric = metric;
  }

  /** Builds a {@code SubBuoyTree} from {@code vectors}, applying splitting where warranted. */
  public static SubBuoyTree build(
      float[][] vectors,
      String[] ids,
      SimilarityFunction metric,
      IvfBuildParams params,
      ClusterSplitter splitter) {
    if (vectors.length == 0) throw new IllegalArgumentException("empty vector set");
    int n = vectors.length;
    int k = Math.min(params.k(), n);

    // Step 1: train root BuoyIndex
    BuoyIndex rootBuoy = BuoyIndex.train(vectors, k, metric, params.buildSoar(), params.seed());

    // Step 2: assign vectors to root clusters
    int[] assignments = new int[n];
    for (int i = 0; i < n; i++) assignments[i] = rootBuoy.route(vectors[i], 1, 0f)[0];

    List<List<Integer>> lists = new ArrayList<>(k);
    for (int c = 0; c < k; c++) lists.add(new ArrayList<>());
    for (int i = 0; i < n; i++) lists.get(assignments[i]).add(i);

    float[][] buoys = rootBuoy.buoyVectors();
    ClusterPartition[] leafPartitions = new ClusterPartition[k];
    SplitNode[] splitNodes = new SplitNode[k];

    // Step 3: for each cluster, either keep as leaf or split
    for (int c = 0; c < k; c++) {
      int[] ordinals = lists.get(c).stream().mapToInt(Integer::intValue).toArray();
      ClusterPartition partition = new ClusterPartition(c, buoys[c], ordinals, ordinals.length);

      if (splitter.shouldSplit(partition)) {
        Optional<ClusterPartition[]> split = splitter.split(partition, vectors, metric, (long) c);
        if (split.isPresent()) {
          ClusterPartition[] children = split.get();
          float[][] childCentroids = {children[0].centroid(), children[1].centroid()};
          CentroidIndex subIdx = new CentroidIndex(childCentroids, metric);
          splitNodes[c] = new SplitNode(subIdx, children[0], children[1]);
        } else {
          leafPartitions[c] = partition;
        }
      } else {
        leafPartitions[c] = partition;
      }
    }

    return new SubBuoyTree(rootBuoy, leafPartitions, splitNodes, vectors, ids, metric);
  }

  /** Searches for the top-{@code request.k()} nearest neighbours of {@code request.query()}. */
  public IvfSearchResult search(IvfSearchRequest request) {
    float[] query = request.query();
    int k = request.k();
    int nprobe = Math.min(request.nprobe(), rootBuoy.k());
    int[] clusterIds = rootBuoy.route(query, nprobe, request.gamma());

    PriorityQueue<IvfHit> heap =
        new PriorityQueue<>(k + 1, (a, b) -> Float.compare(a.score(), b.score()));
    int clustersSearched = 0;

    for (int cid : clusterIds) {
      if (splitNodes[cid] != null) {
        // Split node: scan both children
        SplitNode node = splitNodes[cid];
        scanPartition(node.left(), query, k, request.minScore(), heap, ids);
        scanPartition(node.right(), query, k, request.minScore(), heap, ids);
        clustersSearched += 2;
      } else if (leafPartitions[cid] != null) {
        scanPartition(leafPartitions[cid], query, k, request.minScore(), heap, ids);
        clustersSearched++;
      }
    }

    IvfHit[] arr = heap.toArray(new IvfHit[0]);
    Arrays.sort(arr);
    return new IvfSearchResult(List.of(arr), clustersSearched);
  }

  /** Number of clusters in the root BuoyIndex. */
  public int rootClusterCount() {
    return rootBuoy.k();
  }

  /** Total number of leaf partitions (unsplit clusters + 2 children per split cluster). */
  public int leafCount() {
    int count = 0;
    for (int c = 0; c < rootBuoy.k(); c++) {
      if (splitNodes[c] != null) count += 2;
      else count++;
    }
    return count;
  }

  /** Tree depth: 1 if no clusters were split, 2 if any cluster was split. */
  public int depth() {
    for (SplitNode sn : splitNodes) if (sn != null) return 2;
    return 1;
  }

  // ─── internals ─────────────────────────────────────────────────────────────

  private void scanPartition(
      ClusterPartition p,
      float[] query,
      int k,
      float minScore,
      PriorityQueue<IvfHit> heap,
      String[] ids) {
    for (int ordinal : p.ordinals()) {
      float score = score(query, allVectors[ordinal]);
      if (score < minScore) continue;
      if (heap.size() < k) {
        heap.offer(new IvfHit(ordinal, ids != null ? ids[ordinal] : null, score));
      } else if (score > heap.peek().score()) {
        heap.poll();
        heap.offer(new IvfHit(ordinal, ids != null ? ids[ordinal] : null, score));
      }
    }
  }

  private float score(float[] query, float[] vector) {
    return switch (metric) {
      case EUCLIDEAN -> -VectorUtil.squareDistance(query, vector);
      case DOT_PRODUCT, MAXIMUM_INNER_PRODUCT -> VectorUtil.dotProduct(query, vector);
      case COSINE -> VectorUtil.cosine(query, vector);
    };
  }
}
