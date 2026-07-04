/*
 * Copyright 2025-2026 Integrallis Software, LLC
 *
 * Licensed under the Functional Source License, Version 1.1, Apache 2.0 Future License
 * (the "License"); you may not use this file except in compliance with the License.
 *
 *     https://fsl.software/FSL-1.1-ALv2.txt
 *
 * Unless required by applicable law or agreed to in writing, software distributed under
 * the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
 * ANY KIND, either express or implied. See the License for the specific language
 * governing permissions and limitations under the License.
 *
 * Change Date: April 25, 2028
 * Change License: Apache License, Version 2.0
 */
package com.integrallis.vectors.cluster;

import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.core.filter.Filter;
import com.integrallis.vectors.db.SearchRequest;
import com.integrallis.vectors.db.SearchResult;
import com.integrallis.vectors.db.VectorCollection;
import com.integrallis.vectors.db.VectorCollectionConfig;
import com.integrallis.vectors.distributed.TopKMerger;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A sharded write tier (ROADMAP P3.2): a {@link VectorCollection} facade over {@code N} independent
 * {@code vectors-db} shards, with documents hash-partitioned across the shards by id.
 *
 * <p><b>Write path.</b> {@code add}, {@code upsert}, {@code delete}, and {@code get}/{@code
 * contains} are routed by {@link ConsistentHashRouter#route(String)} to the single shard that owns
 * the document id. {@code addAll} groups the batch by owning shard and forwards one batch per
 * shard, preserving the underlying batch-ingest path. Because each document id lives on exactly one
 * shard, writes scale with the number of shards with no cross-shard coordination.
 *
 * <p><b>Read path.</b> {@code search} scatter-gathers the <em>full</em> {@link SearchRequest}
 * (filter and projection flags intact) to every shard in parallel on virtual threads, then merges
 * the per-shard results with {@link TopKMerger} into a single global top-k. {@code deleteWhere}
 * fans out to every shard, since a metadata predicate may match documents on any shard.
 *
 * <p><b>No consensus.</b> Shards are fully independent. {@code commit}, {@code flush}, and {@code
 * compact} fan out to every shard but each shard commits atomically on its own — there is no
 * cross-shard atomic commit. A crash mid-fan-out can leave shards at different generations; each
 * shard still recovers to its own last durable generation. This is the deliberate Phase-3 trade-off
 * (availability and horizontal write throughput over a global transaction boundary).
 *
 * <p><b>Read availability.</b> If a shard times out or throws during {@code search}, it is skipped
 * with a warning and the merge proceeds over the responding shards (partial results), mirroring the
 * scatter-gather semantics of {@code vectors-distributed}. Size and document enumeration, by
 * contrast, propagate shard failures rather than under-report silently.
 *
 * <p>All shards must share the same {@link VectorCollectionConfig#dimension() dimension} and {@link
 * VectorCollectionConfig#metric() metric}; the builder validates this.
 */
public final class ClusterVectorCollection implements VectorCollection {

  private static final Logger LOG = LoggerFactory.getLogger(ClusterVectorCollection.class);

  /** Default per-query deadline applied across all shards during a scatter-gather search. */
  public static final Duration DEFAULT_READ_TIMEOUT = Duration.ofSeconds(5);

  private final List<VectorCollection> shards;
  private final ConsistentHashRouter router;
  private final Duration readTimeout;

  private ClusterVectorCollection(
      List<VectorCollection> shards, ConsistentHashRouter router, Duration readTimeout) {
    this.shards = List.copyOf(shards);
    this.router = router;
    this.readTimeout = readTimeout;
  }

  /** Returns the number of shards this cluster spans. */
  public int shardCount() {
    return shards.size();
  }

  /** Returns the consistent-hash router used to map document ids to shards. */
  public ConsistentHashRouter router() {
    return router;
  }

  private VectorCollection shardFor(String documentId) {
    return shards.get(router.route(documentId));
  }

  // ---- Write path: route by document id ----

  @Override
  public void add(Document doc) {
    shardFor(doc.id()).add(doc);
  }

  @Override
  public void addAll(Collection<Document> docs) {
    Map<Integer, List<Document>> byShard = new HashMap<>();
    for (Document doc : docs) {
      byShard.computeIfAbsent(router.route(doc.id()), k -> new ArrayList<>()).add(doc);
    }
    byShard.forEach((shardIndex, group) -> shards.get(shardIndex).addAll(group));
  }

  @Override
  public void upsert(Document doc) {
    shardFor(doc.id()).upsert(doc);
  }

  @Override
  public boolean delete(String id) {
    return shardFor(id).delete(id);
  }

  /** Fans out to every shard, since a metadata predicate may match documents on any shard. */
  @Override
  public int deleteWhere(Filter filter) {
    int total = 0;
    for (VectorCollection shard : shards) {
      total += shard.deleteWhere(filter);
    }
    return total;
  }

  // ---- Commit / durability: fan out (no cross-shard atomicity) ----

  @Override
  public void commit() {
    for (VectorCollection shard : shards) {
      shard.commit();
    }
  }

  @Override
  public void flush() {
    for (VectorCollection shard : shards) {
      shard.flush();
    }
  }

  @Override
  public boolean refresh() {
    boolean changed = false;
    for (VectorCollection shard : shards) {
      changed |= shard.refresh();
    }
    return changed;
  }

  @Override
  public void compact() {
    for (VectorCollection shard : shards) {
      shard.compact();
    }
  }

  // ---- Read path ----

  @Override
  public SearchResult search(SearchRequest request) {
    Objects.requireNonNull(request, "request must not be null");
    // Degenerate single-shard cluster: pass through for exact, allocation-free parity.
    if (shards.size() == 1) {
      return shards.get(0).search(request);
    }
    long start = System.nanoTime();
    long deadline = start + readTimeout.toNanos();
    List<SearchResult> partials = new ArrayList<>(shards.size());
    try (ExecutorService exec = Executors.newVirtualThreadPerTaskExecutor()) {
      List<Future<SearchResult>> futures = new ArrayList<>(shards.size());
      for (VectorCollection shard : shards) {
        futures.add(exec.submit(() -> shard.search(request)));
      }
      for (int i = 0; i < futures.size(); i++) {
        Future<SearchResult> future = futures.get(i);
        try {
          long remaining = deadline - System.nanoTime();
          partials.add(future.get(Math.max(0, remaining), TimeUnit.NANOSECONDS));
        } catch (TimeoutException | ExecutionException e) {
          LOG.warn("shard {} failed or timed out during search; returning partial results", i, e);
          future.cancel(true);
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
          future.cancel(true);
          break;
        }
      }
    }
    long elapsed = System.nanoTime() - start;
    // Each document is routed to exactly one shard, so per-shard result id spaces are disjoint:
    // merge without the de-duplication HashMap pass (dedup=false fast path).
    return TopKMerger.merge(partials, request.k(), elapsed, false);
  }

  @Override
  public Document get(String id) {
    return shardFor(id).get(id);
  }

  @Override
  public boolean contains(String id) {
    return shardFor(id).contains(id);
  }

  /** Concatenates the live documents of every shard. */
  @Override
  public List<Document> documents() {
    List<Document> all = new ArrayList<>();
    for (VectorCollection shard : shards) {
      all.addAll(shard.documents());
    }
    return all;
  }

  /** Aggregates the logical (live) size across all shards. */
  @Override
  public int size() {
    int total = 0;
    for (VectorCollection shard : shards) {
      total += shard.size();
    }
    return total;
  }

  /** Aggregates the physical (live + tombstoned) size across all shards. */
  @Override
  public int physicalSize() {
    int total = 0;
    for (VectorCollection shard : shards) {
      total += shard.physicalSize();
    }
    return total;
  }

  /** Returns the representative configuration of shard 0; all shards share dimension and metric. */
  @Override
  public VectorCollectionConfig config() {
    return shards.get(0).config();
  }

  /** Closes every shard, propagating the first failure after attempting to close them all. */
  @Override
  public void close() {
    RuntimeException first = null;
    for (VectorCollection shard : shards) {
      try {
        shard.close();
      } catch (RuntimeException e) {
        if (first == null) {
          first = e;
        } else {
          first.addSuppressed(e);
        }
      }
    }
    if (first != null) {
      throw first;
    }
  }

  // ---- Construction ----

  /** Creates a builder for a storage-backed or factory-driven sharded cluster. */
  public static Builder builder() {
    return new Builder();
  }

  /**
   * Wraps an explicit, ordered list of pre-built shard collections with the default read timeout.
   * The list order defines shard indices; the same list (same order) must be supplied on every
   * reopen so document ids route to the same shards.
   *
   * @param shards the shard collections; must be non-empty and share dimension and metric
   */
  public static ClusterVectorCollection over(List<VectorCollection> shards) {
    return over(shards, DEFAULT_READ_TIMEOUT);
  }

  /**
   * Wraps an explicit, ordered list of pre-built shard collections.
   *
   * @param shards the shard collections; must be non-empty and share dimension and metric
   * @param readTimeout per-query scatter-gather deadline
   */
  public static ClusterVectorCollection over(List<VectorCollection> shards, Duration readTimeout) {
    Objects.requireNonNull(shards, "shards must not be null");
    Objects.requireNonNull(readTimeout, "readTimeout must not be null");
    if (shards.isEmpty()) {
      throw new IllegalArgumentException("shards must not be empty");
    }
    validateHomogeneous(shards);
    return new ClusterVectorCollection(
        shards, new ConsistentHashRouter(shards.size()), readTimeout);
  }

  private static void validateHomogeneous(List<VectorCollection> shards) {
    VectorCollectionConfig head = shards.get(0).config();
    for (int i = 1; i < shards.size(); i++) {
      VectorCollectionConfig c = shards.get(i).config();
      if (c.dimension() != head.dimension() || c.metric() != head.metric()) {
        throw new IllegalArgumentException(
            "all shards must share dimension and metric: shard 0 is "
                + head.dimension()
                + "/"
                + head.metric()
                + " but shard "
                + i
                + " is "
                + c.dimension()
                + "/"
                + c.metric());
      }
    }
  }

  /** Fluent builder for {@link ClusterVectorCollection}. */
  public static final class Builder {

    private int shardCount;
    private Path storageRoot;
    private ShardFactory shardFactory;
    private int virtualNodesPerShard = ConsistentHashRouter.DEFAULT_VIRTUAL_NODES_PER_SHARD;
    private Duration readTimeout = DEFAULT_READ_TIMEOUT;

    private Builder() {}

    /** Sets the number of shards to create. Must be positive. */
    public Builder shardCount(int shardCount) {
      if (shardCount <= 0) {
        throw new IllegalArgumentException("shardCount must be positive: " + shardCount);
      }
      this.shardCount = shardCount;
      return this;
    }

    /**
     * Sets the cluster storage root. Each shard is given {@code <storageRoot>/shard-<index>} (made
     * absolute) as its per-shard directory. Pass {@code null} (the default) for in-memory shards,
     * in which case the factory receives a {@code null} shard path.
     */
    public Builder storageRoot(Path storageRoot) {
      this.storageRoot = storageRoot;
      return this;
    }

    /** Sets the factory invoked once per shard to build its backing collection. Required. */
    public Builder shardFactory(ShardFactory shardFactory) {
      this.shardFactory = Objects.requireNonNull(shardFactory, "shardFactory must not be null");
      return this;
    }

    /** Overrides the virtual-node count per shard on the consistent-hash ring. Must be positive. */
    public Builder virtualNodesPerShard(int virtualNodesPerShard) {
      if (virtualNodesPerShard <= 0) {
        throw new IllegalArgumentException(
            "virtualNodesPerShard must be positive: " + virtualNodesPerShard);
      }
      this.virtualNodesPerShard = virtualNodesPerShard;
      return this;
    }

    /** Sets the per-query scatter-gather deadline. Default: {@link #DEFAULT_READ_TIMEOUT}. */
    public Builder readTimeout(Duration readTimeout) {
      this.readTimeout = Objects.requireNonNull(readTimeout, "readTimeout must not be null");
      return this;
    }

    /** Builds the cluster, creating one collection per shard via the configured factory. */
    public ClusterVectorCollection build() {
      if (shardCount <= 0) {
        throw new IllegalStateException("shardCount is required, call builder.shardCount(n)");
      }
      Objects.requireNonNull(
          shardFactory, "shardFactory is required, call builder.shardFactory(f)");
      Path root = storageRoot == null ? null : storageRoot.toAbsolutePath();
      List<VectorCollection> shards = new ArrayList<>(shardCount);
      try {
        for (int i = 0; i < shardCount; i++) {
          Path shardPath = root == null ? null : root.resolve("shard-" + i);
          VectorCollection shard =
              Objects.requireNonNull(
                  shardFactory.create(i, shardPath), "shardFactory returned null for shard " + i);
          shards.add(shard);
        }
      } catch (RuntimeException e) {
        // Avoid leaking already-built shards if a later one fails to construct.
        for (VectorCollection built : shards) {
          try {
            built.close();
          } catch (RuntimeException suppressed) {
            e.addSuppressed(suppressed);
          }
        }
        throw e;
      }
      validateHomogeneous(shards);
      return new ClusterVectorCollection(
          shards, new ConsistentHashRouter(shardCount, virtualNodesPerShard), readTimeout);
    }
  }
}
