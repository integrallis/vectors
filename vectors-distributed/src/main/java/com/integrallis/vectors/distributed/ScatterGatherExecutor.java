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
package com.integrallis.vectors.distributed;

import com.integrallis.vectors.db.SearchResult;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Stateless scatter-gather executor using virtual threads (one per node per query).
 *
 * <p>Each call to {@link #execute} fans out all {@link LocalSearchRequest}s in parallel, collects
 * results within the configured timeout, merges them via {@link TopKMerger}, and returns. Nodes
 * that time out or throw are skipped with a WARNING log (partial results).
 *
 * <p>Thread-safe: a single instance may be shared across concurrent callers.
 */
public final class ScatterGatherExecutor {

  private static final Logger LOG = LoggerFactory.getLogger(ScatterGatherExecutor.class);

  private final NodeDirectory directory;
  private final long timeoutMillis;
  private final NodeCallContext context;

  /**
   * @param directory resolves NodeId → NodeSearchClient
   * @param timeout per-node call timeout; must be positive
   */
  public ScatterGatherExecutor(NodeDirectory directory, Duration timeout) {
    this(directory, timeout, NodeCallContext.none());
  }

  /**
   * @param directory resolves NodeId → NodeSearchClient
   * @param timeout per-node call timeout; must be positive
   * @param context metadata sent with each node call
   */
  public ScatterGatherExecutor(NodeDirectory directory, Duration timeout, NodeCallContext context) {
    this.directory = Objects.requireNonNull(directory, "directory must not be null");
    Objects.requireNonNull(timeout, "timeout must not be null");
    if (timeout.isNegative() || timeout.isZero()) {
      throw new IllegalArgumentException("timeout must be positive: " + timeout);
    }
    this.timeoutMillis = timeout.toMillis();
    this.context = Objects.requireNonNull(context, "context must not be null");
  }

  /**
   * Execute the plan against all target nodes concurrently and merge top-k.
   *
   * @param plan one {@link LocalSearchRequest} per target node
   * @param k number of top results to return after merging
   * @return merged {@link SearchResult}; may have fewer than k hits if nodes timed out or errored
   */
  public SearchResult execute(List<LocalSearchRequest> plan, int k) {
    Objects.requireNonNull(plan, "plan must not be null");
    if (k <= 0) throw new IllegalArgumentException("k must be positive: " + k);
    if (plan.isEmpty()) {
      return new SearchResult(List.of(), 0L);
    }

    long start = System.nanoTime();
    List<SearchResult> partials = new ArrayList<>(plan.size());

    try (ExecutorService vt = Executors.newVirtualThreadPerTaskExecutor()) {
      List<Callable<SearchResult>> tasks = new ArrayList<>(plan.size());
      for (LocalSearchRequest req : plan) {
        tasks.add(
            () -> {
              NodeSearchClient client = directory.clientFor(req.targetNode());
              return client.search(req, context);
            });
      }

      List<Future<SearchResult>> futures;
      try {
        futures = vt.invokeAll(tasks, timeoutMillis, TimeUnit.MILLISECONDS);
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        LOG.warn("ScatterGather interrupted — returning empty result");
        return new SearchResult(List.of(), System.nanoTime() - start);
      }

      for (int i = 0; i < futures.size(); i++) {
        Future<SearchResult> f = futures.get(i);
        NodeId targetNode = plan.get(i).targetNode();
        if (f.isCancelled()) {
          LOG.warn(
              "Node {} timed out after {} ms — partial results only", targetNode, timeoutMillis);
          continue;
        }
        try {
          partials.add(f.get());
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
          LOG.warn("Interrupted while collecting result from node {}", targetNode);
        } catch (ExecutionException e) {
          LOG.warn("Node {} threw an exception — skipping its results", targetNode, e.getCause());
        }
      }
    }

    long elapsed = System.nanoTime() - start;
    return TopKMerger.merge(partials, k, elapsed);
  }
}
