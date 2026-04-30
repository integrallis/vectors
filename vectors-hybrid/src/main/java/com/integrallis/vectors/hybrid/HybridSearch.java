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
package com.integrallis.vectors.hybrid;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Orchestrates multiple retrievers in parallel (via virtual threads) and fuses the results.
 *
 * <p>Usage:
 *
 * <pre>{@code
 * var search = new HybridSearch(new RRFFusion(), vectorRetriever, textRetriever);
 * List<ScoredId> results = search.search(10);
 * }</pre>
 */
public final class HybridSearch {

  private final FusionStrategy fusion;
  private final List<Retriever> retrievers;

  /**
   * Creates a hybrid search orchestrator.
   *
   * @param fusion the fusion strategy to apply
   * @param retrievers at least one retriever
   */
  public HybridSearch(FusionStrategy fusion, Retriever... retrievers) {
    Objects.requireNonNull(fusion, "fusion");
    Objects.requireNonNull(retrievers, "retrievers");
    if (retrievers.length == 0) {
      throw new IllegalArgumentException("at least one retriever is required");
    }
    this.fusion = fusion;
    this.retrievers = List.of(retrievers);
  }

  /**
   * Run all retrievers in parallel and fuse the results.
   *
   * @param k maximum number of results to return
   * @return fused ranked list (descending by score)
   */
  public List<ScoredId> search(int k) {
    if (retrievers.size() == 1) {
      return retrievers.getFirst().retrieve(k);
    }

    List<List<ScoredId>> allResults = new ArrayList<>(retrievers.size());
    try (ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor()) {
      List<Future<List<ScoredId>>> futures = new ArrayList<>(retrievers.size());
      for (Retriever retriever : retrievers) {
        futures.add(executor.submit(() -> retriever.retrieve(k)));
      }
      for (Future<List<ScoredId>> future : futures) {
        allResults.add(future.get());
      }
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      return List.of();
    } catch (ExecutionException e) {
      throw new RuntimeException("retriever failed", e.getCause());
    }

    return fusion.fuse(allResults, k);
  }
}
