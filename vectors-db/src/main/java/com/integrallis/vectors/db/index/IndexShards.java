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
package com.integrallis.vectors.db.index;

import com.integrallis.vectors.core.SimilarityFunction;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.IntPredicate;

/**
 * Composition wrapper that partitions the corpus across {@code N} {@link IndexSpi} shards and
 * merges their per-shard results into a global top-{@code k}.
 *
 * <p><b>Partitioning.</b> On {@link #build}, the input vector matrix is split into {@code N}
 * contiguous ordinal ranges. Shard {@code i} holds ordinals {@code [offsets[i], offsets[i+1])}; at
 * search time, shard-local ordinals are offset back to the global range before merging, so callers
 * always receive global ordinals. Modelled on FAISS {@code IndexShards}.
 *
 * <p><b>Concurrency.</b> Each {@link #search} fans out one virtual-thread task per shard and joins
 * in order. {@link #build} is sequential per-shard to match the single-threaded build contract of
 * each underlying SPI.
 */
public final class IndexShards implements IndexSpi {

  private final List<IndexSpi> shards;
  private int[] offsets;
  private int totalSize;

  /**
   * @param shards non-empty list of SPIs that will each hold a contiguous ordinal range
   */
  public IndexShards(List<IndexSpi> shards) {
    Objects.requireNonNull(shards, "shards must not be null");
    if (shards.isEmpty()) {
      throw new IllegalArgumentException("shards must not be empty");
    }
    for (int i = 0; i < shards.size(); i++) {
      if (shards.get(i) == null) {
        throw new IllegalArgumentException("shard " + i + " must not be null");
      }
    }
    this.shards = List.copyOf(shards);
    this.offsets = new int[this.shards.size() + 1];
  }

  @Override
  public void build(float[][] vectors, SimilarityFunction metric) {
    Objects.requireNonNull(vectors, "vectors must not be null");
    Objects.requireNonNull(metric, "metric must not be null");
    int n = vectors.length;
    int s = shards.size();
    offsets = new int[s + 1];
    for (int i = 0; i <= s; i++) {
      offsets[i] = (int) ((long) n * i / s);
    }
    for (int i = 0; i < s; i++) {
      int from = offsets[i];
      int to = offsets[i + 1];
      float[][] slice = new float[to - from][];
      System.arraycopy(vectors, from, slice, 0, to - from);
      shards.get(i).build(slice, metric);
    }
    this.totalSize = n;
  }

  @Override
  public SearchOutcome search(float[] query, int k, int searchListSize, float overQueryFactor) {
    return merge(fanOut(s -> s.search(query, k, searchListSize, overQueryFactor)), k);
  }

  @Override
  public SearchOutcome search(
      float[] query, int k, int searchListSize, float overQueryFactor, int searchMultiStart) {
    return merge(
        fanOut(s -> s.search(query, k, searchListSize, overQueryFactor, searchMultiStart)), k);
  }

  @Override
  public SearchOutcome searchWithPredicate(
      float[] query, int k, int searchListSize, float overQueryFactor, IntPredicate predicate) {
    // Wrap the predicate to translate shard-local ordinals to global ordinals before the caller's
    // predicate sees them, preserving the global-ordinal contract on the filter side too.
    List<SearchOutcome> partials = new ArrayList<>(shards.size());
    try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
      List<Future<SearchOutcome>> futs = new ArrayList<>(shards.size());
      for (int i = 0; i < shards.size(); i++) {
        final int shardIdx = i;
        final int offset = offsets[shardIdx];
        IndexSpi shard = shards.get(i);
        futs.add(
            executor.submit(
                () ->
                    shard.searchWithPredicate(
                        query,
                        k,
                        searchListSize,
                        overQueryFactor,
                        localOrd -> predicate.test(localOrd + offset))));
      }
      for (Future<SearchOutcome> f : futs) {
        partials.add(f.get());
      }
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new RuntimeException("IndexShards.searchWithPredicate interrupted", e);
    } catch (ExecutionException e) {
      throw new RuntimeException("IndexShards.searchWithPredicate shard failed", e.getCause());
    }
    return mergeWithOffsets(partials, k);
  }

  @Override
  public int size() {
    return totalSize;
  }

  @Override
  public void close() {
    for (IndexSpi s : shards) {
      s.close();
    }
  }

  private interface ShardCall {
    SearchOutcome call(IndexSpi shard);
  }

  private List<SearchOutcome> fanOut(ShardCall call) {
    List<SearchOutcome> out = new ArrayList<>(shards.size());
    try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
      List<Future<SearchOutcome>> futs = new ArrayList<>(shards.size());
      for (IndexSpi s : shards) {
        futs.add(executor.submit(() -> call.call(s)));
      }
      for (Future<SearchOutcome> f : futs) {
        out.add(f.get());
      }
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new RuntimeException("IndexShards.search interrupted", e);
    } catch (ExecutionException e) {
      throw new RuntimeException("IndexShards.search shard failed", e.getCause());
    }
    return out;
  }

  private SearchOutcome merge(List<SearchOutcome> partials, int k) {
    return mergeWithOffsets(partials, k);
  }

  private SearchOutcome mergeWithOffsets(List<SearchOutcome> partials, int k) {
    int total = 0;
    for (SearchOutcome p : partials) total += p.ordinals().length;
    if (total == 0) {
      return new SearchOutcome(new int[0], new float[0]);
    }
    int[] ids = new int[total];
    float[] scores = new float[total];
    int cursor = 0;
    for (int i = 0; i < partials.size(); i++) {
      SearchOutcome p = partials.get(i);
      int off = offsets[i];
      for (int j = 0; j < p.ordinals().length; j++) {
        ids[cursor] = p.ordinals()[j] + off;
        scores[cursor] = p.scores()[j];
        cursor++;
      }
    }
    int finalK = Math.min(k, total);
    // Partial selection sort: top-k by score descending.
    for (int i = 0; i < finalK; i++) {
      int bestIdx = i;
      float bestScore = scores[i];
      for (int j = i + 1; j < total; j++) {
        if (scores[j] > bestScore) {
          bestScore = scores[j];
          bestIdx = j;
        }
      }
      if (bestIdx != i) {
        float ts = scores[i];
        int ti = ids[i];
        scores[i] = scores[bestIdx];
        ids[i] = ids[bestIdx];
        scores[bestIdx] = ts;
        ids[bestIdx] = ti;
      }
    }
    int[] outIds = new int[finalK];
    float[] outScores = new float[finalK];
    System.arraycopy(ids, 0, outIds, 0, finalK);
    System.arraycopy(scores, 0, outScores, 0, finalK);
    return new SearchOutcome(outIds, outScores);
  }
}
