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
package com.integrallis.vectors.ingest.sinks;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.ingest.Batch;
import com.integrallis.vectors.ingest.EmbeddedDoc;
import com.integrallis.vectors.ingest.VectorSink;
import com.integrallis.vectors.ivf.ClusterSplitter;
import com.integrallis.vectors.ivf.DistributedVectorCollection;
import com.integrallis.vectors.ivf.IvfBuildParams;
import com.integrallis.vectors.ivf.TierPolicy;
import com.integrallis.vectors.storage.backend.StorageBackend;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * {@link VectorSink} backed by a {@link DistributedVectorCollection}. Two factories cover the
 * fresh-build and reopen lifecycles:
 *
 * <ul>
 *   <li>{@link #bootstrapping bootstrapping(...)} — buffers the first batch, then calls {@link
 *       DistributedVectorCollection#build build(...)} to train the routing index. Subsequent
 *       batches use {@code add()/commit()}.
 *   <li>{@link #opening opening(...)} — reopens an existing collection via {@link
 *       DistributedVectorCollection#open open(...)} and uses {@code add()/commit()} from the start.
 * </ul>
 *
 * <p>The sink optionally seals (rotates) the WAL every {@code walSealEveryNBatches} commits so
 * segment sizes stay bounded for predictable replay cost. Set to {@code 0} to disable rotation.
 */
public final class DistributedVectorSink implements VectorSink {

  /** Bootstrapping factory: first batch trains the IVF index, then add()/commit() takes over. */
  public static DistributedVectorSink bootstrapping(
      Path walDir,
      StorageBackend t3Backend,
      IvfBuildParams params,
      ClusterSplitter splitter,
      TierPolicy tierPolicy,
      SimilarityFunction metric) {
    return new DistributedVectorSink(walDir, t3Backend, params, splitter, tierPolicy, metric, 0);
  }

  /** Bootstrapping factory with WAL rotation cadence (in batches; 0 disables). */
  public static DistributedVectorSink bootstrapping(
      Path walDir,
      StorageBackend t3Backend,
      IvfBuildParams params,
      ClusterSplitter splitter,
      TierPolicy tierPolicy,
      SimilarityFunction metric,
      int walSealEveryNBatches) {
    return new DistributedVectorSink(
        walDir, t3Backend, params, splitter, tierPolicy, metric, walSealEveryNBatches);
  }

  /** Opening factory: reopens an existing collection and resumes incremental commits. */
  public static DistributedVectorSink opening(
      Path walDir, StorageBackend t3Backend, SimilarityFunction metric, TierPolicy tierPolicy)
      throws IOException {
    DistributedVectorCollection col =
        DistributedVectorCollection.open(walDir, metric, tierPolicy, t3Backend);
    return new DistributedVectorSink(col, 0);
  }

  /** Opening factory with rotation cadence. */
  public static DistributedVectorSink opening(
      Path walDir,
      StorageBackend t3Backend,
      SimilarityFunction metric,
      TierPolicy tierPolicy,
      int walSealEveryNBatches)
      throws IOException {
    DistributedVectorCollection col =
        DistributedVectorCollection.open(walDir, metric, tierPolicy, t3Backend);
    return new DistributedVectorSink(col, walSealEveryNBatches);
  }

  // ─── instance state ──────────────────────────────────────────────────────

  private final Path walDir;
  private final StorageBackend t3Backend;
  private final IvfBuildParams params;
  private final ClusterSplitter splitter;
  private final TierPolicy tierPolicy;
  private final SimilarityFunction metric;
  private final int walSealEveryNBatches;
  private DistributedVectorCollection collection;
  private final List<EmbeddedDoc> bootstrapBuffer = new ArrayList<>();
  private long committedCount;
  private long batchesSinceRotation;
  private boolean closed;

  /** Bootstrapping constructor — collection is created on first commit. */
  private DistributedVectorSink(
      Path walDir,
      StorageBackend t3Backend,
      IvfBuildParams params,
      ClusterSplitter splitter,
      TierPolicy tierPolicy,
      SimilarityFunction metric,
      int walSealEveryNBatches) {
    this.walDir = Objects.requireNonNull(walDir, "walDir");
    this.t3Backend = Objects.requireNonNull(t3Backend, "t3Backend");
    this.params = Objects.requireNonNull(params, "params");
    this.splitter = Objects.requireNonNull(splitter, "splitter");
    this.tierPolicy = Objects.requireNonNull(tierPolicy, "tierPolicy");
    this.metric = Objects.requireNonNull(metric, "metric");
    if (walSealEveryNBatches < 0) {
      throw new IllegalArgumentException("walSealEveryNBatches must be >= 0");
    }
    this.walSealEveryNBatches = walSealEveryNBatches;
    this.collection = null;
  }

  /** Opening constructor — collection is already live. */
  private DistributedVectorSink(DistributedVectorCollection col, int walSealEveryNBatches) {
    this.collection = Objects.requireNonNull(col, "collection");
    this.walDir = null;
    this.t3Backend = null;
    this.params = null;
    this.splitter = null;
    this.tierPolicy = null;
    this.metric = null;
    if (walSealEveryNBatches < 0) {
      throw new IllegalArgumentException("walSealEveryNBatches must be >= 0");
    }
    this.walSealEveryNBatches = walSealEveryNBatches;
    this.committedCount = col.size();
  }

  /** Visible for tests: returns the underlying collection (null while bootstrap is pending). */
  public DistributedVectorCollection collection() {
    return collection;
  }

  // ─── VectorSink ──────────────────────────────────────────────────────────

  @Override
  public void addAll(Batch batch) {
    if (closed) throw new IllegalStateException("sink is closed");
    // addAll must be idempotent: commitBatch wraps it in retryPolicy.execute, so a mid-loop failure
    // re-runs addAll(batch) from the top. Staging is never durable until commit(), so we discard
    // any
    // docs staged by a failed prior attempt before re-staging — otherwise a retry appends the
    // already-staged prefix again, duplicating those vectors in the committed generation.
    if (collection == null) {
      // Bootstrap path: replace (not append) the pending first-batch buffer.
      bootstrapBuffer.clear();
      bootstrapBuffer.addAll(batch.docs());
      return;
    }
    collection.discardStaging();
    for (EmbeddedDoc d : batch.docs()) {
      collection.add(d.doc().id(), d.vector());
    }
  }

  @Override
  public void commit() throws IOException {
    if (closed) throw new IllegalStateException("sink is closed");
    if (collection == null) {
      if (bootstrapBuffer.isEmpty()) {
        throw new IllegalStateException(
            "DistributedVectorSink.bootstrapping requires a non-empty first batch");
      }
      float[][] vectors = new float[bootstrapBuffer.size()][];
      String[] ids = new String[bootstrapBuffer.size()];
      for (int i = 0; i < bootstrapBuffer.size(); i++) {
        vectors[i] = bootstrapBuffer.get(i).vector();
        ids[i] = bootstrapBuffer.get(i).doc().id();
      }
      collection =
          DistributedVectorCollection.build(
              vectors, ids, metric, params, splitter, tierPolicy, walDir, t3Backend);
      committedCount = collection.size();
      bootstrapBuffer.clear();
    } else {
      collection.commit();
      committedCount = collection.size();
    }
    batchesSinceRotation++;
    if (walSealEveryNBatches > 0 && batchesSinceRotation >= walSealEveryNBatches) {
      collection.rotateWalSegment();
      batchesSinceRotation = 0;
    }
  }

  @Override
  public long committedCount() {
    return committedCount;
  }

  @Override
  public void close() throws IOException {
    if (closed) return;
    closed = true;
    if (collection != null) {
      collection.close();
    }
  }
}
