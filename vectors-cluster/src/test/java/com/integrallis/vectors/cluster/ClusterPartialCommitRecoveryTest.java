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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.core.filter.Filter;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.SearchRequest;
import com.integrallis.vectors.db.SearchResult;
import com.integrallis.vectors.db.VectorCollection;
import com.integrallis.vectors.db.VectorCollectionConfig;
import java.util.Collection;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Pins {@link ClusterVectorCollection}'s documented "no cross-shard atomic commit" semantics. When
 * shard <i>i</i>'s commit throws, the fan-out stops; shards 0..i-1 are already committed, shard i
 * is partial, and shards i+1..N have not been attempted. A subsequent retry must succeed.
 *
 * <p>Audit T3.9: this contract was documented in the cluster Javadoc but had no regression test — a
 * future refactor that "helpfully" caught and swallowed mid-fan-out exceptions would silently lose
 * data without any test failing.
 */
@Tag("unit")
class ClusterPartialCommitRecoveryTest {

  private static final int DIM = 8;

  private static VectorCollection flatInMemory() {
    return VectorCollection.builder()
        .dimension(DIM)
        .metric(SimilarityFunction.DOT_PRODUCT)
        .indexType(IndexType.FLAT)
        .build();
  }

  private static float[] unitVec(Random rng) {
    float[] v = new float[DIM];
    double norm = 0;
    for (int i = 0; i < DIM; i++) {
      v[i] = rng.nextFloat() * 2f - 1f;
      norm += v[i] * v[i];
    }
    float n = (float) Math.sqrt(norm);
    if (n == 0f) v[0] = 1f;
    else for (int i = 0; i < DIM; i++) v[i] /= n;
    return v;
  }

  @Test
  void midFanOutCommitFailurePropagatesAndRetrySucceeds() {
    Random rng = new Random(0xC0FFEEL);
    VectorCollection shard0 = flatInMemory();
    FailOnDemandCollection shard1 = new FailOnDemandCollection(flatInMemory());
    VectorCollection shard2 = flatInMemory();
    ClusterVectorCollection cluster = ClusterVectorCollection.over(List.of(shard0, shard1, shard2));

    int total = 60;
    for (int i = 0; i < total; i++) {
      cluster.add(Document.of("doc-" + i, unitVec(rng)));
    }
    // Sanity: docs are staged but not yet committed; per-shard live size is zero.
    assertThat(shard0.size()).isZero();
    assertThat(shard1.size()).isZero();
    assertThat(shard2.size()).isZero();

    // Arm the failure injection on shard 1 only.
    shard1.failOnNextCommit(new RuntimeException("simulated shard-commit failure"));

    // The cluster commit must propagate the failure (it must not be swallowed).
    assertThatThrownBy(cluster::commit).isInstanceOf(RuntimeException.class);

    // Mixed state: shard 0 committed before shard 1 threw; shards 1 and 2 did not advance.
    int shard0Size = shard0.size();
    assertThat(shard0Size).as("shard 0 must have committed before shard 1 threw").isPositive();
    assertThat(shard1.size())
        .as("shard 1's failed commit must leave its previous live count unchanged")
        .isZero();
    assertThat(shard2.size()).as("shard 2's commit must not have been attempted yet").isZero();

    // Total live count across shards equals shard 0's contribution. The cluster's overall size()
    // is summed across shards.
    assertThat(cluster.size()).isEqualTo(shard0Size);

    // Recovery: a clean retry must finish what the previous fan-out started.
    cluster.commit();
    assertThat(shard0.size() + shard1.size() + shard2.size())
        .as("every doc must be live across shards after the recovery commit")
        .isEqualTo(total);
    assertThat(cluster.size()).isEqualTo(total);

    // And every previously-added doc must be retrievable via the cluster.
    for (int i = 0; i < total; i++) {
      String id = "doc-" + i;
      assertThat(cluster.contains(id))
          .as("doc '%s' must be retrievable post-recovery", id)
          .isTrue();
    }

    cluster.close();
  }

  /**
   * Thin delegate that forwards every call to a real {@link VectorCollection} but can be armed to
   * throw on the next {@link #commit()} call — used to simulate a single-shard commit failure
   * mid-fan-out without monkey-patching the production code.
   */
  private static final class FailOnDemandCollection implements VectorCollection {
    private final VectorCollection delegate;
    private volatile RuntimeException pendingFailure;

    FailOnDemandCollection(VectorCollection delegate) {
      this.delegate = delegate;
    }

    void failOnNextCommit(RuntimeException failure) {
      this.pendingFailure = failure;
    }

    @Override
    public void commit() {
      RuntimeException toThrow = pendingFailure;
      if (toThrow != null) {
        // The contract under test: the underlying delegate's state stays at the prior generation
        // when commit throws. We do NOT call delegate.commit() — emulating an early failure (e.g.
        // a disk-full error caught before fsync) is sufficient for the cluster-level test.
        pendingFailure = null;
        throw toThrow;
      }
      delegate.commit();
    }

    @Override
    public void add(Document doc) {
      delegate.add(doc);
    }

    @Override
    public void addAll(Collection<Document> docs) {
      delegate.addAll(docs);
    }

    @Override
    public void upsert(Document doc) {
      delegate.upsert(doc);
    }

    @Override
    public boolean delete(String id) {
      return delegate.delete(id);
    }

    @Override
    public int deleteWhere(Filter filter) {
      return delegate.deleteWhere(filter);
    }

    @Override
    public void compact() {
      delegate.compact();
    }

    @Override
    public SearchResult search(SearchRequest request) {
      return delegate.search(request);
    }

    @Override
    public Document get(String id) {
      return delegate.get(id);
    }

    @Override
    public boolean contains(String id) {
      return delegate.contains(id);
    }

    @Override
    public List<Document> documents() {
      return delegate.documents();
    }

    @Override
    public int size() {
      return delegate.size();
    }

    @Override
    public int physicalSize() {
      return delegate.physicalSize();
    }

    @Override
    public VectorCollectionConfig config() {
      return delegate.config();
    }

    @Override
    public void close() {
      delegate.close();
    }
  }
}
