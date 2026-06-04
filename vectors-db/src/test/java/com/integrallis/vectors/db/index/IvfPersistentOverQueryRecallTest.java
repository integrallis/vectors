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

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.storage.MappedIvfFlatAdapter;
import com.integrallis.vectors.db.storage.MappedIvfPqAdapter;
import com.integrallis.vectors.ivf.IvfBuildParams;
import com.integrallis.vectors.ivf.IvfIndex;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import org.assertj.core.data.Offset;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Pins over-query recall behaviour for the PERSISTENT IVF adapters ({@link MappedIvfPqAdapter},
 * {@link MappedIvfFlatAdapter}) against their in-memory twins (P1.3a).
 *
 * <p>Regression guarded: both persistent adapters previously ignored {@code overQueryFactor} for
 * the probe count (IVF_PQ scaled only the rescore pool; IVF_FLAT scaled nothing), leaving
 * persistent recall flat as over-query rose — diverging from the in-memory adapters. Over-query is
 * meant to widen the search; for IVF that means probing more clusters, since recall is gated by
 * cluster coverage. These tests assert (a) persistent recall rises strictly with over-query and (b)
 * persistent recall matches the in-memory path at every over-query factor.
 */
@Tag("unit")
class IvfPersistentOverQueryRecallTest {

  private static final int N = 5000;
  private static final int DIM = 64;
  private static final int K = 10;
  private static final int IVF_K = 64;
  private static final int NPROBE_BASE = 4;
  private static final int RESCORE = 4;
  private static final int PQ_SUBSPACES = 16;
  private static final int PQ_CLUSTERS = 256;
  private static final long SEED = 42L;
  private static final float[] OVER_QUERY = {1.0f, 2.0f, 4.0f, 8.0f, 16.0f};

  private static final float[][] DATA = randomVecs(N, DIM, 3L);
  private static final float[][] QUERIES = randomVecs(100, DIM, 4L);

  private static float[][] randomVecs(int n, int dim, long seed) {
    Random rng = new Random(seed);
    float[][] m = new float[n][dim];
    for (float[] row : m) {
      for (int d = 0; d < dim; d++) {
        row[d] = rng.nextFloat() * 2f - 1f;
      }
    }
    return m;
  }

  private static int[] bruteForceTopK(float[] query, float[][] data, int k) {
    Integer[] order = new Integer[data.length];
    for (int i = 0; i < order.length; i++) {
      order[i] = i;
    }
    Arrays.sort(order, (a, b) -> Float.compare(sqDist(query, data[a]), sqDist(query, data[b])));
    int[] top = new int[k];
    for (int i = 0; i < k; i++) {
      top[i] = order[i];
    }
    return top;
  }

  private static float sqDist(float[] q, float[] v) {
    float s = 0f;
    for (int i = 0; i < q.length; i++) {
      float d = q[i] - v[i];
      s += d * d;
    }
    return s;
  }

  private static float recallAt(int[] found, int[] truth) {
    Set<Integer> t = new HashSet<>();
    for (int i : truth) {
      t.add(i);
    }
    int hit = 0;
    for (int i : found) {
      if (t.contains(i)) {
        hit++;
      }
    }
    return (float) hit / truth.length;
  }

  private static float meanRecall(IndexSpi adapter, float overQueryFactor) {
    float sum = 0f;
    for (float[] q : QUERIES) {
      int[] truth = bruteForceTopK(q, DATA, K);
      sum += recallAt(adapter.search(q, K, NPROBE_BASE, overQueryFactor).ordinals(), truth);
    }
    return sum / QUERIES.length;
  }

  private static IvfIndex buildIvfIndex(boolean pq) {
    IvfBuildParams base = new IvfBuildParams(IVF_K, 25, 0f, false, SEED, 0);
    IvfBuildParams params = pq ? base.withPq(PQ_SUBSPACES, PQ_CLUSTERS, -1f) : base;
    return IvfIndex.build(DATA, null, SimilarityFunction.EUCLIDEAN, params);
  }

  private static IvfPqAdapter inMemoryPq() {
    IvfPqAdapter a =
        new IvfPqAdapter(
            IVF_K, NPROBE_BASE, 25, 0f, false, SEED, PQ_SUBSPACES, PQ_CLUSTERS, -1f, RESCORE);
    a.build(DATA, SimilarityFunction.EUCLIDEAN);
    return a;
  }

  private static IvfFlatAdapter inMemoryFlat() {
    IvfFlatAdapter a = new IvfFlatAdapter(IVF_K, NPROBE_BASE, 25, 0f, false, SEED);
    a.build(DATA, SimilarityFunction.EUCLIDEAN);
    return a;
  }

  @Test
  void persistentPqOverQueryMeaningfullyImprovesRecall() {
    MappedIvfPqAdapter persistent =
        new MappedIvfPqAdapter(buildIvfIndex(true), NPROBE_BASE, 0f, RESCORE, DIM);
    float baseline = meanRecall(persistent, 1.0f);
    float overQueried = meanRecall(persistent, 8.0f);
    // The regression left these equal (probe count never widened); the fix lifts recall by ~0.6.
    assertThat(overQueried)
        .as("persistent IVF_PQ recall@10 must rise with overQueryFactor")
        .isGreaterThan(baseline + 0.3f);
  }

  @Test
  void persistentFlatOverQueryImprovesRecall() {
    MappedIvfFlatAdapter persistent =
        new MappedIvfFlatAdapter(buildIvfIndex(false), NPROBE_BASE, 0f, DIM);
    float baseline = meanRecall(persistent, 1.0f);
    float overQueried = meanRecall(persistent, 8.0f);
    // IVF_FLAT scoring is exact, so the only over-query recall lever is probe count. The regression
    // left this flat (no probe widening); the fix makes more clusters get scanned.
    assertThat(overQueried)
        .as("persistent IVF_FLAT recall@10 must rise with overQueryFactor")
        .isGreaterThan(baseline);
  }

  @Test
  void persistentPqMatchesInMemoryAcrossOverQuery() {
    assertParityAndMonotonic(
        inMemoryPq(), new MappedIvfPqAdapter(buildIvfIndex(true), NPROBE_BASE, 0f, RESCORE, DIM));
  }

  @Test
  void persistentFlatMatchesInMemoryAcrossOverQuery() {
    assertParityAndMonotonic(
        inMemoryFlat(), new MappedIvfFlatAdapter(buildIvfIndex(false), NPROBE_BASE, 0f, DIM));
  }

  private static void assertParityAndMonotonic(IndexSpi inMemory, IndexSpi persistent) {
    float prevPersistent = -1f;
    for (float oq : OVER_QUERY) {
      float inMem = meanRecall(inMemory, oq);
      float pers = meanRecall(persistent, oq);
      // Same underlying index + identical over-query logic => identical recall. Guards the two
      // adapters from silently diverging again.
      assertThat(pers)
          .as("persistent vs in-memory recall parity at oq=%s", oq)
          .isCloseTo(inMem, Offset.offset(1e-6f));
      assertThat(pers)
          .as("persistent recall non-decreasing at oq=%s", oq)
          .isGreaterThanOrEqualTo(prevPersistent);
      prevPersistent = pers;
    }
  }
}
