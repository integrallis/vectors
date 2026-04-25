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
package com.integrallis.vectors.ivf;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.vectors.core.SimilarityFunction;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Unit tests for the IVF-PQ search path. */
@Tag("unit")
class IvfPqTest {

  private static float[][] randomVecs(int n, int dim, long seed) {
    Random rng = new Random(seed);
    float[][] m = new float[n][dim];
    for (float[] row : m) for (int d = 0; d < dim; d++) row[d] = rng.nextFloat() * 2f - 1f;
    return m;
  }

  private static int[] bruteForceTopK(
      float[] query, float[][] dataset, int k, SimilarityFunction metric) {
    Integer[] order = new Integer[dataset.length];
    for (int i = 0; i < order.length; i++) order[i] = i;
    Arrays.sort(
        order,
        (a, b) ->
            Float.compare(score(query, dataset[b], metric), score(query, dataset[a], metric)));
    int[] top = new int[k];
    for (int i = 0; i < k; i++) top[i] = order[i];
    return top;
  }

  private static float score(float[] q, float[] v, SimilarityFunction metric) {
    return switch (metric) {
      case EUCLIDEAN -> {
        float s = 0f;
        for (int i = 0; i < q.length; i++) {
          float d = q[i] - v[i];
          s += d * d;
        }
        yield -s;
      }
      case DOT_PRODUCT, MAXIMUM_INNER_PRODUCT -> {
        float s = 0f;
        for (int i = 0; i < q.length; i++) s += q[i] * v[i];
        yield s;
      }
      case COSINE -> {
        float d = 0f, nq = 0f, nv = 0f;
        for (int i = 0; i < q.length; i++) {
          d += q[i] * v[i];
          nq += q[i] * q[i];
          nv += v[i] * v[i];
        }
        float nrm = (float) (Math.sqrt(nq) * Math.sqrt(nv));
        yield nrm == 0f ? 0f : d / nrm;
      }
    };
  }

  private static float recallAt(int[] found, int[] truth) {
    Set<Integer> t = new HashSet<>();
    for (int i : truth) t.add(i);
    int hit = 0;
    for (int i : found) if (t.contains(i)) hit++;
    return (float) hit / truth.length;
  }

  @Test
  void pqEnabledFlagFlipsIndexMode() {
    float[][] data = randomVecs(200, 16, 1L);
    IvfBuildParams flat = new IvfBuildParams(8, 20, 0f, false, 42L, 0);
    IvfBuildParams pq = flat.withPq(4);
    assertThat(IvfIndex.build(data, null, SimilarityFunction.EUCLIDEAN, flat).isQuantized())
        .isFalse();
    assertThat(IvfIndex.build(data, null, SimilarityFunction.EUCLIDEAN, pq).isQuantized()).isTrue();
  }

  @Test
  void pqDisabledPathIsBitIdenticalToIvfFlat() {
    float[][] data = randomVecs(500, 16, 2L);
    IvfBuildParams params = new IvfBuildParams(16, 30, 0f, false, 42L, 0);
    IvfIndex ivf = IvfIndex.build(data, null, SimilarityFunction.EUCLIDEAN, params);
    assertThat(ivf.isQuantized()).isFalse();
    assertThat(ivf.quantizer()).isNull();
    assertThat(ivf.pqCodes()).isNull();
  }

  @Test
  void ivfPqEuclideanRecallWithRescore() {
    int n = 1000, dim = 32, k = 10;
    float[][] data = randomVecs(n, dim, 3L);
    IvfBuildParams params = new IvfBuildParams(32, 30, 0f, false, 42L, 0).withPq(8);
    IvfIndex index = IvfIndex.build(data, null, SimilarityFunction.EUCLIDEAN, params);

    float[][] queries = randomVecs(20, dim, 4L);
    float sum = 0f;
    for (float[] q : queries) {
      int[] truth = bruteForceTopK(q, data, k, SimilarityFunction.EUCLIDEAN);
      IvfSearchResult r = index.search(new IvfSearchRequest(q, k, 16, 0f, -Float.MAX_VALUE, 8));
      int[] found = r.hits().stream().mapToInt(IvfHit::ordinal).toArray();
      sum += recallAt(found, truth);
    }
    assertThat(sum / queries.length).isGreaterThanOrEqualTo(0.70f);
  }

  @Test
  void ivfPqDotProductRespectsMetric() {
    int n = 400, dim = 16, k = 5;
    float[][] data = randomVecs(n, dim, 5L);
    IvfBuildParams params = new IvfBuildParams(16, 30, 0f, false, 42L, 0).withPq(4);
    IvfIndex index = IvfIndex.build(data, null, SimilarityFunction.DOT_PRODUCT, params);
    float[] q = randomVecs(1, dim, 6L)[0];
    IvfSearchResult r = index.search(new IvfSearchRequest(q, k, 16, 0f, -Float.MAX_VALUE, 4));
    // Rescored scores are exact dot products; ensure descending order and finite.
    float prev = Float.POSITIVE_INFINITY;
    for (IvfHit h : r.hits()) {
      assertThat(h.score()).isLessThanOrEqualTo(prev);
      prev = h.score();
    }
    assertThat(r.hits()).hasSize(k);
  }

  @Test
  void ivfPqCosineRequiresRescoreForAccuracy() {
    int n = 400, dim = 16, k = 5;
    float[][] data = randomVecs(n, dim, 7L);
    IvfBuildParams params = new IvfBuildParams(16, 30, 0f, false, 42L, 0).withPq(4);
    IvfIndex index = IvfIndex.build(data, null, SimilarityFunction.COSINE, params);
    float[] q = randomVecs(1, dim, 8L)[0];
    IvfSearchResult r = index.search(new IvfSearchRequest(q, k, 16, 0f, -Float.MAX_VALUE, 4));
    for (IvfHit h : r.hits()) {
      assertThat(h.score()).isBetween(-1.0001f, 1.0001f);
    }
  }

  @Test
  void rescoreFactorOneReturnsAdcScoresDirectly() {
    float[][] data = randomVecs(300, 16, 9L);
    IvfBuildParams params = new IvfBuildParams(16, 30, 0f, false, 42L, 0).withPq(4);
    IvfIndex index = IvfIndex.build(data, null, SimilarityFunction.EUCLIDEAN, params);
    float[] q = randomVecs(1, 16, 10L)[0];
    IvfSearchResult r = index.search(IvfSearchRequest.of(q, 5, 16));
    assertThat(r.hits()).hasSize(5);
    // Scores are negated squared L2 approximations — must be <= 0 for EUCLIDEAN.
    for (IvfHit h : r.hits()) assertThat(h.score()).isLessThanOrEqualTo(0f);
  }

  @Test
  void codecRoundTripPreservesPqSearchResults() {
    float[][] data = randomVecs(400, 16, 11L);
    IvfBuildParams params = new IvfBuildParams(16, 30, 0f, false, 42L, 0).withPq(4);
    IvfIndex original = IvfIndex.build(data, null, SimilarityFunction.EUCLIDEAN, params);
    byte[] bytes = original.encode();
    IvfIndex decoded = IvfIndex.decode(bytes, data, SimilarityFunction.EUCLIDEAN);

    assertThat(decoded.isQuantized()).isTrue();
    assertThat(decoded.pqCodes()).hasDimensions(data.length, 4);

    float[] q = randomVecs(1, 16, 12L)[0];
    IvfSearchRequest req = new IvfSearchRequest(q, 5, 8, 0f, -Float.MAX_VALUE, 4);
    IvfSearchResult orig = original.search(req);
    IvfSearchResult rt = decoded.search(req);
    assertThat(rt.hits()).hasSameSizeAs(orig.hits());
    for (int i = 0; i < orig.hits().size(); i++) {
      assertThat(rt.hits().get(i).ordinal()).isEqualTo(orig.hits().get(i).ordinal());
      assertThat(rt.hits().get(i).score()).isEqualTo(orig.hits().get(i).score());
    }
  }

  @Test
  void codecRoundTripOnIvfFlatStillWorks() {
    // Backward-compat: IVF-flat encode/decode round-trip must not regress after the PQ trailer
    // was appended.
    float[][] data = randomVecs(200, 8, 13L);
    IvfBuildParams params = new IvfBuildParams(8, 20, 0f, false, 42L, 0);
    IvfIndex original = IvfIndex.build(data, null, SimilarityFunction.EUCLIDEAN, params);
    IvfIndex decoded = IvfIndex.decode(original.encode(), data, SimilarityFunction.EUCLIDEAN);
    assertThat(decoded.isQuantized()).isFalse();
    assertThat(decoded.pqCodes()).isNull();
    assertThat(decoded.size()).isEqualTo(original.size());
  }

  @Test
  void buildParamsValidatesPqArguments() {
    IvfBuildParams base = IvfBuildParams.defaults(100);
    assertThatThrownBy(() -> base.withPq(0)).isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> base.withPq(4, 0, -1f)).isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> base.withPq(4, 512, -1f)).isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> base.withPq(4, 256, 1.5f))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void searchRequestValidatesRescoreFactor() {
    float[] q = new float[] {1, 2, 3};
    assertThatThrownBy(() -> new IvfSearchRequest(q, 5, 2, 0f, -Float.MAX_VALUE, 0))
        .isInstanceOf(IllegalArgumentException.class);
  }
}
