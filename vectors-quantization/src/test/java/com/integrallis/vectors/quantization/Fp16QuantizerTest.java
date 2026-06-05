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
package com.integrallis.vectors.quantization;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatIllegalArgumentException;
import static org.assertj.core.api.Assertions.within;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.core.VectorUtil;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class Fp16QuantizerTest {

  private static final int DIM = 96;
  private static final int N = 300;

  private static float[][] gaussianVectors(long seed) {
    Random rng = new Random(seed);
    float[][] v = new float[N][DIM];
    for (int i = 0; i < N; i++) {
      for (int d = 0; d < DIM; d++) {
        v[i][d] = (float) rng.nextGaussian();
      }
    }
    return v;
  }

  private static float[] unit(float[] v) {
    double norm = 0;
    for (float x : v) {
      norm += (double) x * x;
    }
    float inv = (float) (1.0 / Math.sqrt(norm));
    float[] out = new float[v.length];
    for (int i = 0; i < v.length; i++) {
      out[i] = v[i] * inv;
    }
    return out;
  }

  @Test
  void compressionRatioIsTwo() {
    assertThat(new Fp16Quantizer(DIM).compressionRatio()).isEqualTo(2.0f);
  }

  @Test
  void encodedSizeIsTwoBytesPerDimension() {
    assertThat(new Fp16Quantizer(DIM).encode(new float[DIM])).hasSize(DIM * 2);
  }

  @Test
  void rejectsNonPositiveDimension() {
    assertThatIllegalArgumentException().isThrownBy(() -> new Fp16Quantizer(0));
  }

  @Test
  void rejectsWrongLengthVector() {
    assertThatIllegalArgumentException()
        .isThrownBy(() -> new Fp16Quantizer(DIM).encode(new float[DIM - 1]));
  }

  @Test
  void roundTripIsNearLossless() {
    Fp16Quantizer q = new Fp16Quantizer(DIM);
    for (float[] v : gaussianVectors(1L)) {
      float[] back = q.decode(q.encode(v));
      for (int d = 0; d < DIM; d++) {
        // binary16 has an 11-bit significand → relative error ≤ ~2^-11 ≈ 5e-4.
        float tol = Math.max(1e-3f, Math.abs(v[d]) * 1e-3f);
        assertThat(back[d]).isCloseTo(v[d], within(tol));
      }
    }
  }

  @Test
  void scoresTrackFullPrecision() {
    float[][] raw = gaussianVectors(2L);
    float[][] vecs = new float[N][];
    for (int i = 0; i < N; i++) {
      vecs[i] = unit(raw[i]);
    }
    ArrayVectorDataset dataset = new ArrayVectorDataset(vecs);
    Fp16Quantizer q = Fp16Quantizer.train(dataset);
    assertThat(q.dimension()).isEqualTo(DIM);
    Fp16QuantizedVectors cv = q.encodeAll(dataset);
    assertThat(cv.size()).isEqualTo(N);

    float[] query = unit(gaussianVectors(7L)[0]);
    for (SimilarityFunction sim : SimilarityFunction.values()) {
      ScoreFunction scorer = cv.scoreFunctionFor(query, sim);
      for (int i = 0; i < N; i++) {
        // The scorer differs from the exact full-precision score only by the fp16 reconstruction
        // error on the stored vector — tiny for unit-norm vectors.
        float exact = exactScore(query, vecs[i], sim);
        assertThat(scorer.score(i))
            .as("similarity %s, vector %d", sim, i)
            .isCloseTo(exact, within(5e-3f));
      }
    }
  }

  private static float exactScore(float[] query, float[] vec, SimilarityFunction sim) {
    return switch (sim) {
      case DOT_PRODUCT, COSINE -> Math.max((1f + VectorUtil.dotProduct(query, vec)) / 2f, 0f);
      case EUCLIDEAN -> 1f / (1f + VectorUtil.squareDistance(query, vec));
      case MAXIMUM_INNER_PRODUCT ->
          SimilarityFunction.scaleMaxInnerProductScore(VectorUtil.dotProduct(query, vec));
    };
  }
}
