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

import com.integrallis.vectors.core.SimilarityFunction;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The co-located block search scores neighbor codes read straight from a block via {@link
 * ExtendedRaBitQuantizedVectors#rawScorerFor} — with no ordinal lookup. This proves that path
 * returns exactly what the ordinal-based {@link ExtendedRaBitQuantizedVectors#scoreFunctionFor}
 * returns for the same code, across every similarity function and bit-width.
 */
@Tag("unit")
class ExtendedRaBitQRawScorerTest {

  @Test
  void rawScorerMatchesOrdinalScoreForEveryMetricAndBits() {
    int n = 300;
    int dim = 64;
    Random r = new Random(11);
    float[][] vecs = new float[n][dim];
    for (int i = 0; i < n; i++) {
      for (int d = 0; d < dim; d++) {
        vecs[i][d] = r.nextFloat() * 2f - 1f;
      }
    }
    ArrayVectorDataset ds = new ArrayVectorDataset(vecs);
    float[][] queries = new float[10][dim];
    for (int q = 0; q < queries.length; q++) {
      for (int d = 0; d < dim; d++) {
        queries[q][d] = r.nextFloat() * 2f - 1f;
      }
    }

    SimilarityFunction[] metrics = {
      SimilarityFunction.EUCLIDEAN,
      SimilarityFunction.DOT_PRODUCT,
      SimilarityFunction.COSINE,
      SimilarityFunction.MAXIMUM_INNER_PRODUCT
    };

    for (int bits : new int[] {2, 4, 7}) {
      ExtendedRaBitQuantizer quantizer = ExtendedRaBitQuantizer.train(ds, bits, 99L);
      ExtendedRaBitQuantizedVectors codes = quantizer.encodeAll(ds);
      for (SimilarityFunction metric : metrics) {
        for (float[] query : queries) {
          ScoreFunction byOrdinal = codes.scoreFunctionFor(query, metric);
          ExtendedRaBitQuantizedVectors.RawCodeScorer byRawCode = codes.rawScorerFor(query, metric);
          for (int ord = 0; ord < n; ord++) {
            float expected = byOrdinal.score(ord);
            float actual =
                byRawCode.score(
                    codes.getSignCodes(ord), codes.getMagCodes(ord), codes.getCorrections(ord));
            assertThat(actual)
                .as("bits=%d metric=%s ordinal=%d", bits, metric, ord)
                .isEqualTo(expected);
          }
        }
      }
    }
  }
}
