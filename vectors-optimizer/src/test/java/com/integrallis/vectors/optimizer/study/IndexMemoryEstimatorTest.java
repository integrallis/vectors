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
package com.integrallis.vectors.optimizer.study;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.QuantizerKind;
import com.integrallis.vectors.db.QuantizerParams;
import com.integrallis.vectors.db.VectorCollectionConfig;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class IndexMemoryEstimatorTest {

  private static final int DIMENSION = 8;
  private static final int PHYSICAL_SIZE = 200;

  @Test
  void flatWithoutQuantizerReportsRawVectorPayload() {
    VectorCollectionConfig config = flatConfig(QuantizerKind.NONE, null);

    long memory = IndexMemoryEstimator.estimateIndexPayloadBytes(config, PHYSICAL_SIZE);

    assertThat(memory).isEqualTo((long) PHYSICAL_SIZE * DIMENSION * Float.BYTES);
  }

  @Test
  void quantizedFlatReportsRetainedRawVectorsAndCompressedPayload() {
    VectorCollectionConfig config =
        flatConfig(QuantizerKind.SQ8, new QuantizerParams.ScalarParams());

    long memory = IndexMemoryEstimator.estimateIndexPayloadBytes(config, PHYSICAL_SIZE);

    long raw = (long) PHYSICAL_SIZE * DIMENSION * Float.BYTES;
    long scalarCodesAndCorrections = (long) PHYSICAL_SIZE * (DIMENSION + Float.BYTES);
    long scalarState = Integer.BYTES + 5L * Float.BYTES;
    assertThat(memory).isEqualTo(raw + scalarCodesAndCorrections + scalarState);
  }

  @Test
  void hnswReportsVectorPayloadPlusGraphPayload() {
    VectorCollectionConfig flat = flatConfig(QuantizerKind.NONE, null);
    VectorCollectionConfig hnsw =
        new VectorCollectionConfig(
            DIMENSION,
            SimilarityFunction.COSINE,
            IndexType.HNSW,
            QuantizerKind.NONE,
            Integer.MAX_VALUE,
            null,
            new VectorCollectionConfig.HnswParams(16, 200),
            null,
            null,
            null,
            null,
            null);

    long flatMemory = IndexMemoryEstimator.estimateIndexPayloadBytes(flat, PHYSICAL_SIZE);
    long hnswMemory = IndexMemoryEstimator.estimateIndexPayloadBytes(hnsw, PHYSICAL_SIZE);

    assertThat(hnswMemory).isGreaterThan(flatMemory);
  }

  @Test
  void ivfPqReportsAdditionalPqPayloadOverIvfFlat() {
    VectorCollectionConfig ivfFlat =
        new VectorCollectionConfig(
            DIMENSION,
            SimilarityFunction.COSINE,
            IndexType.IVF_FLAT,
            QuantizerKind.NONE,
            Integer.MAX_VALUE,
            null,
            null,
            null,
            null,
            new VectorCollectionConfig.IvfParams(16, 4, 30, 0f, false, 42L),
            null,
            null);
    VectorCollectionConfig ivfPq =
        new VectorCollectionConfig(
            DIMENSION,
            SimilarityFunction.COSINE,
            IndexType.IVF_PQ,
            QuantizerKind.NONE,
            Integer.MAX_VALUE,
            null,
            null,
            null,
            null,
            null,
            null,
            new VectorCollectionConfig.IvfPqParams(16, 4, 30, 0f, false, 42L, 2, 16, -1f, 2));

    long ivfFlatMemory = IndexMemoryEstimator.estimateIndexPayloadBytes(ivfFlat, PHYSICAL_SIZE);
    long ivfPqMemory = IndexMemoryEstimator.estimateIndexPayloadBytes(ivfPq, PHYSICAL_SIZE);

    assertThat(ivfPqMemory).isGreaterThan(ivfFlatMemory);
  }

  private static VectorCollectionConfig flatConfig(
      QuantizerKind kind, QuantizerParams quantizerParams) {
    return new VectorCollectionConfig(
        DIMENSION,
        SimilarityFunction.COSINE,
        IndexType.FLAT,
        kind,
        Integer.MAX_VALUE,
        null,
        null,
        null,
        quantizerParams,
        null,
        null,
        null);
  }
}
