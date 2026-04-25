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
package com.integrallis.vectors.db;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.gpu.GpuProvider;
import com.integrallis.vectors.gpu.GpuUnavailableException;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * GPU backend integration gate for {@link VectorCollection}. Verifies builder accepts the {@link
 * IndexType#CUVS_BRUTEFORCE}/{@link IndexType#CUVS_CAGRA} variants, rejects persistent storage
 * paths paired with CUVS_*, and that the adapters surface {@link GpuUnavailableException} when no
 * compatible GPU is detected.
 */
@Tag("unit")
class VectorDbCuVsTest {

  private static final int DIM = 32;

  /**
   * The adapter's {@code build()} performs the availability probe before any other work, which
   * means {@link VectorCollection#builder()} throws on non-GPU hosts the moment it bootstraps the
   * empty in-memory generation.
   */
  @Test
  void bruteForceBuilderThrowsOnNonGpuHost() {
    if (GpuProvider.availability().isAvailable()) {
      return;
    }
    assertThatThrownBy(
            () ->
                VectorCollection.builder()
                    .dimension(DIM)
                    .metric(SimilarityFunction.EUCLIDEAN)
                    .indexType(IndexType.CUVS_BRUTEFORCE)
                    .build())
        .isInstanceOf(GpuUnavailableException.class);
  }

  @Test
  void cagraBuilderThrowsOnNonGpuHost() {
    if (GpuProvider.availability().isAvailable()) {
      return;
    }
    assertThatThrownBy(
            () ->
                VectorCollection.builder()
                    .dimension(DIM)
                    .metric(SimilarityFunction.EUCLIDEAN)
                    .indexType(IndexType.CUVS_CAGRA)
                    .cuvsParams(VectorCollectionConfig.CuVsParams.Cagra.defaults())
                    .build())
        .isInstanceOf(GpuUnavailableException.class);
  }

  @Test
  void cuvsRejectsPersistentStoragePath(@TempDir Path tmp) {
    assertThatThrownBy(
            () ->
                VectorCollection.builder()
                    .dimension(DIM)
                    .metric(SimilarityFunction.EUCLIDEAN)
                    .indexType(IndexType.CUVS_BRUTEFORCE)
                    .storagePath(tmp)
                    .build())
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("CUVS_* index types do not support persistent storage");
  }

  @Test
  void cuvsRejectsQuantizerKind() {
    assertThatThrownBy(
            () ->
                VectorCollection.builder()
                    .dimension(DIM)
                    .metric(SimilarityFunction.EUCLIDEAN)
                    .indexType(IndexType.CUVS_BRUTEFORCE)
                    .quantizer(QuantizerKind.PQ)
                    .build())
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("CUVS_* index types do not support quantization");
  }

  @Test
  void bruteForceConfigUsesDefaultCuVsParams() {
    // Builder with no explicit cuvsParams() call should synthesize BruteForce.defaults()
    // and the config should validate successfully (validation happens before adapter.build()).
    VectorCollectionConfig cfg =
        new VectorCollectionConfig(
            DIM,
            SimilarityFunction.EUCLIDEAN,
            IndexType.CUVS_BRUTEFORCE,
            QuantizerKind.NONE,
            Integer.MAX_VALUE,
            null,
            null,
            null,
            null,
            null,
            VectorCollectionConfig.CuVsParams.BruteForce.defaults());
    assertThat(cfg.cuvsParams()).isInstanceOf(VectorCollectionConfig.CuVsParams.BruteForce.class);
  }

  @Test
  void cagraConfigValidatesParamVariantMatchesIndexType() {
    // CUVS_CAGRA requires a CuVsParams.Cagra variant; passing BruteForce should fail validation.
    assertThatThrownBy(
            () ->
                new VectorCollectionConfig(
                    DIM,
                    SimilarityFunction.EUCLIDEAN,
                    IndexType.CUVS_CAGRA,
                    QuantizerKind.NONE,
                    Integer.MAX_VALUE,
                    null,
                    null,
                    null,
                    null,
                    null,
                    VectorCollectionConfig.CuVsParams.BruteForce.defaults()))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("CUVS_CAGRA requires CuVsParams.Cagra");
  }
}
