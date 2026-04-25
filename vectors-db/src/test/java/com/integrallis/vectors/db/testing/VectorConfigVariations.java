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
package com.integrallis.vectors.db.testing;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.QuantizerKind;
import com.integrallis.vectors.db.VectorCollection;
import com.integrallis.vectors.db.VectorCollectionBuilder;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.Consumer;
import java.util.stream.Stream;
import org.junit.jupiter.api.DynamicTest;

/**
 * Combinatorial test configuration generator.
 *
 * <p>Inspired by Apache Ignite's {@code ConfigVariationsTestSuiteBuilder}. Write a test body once
 * and run it against every combination of index type, quantizer, metric, and dimension:
 *
 * <pre>{@code
 * @TestFactory
 * Stream<DynamicTest> searchConsistency() {
 *     return VectorConfigVariations.builder()
 *         .indexTypes(IndexType.HNSW, IndexType.VAMANA)
 *         .quantizers(QuantizerKind.NONE, QuantizerKind.SQ8)
 *         .metrics(SimilarityFunction.COSINE, SimilarityFunction.EUCLIDEAN)
 *         .dimensions(64, 128)
 *         .buildTests(config -> {
 *             var collection = config.newBuilder().build();
 *             // insert, search, verify recall
 *         });
 * }
 * }</pre>
 */
public final class VectorConfigVariations {

  private VectorConfigVariations() {}

  /** Returns a new builder. */
  public static Builder builder() {
    return new Builder();
  }

  /**
   * A single configuration variation. Passed to the test body lambda in {@link
   * Builder#buildTests(Consumer)}.
   */
  public record Config(
      IndexType indexType, QuantizerKind quantizer, SimilarityFunction metric, int dimension) {

    /** Creates a pre-configured {@link VectorCollectionBuilder} matching this config. */
    public VectorCollectionBuilder newBuilder() {
      return VectorCollection.builder()
          .dimension(dimension)
          .metric(metric)
          .indexType(indexType)
          .quantizer(quantizer);
    }

    @Override
    public String toString() {
      return indexType + "/" + quantizer + "/" + metric + "/dim=" + dimension;
    }
  }

  /** Fluent builder for generating config combinations. */
  public static final class Builder {
    private IndexType[] indexTypes = {IndexType.HNSW, IndexType.VAMANA};
    private QuantizerKind[] quantizers = {QuantizerKind.NONE, QuantizerKind.SQ8};
    private SimilarityFunction[] metrics = {
      SimilarityFunction.COSINE, SimilarityFunction.EUCLIDEAN
    };
    private int[] dimensions = {64, 128};

    private Builder() {}

    /** Sets the index types to vary. */
    public Builder indexTypes(IndexType... indexTypes) {
      this.indexTypes = Objects.requireNonNull(indexTypes);
      return this;
    }

    /** Sets the quantizers to vary. */
    public Builder quantizers(QuantizerKind... quantizers) {
      this.quantizers = Objects.requireNonNull(quantizers);
      return this;
    }

    /** Sets the similarity functions to vary. */
    public Builder metrics(SimilarityFunction... metrics) {
      this.metrics = Objects.requireNonNull(metrics);
      return this;
    }

    /** Sets the dimensions to vary. */
    public Builder dimensions(int... dimensions) {
      this.dimensions = Objects.requireNonNull(dimensions);
      return this;
    }

    /** Returns all config combinations as a list. */
    public List<Config> build() {
      List<Config> configs = new ArrayList<>();
      for (IndexType idx : indexTypes) {
        for (QuantizerKind q : quantizers) {
          for (SimilarityFunction m : metrics) {
            for (int d : dimensions) {
              configs.add(new Config(idx, q, m, d));
            }
          }
        }
      }
      return configs;
    }

    /**
     * Generates a stream of {@link DynamicTest} instances, one per config combination.
     *
     * @param testBody the test logic to execute for each configuration
     * @return stream of dynamic tests suitable for {@code @TestFactory}
     */
    public Stream<DynamicTest> buildTests(Consumer<Config> testBody) {
      return build().stream()
          .map(config -> DynamicTest.dynamicTest(config.toString(), () -> testBody.accept(config)));
    }
  }
}
