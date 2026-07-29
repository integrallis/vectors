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
package com.integrallis.vectors.spring.boot;

import com.integrallis.vectors.db.VectorCollection;
import com.integrallis.vectors.db.VectorCollectionBuilder;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;

/**
 * Spring Boot auto-configuration for java-vectors.
 *
 * <p>Registers a {@link VectorCollection} bean from {@code java-vectors.*} properties. When Spring
 * AI is on the classpath an additional {@link
 * com.integrallis.vectors.spring.ai.JavaVectorsVectorStore} bean is registered, wiring the
 * collection to the application's {@code EmbeddingModel}.
 *
 * <p><strong>Zero-config path.</strong> With the starter and a Spring AI {@code EmbeddingModel}
 * bean on the classpath, no {@code java-vectors.*} configuration is required: {@code metric}
 * defaults to {@code COSINE} and the collection {@code dimension} is inferred from {@code
 * EmbeddingModel.dimensions()}. Adding the dependency is enough to get an indexed {@code
 * VectorStore}.
 *
 * <p>Override any default via {@code application.yml}:
 *
 * <pre>{@code
 * java-vectors:
 *   dimension: 1536          # optional when an EmbeddingModel is present (inferred otherwise)
 *   metric: COSINE           # optional; defaults to COSINE
 *   index-type: HNSW
 *   quantizer: SQ8
 *   storage-path: /var/lib/vectors/my-collection
 *   auto-commit-threshold: 1
 *   hnsw:
 *     m: 16
 *     ef-construction: 200
 * }</pre>
 *
 * <p>To replace the auto-configured beans, declare your own {@code @Bean} of the same type — the
 * {@code @ConditionalOnMissingBean} guards will skip autoconfiguration automatically.
 */
@AutoConfiguration
@EnableConfigurationProperties(JavaVectorsProperties.class)
public class JavaVectorsAutoConfiguration {

  /**
   * Creates a {@link VectorCollection} when Spring AI is <em>not</em> in play (no {@code
   * EmbeddingModel} bean is available to infer the dimension from). In that case the {@code
   * java-vectors.dimension} property is required.
   *
   * <p>When an {@code EmbeddingModel} bean is present, {@link
   * SpringAiConfiguration#vectorCollection} takes over instead and infers the dimension. The two
   * beans are mutually exclusive by condition — this one only matches when no {@code
   * EmbeddingModel} bean exists — so ordering between the auto-configurations does not matter.
   *
   * <p>The bean is {@link AutoCloseable}; Spring Boot closes it on application shutdown.
   *
   * @param props the bound {@code java-vectors.*} properties
   * @return a fully configured, open {@link VectorCollection}
   */
  @Bean
  @ConditionalOnMissingBean(
      value = VectorCollection.class,
      type = "org.springframework.ai.embedding.EmbeddingModel")
  public VectorCollection vectorCollection(JavaVectorsProperties props) {
    if (props.getDimension() <= 0) {
      throw new IllegalStateException(
          "java-vectors.dimension must be set to a positive value. No Spring AI EmbeddingModel bean "
              + "is available to infer it from — set java-vectors.dimension in your configuration, "
              + "or add a Spring AI EmbeddingModel bean so the dimension can be inferred.");
    }
    return buildCollection(props, props.getDimension());
  }

  /**
   * Builds a {@link VectorCollection} from {@code props} using an explicit {@code dimension}.
   *
   * <p>Shared by both the plain and Spring-AI auto-configured beans so the property-to-builder
   * mapping lives in one place; only the dimension source differs between the two.
   */
  static VectorCollection buildCollection(JavaVectorsProperties props, int dimension) {
    VectorCollectionBuilder builder =
        VectorCollection.builder()
            .dimension(dimension)
            .metric(props.getMetric())
            .indexType(props.getIndexType())
            .quantizer(props.getQuantizer())
            .autoCommitThreshold(props.getAutoCommitThreshold())
            .cacheSize(props.getCacheSize());

    if (props.getStoragePath() != null) {
      builder.storagePath(props.getStoragePath());
    }

    // HNSW parameters (only applied when indexType == HNSW, builder ignores them otherwise).
    JavaVectorsProperties.HnswProperties hnsw = props.getHnsw();
    if (hnsw != null) {
      builder.hnswM(hnsw.getM()).hnswEfConstruction(hnsw.getEfConstruction());
    }

    // Vamana parameters.
    JavaVectorsProperties.VamanaProperties vamana = props.getVamana();
    if (vamana != null) {
      builder
          .vamanaMaxDegree(vamana.getMaxDegree())
          .vamanaSearchListSize(vamana.getSearchListSize())
          .vamanaAlpha(vamana.getAlpha());
    }

    // IVF parameters.
    JavaVectorsProperties.IvfProperties ivf = props.getIvf();
    if (ivf != null) {
      builder.ivfK(ivf.getK()).ivfNprobe(ivf.getNprobe()).ivfMaxIter(ivf.getMaxIter());
    }

    // PQ parameters.
    JavaVectorsProperties.PqProperties pq = props.getPq();
    if (pq != null) {
      if (pq.getSubspaces() != null) {
        builder.pqSubspaces(pq.getSubspaces());
      }
      builder.pqClusters(pq.getClusters());
    }

    return builder.build();
  }

  /**
   * Spring AI wiring, activated only when {@code EmbeddingModel} is on the classpath so the starter
   * remains usable without Spring AI.
   *
   * <p>Registers two beans: a {@link VectorCollection} whose dimension is inferred from the {@code
   * EmbeddingModel} when {@code java-vectors.dimension} is unset, and a {@link
   * com.integrallis.vectors.spring.ai.JavaVectorsVectorStore} that wires that collection to the
   * model. Both are {@code @ConditionalOnMissingBean}, so a user-declared bean of either type wins.
   */
  @AutoConfiguration
  @ConditionalOnClass(name = "org.springframework.ai.embedding.EmbeddingModel")
  @EnableConfigurationProperties(JavaVectorsProperties.class)
  public static class SpringAiConfiguration {

    /**
     * Creates a {@link VectorCollection}, inferring the dimension from the {@code EmbeddingModel}
     * when {@code java-vectors.dimension} is unset (or non-positive). An explicit positive {@code
     * java-vectors.dimension} always takes precedence.
     *
     * @param props the bound {@code java-vectors.*} properties
     * @param embeddingModel the application's Spring AI embedding model
     * @return a fully configured, open {@link VectorCollection}
     */
    @Bean
    @ConditionalOnBean(type = "org.springframework.ai.embedding.EmbeddingModel")
    @ConditionalOnMissingBean(VectorCollection.class)
    public VectorCollection vectorCollection(
        JavaVectorsProperties props,
        org.springframework.ai.embedding.EmbeddingModel embeddingModel) {
      int dimension = props.getDimension() > 0 ? props.getDimension() : embeddingModel.dimensions();
      return buildCollection(props, dimension);
    }

    @Bean
    @ConditionalOnBean(type = "org.springframework.ai.embedding.EmbeddingModel")
    @ConditionalOnMissingBean(
        name = "javaVectorsVectorStore",
        value = com.integrallis.vectors.spring.ai.JavaVectorsVectorStore.class)
    public com.integrallis.vectors.spring.ai.JavaVectorsVectorStore javaVectorsVectorStore(
        org.springframework.ai.embedding.EmbeddingModel embeddingModel,
        VectorCollection collection,
        JavaVectorsProperties props) {
      return com.integrallis.vectors.spring.ai.JavaVectorsVectorStore.builder(
              embeddingModel, collection)
          .commitAfterAdd(props.isCommitAfterAdd())
          .build();
    }
  }
}
