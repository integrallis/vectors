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
 * <p>Typical {@code application.yml}:
 *
 * <pre>{@code
 * java-vectors:
 *   dimension: 1536
 *   metric: COSINE
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
   * Creates and configures a {@link VectorCollection} from the bound properties.
   *
   * <p>The bean is {@link AutoCloseable}; Spring Boot closes it on application shutdown.
   *
   * @param props the bound {@code java-vectors.*} properties
   * @return a fully configured, open {@link VectorCollection}
   */
  @Bean
  @ConditionalOnMissingBean
  public VectorCollection vectorCollection(JavaVectorsProperties props) {
    VectorCollectionBuilder builder =
        VectorCollection.builder()
            .dimension(props.getDimension())
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
   * Registers a {@link com.integrallis.vectors.spring.ai.JavaVectorsVectorStore} bean that wires
   * the auto-configured {@link VectorCollection} to the application's Spring AI {@code
   * EmbeddingModel}.
   *
   * <p>This inner configuration class is only activated when Spring AI's {@code EmbeddingModel} is
   * on the classpath, so the starter remains usable without Spring AI.
   */
  @AutoConfiguration
  @ConditionalOnClass(name = "org.springframework.ai.embedding.EmbeddingModel")
  @EnableConfigurationProperties(JavaVectorsProperties.class)
  public static class SpringAiConfiguration {

    @Bean
    @ConditionalOnBean(type = "org.springframework.ai.embedding.EmbeddingModel")
    @ConditionalOnMissingBean(
        name = "javaVectorsVectorStore",
        value = com.integrallis.vectors.spring.ai.JavaVectorsVectorStore.class)
    public com.integrallis.vectors.spring.ai.JavaVectorsVectorStore javaVectorsVectorStore(
        org.springframework.ai.embedding.EmbeddingModel embeddingModel,
        VectorCollection collection) {
      return com.integrallis.vectors.spring.ai.JavaVectorsVectorStore.builder(
              embeddingModel, collection)
          .build();
    }
  }
}
