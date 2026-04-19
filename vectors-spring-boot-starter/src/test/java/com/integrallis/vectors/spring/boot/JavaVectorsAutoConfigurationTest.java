package com.integrallis.vectors.spring.boot;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.QuantizerKind;
import com.integrallis.vectors.db.VectorCollection;
import com.integrallis.vectors.spring.ai.JavaVectorsVectorStore;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.embedding.EmbeddingRequest;
import org.springframework.ai.embedding.EmbeddingResponse;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * Gate tests for {@link JavaVectorsAutoConfiguration} (DP6).
 *
 * <p>Uses {@link ApplicationContextRunner} — no Spring Boot application context, no HTTP server,
 * fast unit-test startup.
 */
class JavaVectorsAutoConfigurationTest {

  /** Minimal stub EmbeddingModel that returns 4-dimensional embeddings. */
  static final class StubEmbeddingModel implements EmbeddingModel {

    private static final float[] STUB = {1f, 0f, 0f, 0f};

    @Override
    public float[] embed(org.springframework.ai.document.Document document) {
      return STUB;
    }

    @Override
    public java.util.List<float[]> embed(
        java.util.List<org.springframework.ai.document.Document> documents,
        org.springframework.ai.embedding.EmbeddingOptions options,
        org.springframework.ai.embedding.BatchingStrategy batchingStrategy) {
      return documents.stream().map(d -> STUB).toList();
    }

    @Override
    public EmbeddingResponse call(EmbeddingRequest request) {
      java.util.List<org.springframework.ai.embedding.Embedding> embeddings =
          request.getInstructions().stream()
              .map(
                  t ->
                      new org.springframework.ai.embedding.Embedding(
                          STUB, request.getInstructions().indexOf(t)))
              .toList();
      return new EmbeddingResponse(embeddings);
    }

    @Override
    public int dimensions() {
      return 4;
    }
  }

  @Configuration
  static class StubEmbeddingModelConfig {
    @Bean
    EmbeddingModel embeddingModel() {
      return new StubEmbeddingModel();
    }
  }

  private final ApplicationContextRunner runner =
      new ApplicationContextRunner()
          .withConfiguration(AutoConfigurations.of(JavaVectorsAutoConfiguration.class));

  // =========================================================================
  // Tests
  // =========================================================================

  @Nested
  @Tag("unit")
  class AutoConfigurationTests {

    @Test
    void vectorCollection_isCreated_withMinimalProperties() {
      runner
          .withPropertyValues("java-vectors.dimension=4", "java-vectors.metric=COSINE")
          .run(
              ctx -> {
                assertThat(ctx).hasSingleBean(VectorCollection.class);
                VectorCollection col = ctx.getBean(VectorCollection.class);
                assertThat(col.config().dimension()).isEqualTo(4);
                assertThat(col.config().metric()).isEqualTo(SimilarityFunction.COSINE);
                assertThat(col.config().indexType()).isEqualTo(IndexType.FLAT); // default
                assertThat(col.config().quantizerKind()).isEqualTo(QuantizerKind.NONE); // default
              });
    }

    @Test
    void vectorCollection_indexType_isWired_fromProperties() {
      runner
          .withPropertyValues(
              "java-vectors.dimension=4",
              "java-vectors.metric=EUCLIDEAN",
              "java-vectors.index-type=HNSW",
              "java-vectors.hnsw.m=8",
              "java-vectors.hnsw.ef-construction=100")
          .run(
              ctx -> {
                VectorCollection col = ctx.getBean(VectorCollection.class);
                assertThat(col.config().indexType()).isEqualTo(IndexType.HNSW);
                assertThat(col.config().hnswParams().m()).isEqualTo(8);
              });
    }

    @Test
    void vectorCollection_quantizer_isWired_fromProperties() {
      runner
          .withPropertyValues(
              "java-vectors.dimension=8",
              "java-vectors.metric=DOT_PRODUCT",
              "java-vectors.index-type=HNSW",
              "java-vectors.quantizer=SQ8")
          .run(
              ctx -> {
                VectorCollection col = ctx.getBean(VectorCollection.class);
                assertThat(col.config().quantizerKind()).isEqualTo(QuantizerKind.SQ8);
              });
    }

    @Test
    void vectorCollection_storagePath_wiresPersistentCollection(@TempDir Path tmp)
        throws IOException {
      Path colDir = tmp.resolve("test-collection");
      Files.createDirectories(colDir);
      runner
          .withPropertyValues(
              "java-vectors.dimension=4",
              "java-vectors.metric=COSINE",
              "java-vectors.storage-path=" + colDir.toAbsolutePath())
          .run(
              ctx -> {
                assertThat(ctx).hasSingleBean(VectorCollection.class);
                VectorCollection col = ctx.getBean(VectorCollection.class);
                assertThat(col.config().storageRoot()).isEqualTo(colDir);
              });
    }

    @Test
    void springAiVectorStore_isCreated_whenEmbeddingModelPresent() {
      runner
          .withUserConfiguration(StubEmbeddingModelConfig.class)
          .withPropertyValues("java-vectors.dimension=4", "java-vectors.metric=COSINE")
          .run(
              ctx -> {
                assertThat(ctx).hasSingleBean(JavaVectorsVectorStore.class);
              });
    }

    @Test
    void vectorCollection_cacheSize_bindsFromProperties_contextLoads() {
      // Verifies that java-vectors.cache-size=256 reaches VectorCollectionBuilder.cacheSize()
      // without throwing. The property cannot be inspected across package boundaries, so a clean
      // context startup is the gate.
      runner
          .withPropertyValues(
              "java-vectors.dimension=4",
              "java-vectors.metric=COSINE",
              "java-vectors.cache-size=256")
          .run(ctx -> assertThat(ctx).hasSingleBean(VectorCollection.class));
    }

    @Test
    void vectorCollection_cacheSize_defaultZero_contextLoads() {
      // cacheSize=0 (the default) disables the cache — must also produce a healthy context.
      runner
          .withPropertyValues("java-vectors.dimension=4", "java-vectors.metric=COSINE")
          .run(ctx -> assertThat(ctx).hasSingleBean(VectorCollection.class));
    }

    @Test
    void vectorCollection_cacheSize_negative_failsContextLoad() {
      // VectorCollectionBuilder.cacheSize(-1) throws IllegalArgumentException, so the context
      // should fail to start when cache-size is negative.
      runner
          .withPropertyValues(
              "java-vectors.dimension=4",
              "java-vectors.metric=COSINE",
              "java-vectors.cache-size=-1")
          .run(ctx -> assertThat(ctx).hasFailed());
    }

    @Test
    void userProvidedVectorCollection_suppressesAutoConfiguration() {
      runner
          .withPropertyValues("java-vectors.dimension=4", "java-vectors.metric=COSINE")
          .withUserConfiguration(UserVectorCollectionConfig.class)
          .run(
              ctx -> {
                assertThat(ctx).hasSingleBean(VectorCollection.class);
                // The user-defined bean should be the one present, not the auto-configured one
                assertThat(ctx.getBean(VectorCollection.class))
                    .isSameAs(ctx.getBean("userCollection"));
              });
    }
  }

  @Configuration
  static class UserVectorCollectionConfig {
    @Bean(name = "userCollection")
    VectorCollection userCollection() {
      return VectorCollection.builder()
          .dimension(4)
          .metric(SimilarityFunction.EUCLIDEAN)
          .indexType(IndexType.FLAT)
          .build();
    }
  }
}
