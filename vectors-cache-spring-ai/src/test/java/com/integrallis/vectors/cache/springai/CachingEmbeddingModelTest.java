package com.integrallis.vectors.cache.springai;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.cache.CaffeineVectorCache;
import com.integrallis.vectors.cache.VectorCache;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingRequest;
import org.springframework.ai.embedding.EmbeddingResponse;

class CachingEmbeddingModelTest {

  private VectorCache<String, float[]> newCache() {
    return CaffeineVectorCache.<String, float[]>builder().maximumSize(1024).build();
  }

  @Test
  void embedStringDeduplicates() {
    FakeEmbeddingModel fake = new FakeEmbeddingModel();
    CachingEmbeddingModel model = new CachingEmbeddingModel(fake, newCache());

    float[] a = model.embed("hello world");
    float[] b = model.embed("hello world");
    float[] c = model.embed("different");

    assertThat(a).containsExactly(b);
    assertThat(c).isNotEqualTo(a);
    assertThat(fake.singleCalls.get()).isEqualTo(2);
  }

  @Test
  void embedBatchCoalescesMissesIntoOneDelegateCall() {
    FakeEmbeddingModel fake = new FakeEmbeddingModel();
    CachingEmbeddingModel model = new CachingEmbeddingModel(fake, newCache());

    model.embed("warm");
    fake.singleCalls.set(0);

    List<float[]> out = model.embed(List.of("warm", "cold", "neutral"));

    assertThat(out).hasSize(3);
    assertThat(fake.batchCalls.get()).isEqualTo(1);
    assertThat(fake.singleCalls.get()).isZero();

    List<float[]> again = model.embed(List.of("warm", "cold", "neutral"));
    assertThat(again).hasSize(3);
    assertThat(fake.batchCalls.get()).isEqualTo(1);
  }

  @Test
  void embedBatchPreservesRequestOrder() {
    FakeEmbeddingModel fake = new FakeEmbeddingModel();
    CachingEmbeddingModel model = new CachingEmbeddingModel(fake, newCache());

    model.embed("b");
    List<float[]> out = model.embed(List.of("a", "b", "c"));

    assertThat(out.get(0)).containsExactly(FakeEmbeddingModel.encode("a"));
    assertThat(out.get(1)).containsExactly(FakeEmbeddingModel.encode("b"));
    assertThat(out.get(2)).containsExactly(FakeEmbeddingModel.encode("c"));
  }

  @Test
  void documentEmbeddingUsesTextForKey() {
    FakeEmbeddingModel fake = new FakeEmbeddingModel();
    CachingEmbeddingModel model = new CachingEmbeddingModel(fake, newCache());

    model.embed("the quick brown fox");
    float[] v = model.embed(new Document("the quick brown fox"));

    assertThat(v).containsExactly(FakeEmbeddingModel.encode("the quick brown fox"));
    assertThat(fake.singleCalls.get()).isEqualTo(1);
  }

  @Test
  void callReturnsEmbeddingResponseWithIndexes() {
    FakeEmbeddingModel fake = new FakeEmbeddingModel();
    CachingEmbeddingModel model = new CachingEmbeddingModel(fake, newCache());

    EmbeddingResponse resp = model.call(new EmbeddingRequest(List.of("a", "b"), null));

    assertThat(resp.getResults()).hasSize(2);
    assertThat(resp.getResults().get(0).getIndex()).isZero();
    assertThat(resp.getResults().get(1).getIndex()).isEqualTo(1);
  }

  @Test
  void customKeyFnCanonicalizesInput() {
    FakeEmbeddingModel fake = new FakeEmbeddingModel();
    CachingEmbeddingModel model = new CachingEmbeddingModel(fake, newCache(), String::toLowerCase);

    model.embed("Hello");
    model.embed("hello");
    model.embed("HELLO");

    assertThat(fake.singleCalls.get()).isEqualTo(1);
  }

  @Test
  void dimensionsDelegate() {
    FakeEmbeddingModel fake = new FakeEmbeddingModel();
    CachingEmbeddingModel model = new CachingEmbeddingModel(fake, newCache());
    assertThat(model.dimensions()).isEqualTo(FakeEmbeddingModel.DIM);
  }
}
