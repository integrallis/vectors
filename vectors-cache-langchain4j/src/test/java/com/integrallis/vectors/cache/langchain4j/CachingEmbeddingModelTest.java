package com.integrallis.vectors.cache.langchain4j;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.cache.CaffeineVectorCache;
import com.integrallis.vectors.cache.VectorCache;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.output.Response;
import java.util.List;
import org.junit.jupiter.api.Test;

class CachingEmbeddingModelTest {

  private VectorCache<String, float[]> newCache() {
    return CaffeineVectorCache.<String, float[]>builder().maximumSize(1024).build();
  }

  @Test
  void embedStringDeduplicates() {
    FakeEmbeddingModel fake = new FakeEmbeddingModel();
    CachingEmbeddingModel model = new CachingEmbeddingModel(fake, newCache());

    Response<Embedding> a = model.embed("hello world");
    Response<Embedding> b = model.embed("hello world");
    Response<Embedding> c = model.embed("different");

    assertThat(a.content().vector()).containsExactly(b.content().vector());
    assertThat(c.content().vector()).isNotEqualTo(a.content().vector());
    assertThat(fake.singleCalls.get()).isEqualTo(2);
  }

  @Test
  void embedAllCoalescesMissesIntoOneBatchCall() {
    FakeEmbeddingModel fake = new FakeEmbeddingModel();
    CachingEmbeddingModel model = new CachingEmbeddingModel(fake, newCache());

    model.embed("warm");
    fake.singleCalls.set(0);

    List<TextSegment> segs =
        List.of(TextSegment.from("warm"), TextSegment.from("cold"), TextSegment.from("neutral"));
    Response<List<Embedding>> out = model.embedAll(segs);

    assertThat(out.content()).hasSize(3);
    assertThat(fake.batchCalls.get()).isEqualTo(1);
    assertThat(fake.singleCalls.get()).isZero();

    model.embedAll(segs);
    assertThat(fake.batchCalls.get()).isEqualTo(1);
  }

  @Test
  void embedAllPreservesRequestOrder() {
    FakeEmbeddingModel fake = new FakeEmbeddingModel();
    CachingEmbeddingModel model = new CachingEmbeddingModel(fake, newCache());

    model.embed("b");
    List<TextSegment> segs =
        List.of(TextSegment.from("a"), TextSegment.from("b"), TextSegment.from("c"));
    List<Embedding> out = model.embedAll(segs).content();

    assertThat(out.get(0).vector()).containsExactly(FakeEmbeddingModel.encode("a"));
    assertThat(out.get(1).vector()).containsExactly(FakeEmbeddingModel.encode("b"));
    assertThat(out.get(2).vector()).containsExactly(FakeEmbeddingModel.encode("c"));
  }

  @Test
  void segmentAndStringShareCacheKey() {
    FakeEmbeddingModel fake = new FakeEmbeddingModel();
    CachingEmbeddingModel model = new CachingEmbeddingModel(fake, newCache());

    model.embed("identical");
    model.embed(TextSegment.from("identical"));

    assertThat(fake.singleCalls.get()).isEqualTo(1);
  }

  @Test
  void customKeyFnCanonicalizesInput() {
    FakeEmbeddingModel fake = new FakeEmbeddingModel();
    CachingEmbeddingModel model = new CachingEmbeddingModel(fake, newCache(), String::toLowerCase);

    model.embed("Hello");
    model.embed("hello");

    assertThat(fake.singleCalls.get()).isEqualTo(1);
  }

  @Test
  void dimensionDelegate() {
    FakeEmbeddingModel fake = new FakeEmbeddingModel();
    CachingEmbeddingModel model = new CachingEmbeddingModel(fake, newCache());
    assertThat(model.dimension()).isEqualTo(FakeEmbeddingModel.DIM);
  }
}
