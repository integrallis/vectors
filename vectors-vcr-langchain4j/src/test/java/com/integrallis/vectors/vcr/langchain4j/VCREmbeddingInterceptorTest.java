package com.integrallis.vectors.vcr.langchain4j;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.vectors.storage.backend.HeapStorageBackend;
import com.integrallis.vectors.vcr.CassetteStore;
import com.integrallis.vectors.vcr.ExactCassetteStore;
import com.integrallis.vectors.vcr.VCRCassetteMissingException;
import com.integrallis.vectors.vcr.VCRMode;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class VCREmbeddingInterceptorTest {

  static final class TestInterceptor extends VCREmbeddingInterceptor {
    final AtomicInteger singleCalls = new AtomicInteger();
    final AtomicInteger batchCalls = new AtomicInteger();

    TestInterceptor(CassetteStore store) {
      super(store);
    }

    @Override
    protected float[] callRealEmbedding(String text) {
      singleCalls.incrementAndGet();
      return new float[] {(float) text.length(), 42f};
    }

    @Override
    protected List<float[]> callRealBatchEmbedding(List<String> texts) {
      batchCalls.incrementAndGet();
      return texts.stream().map(t -> new float[] {(float) t.length(), 7f}).toList();
    }
  }

  CassetteStore store;
  TestInterceptor interceptor;

  @BeforeEach
  void setUp() {
    store = new ExactCassetteStore(new HeapStorageBackend());
    interceptor = new TestInterceptor(store);
    interceptor.setTestId("MyTest.method");
  }

  @Test
  void offModeBypassesStore() {
    interceptor.setMode(VCRMode.OFF);
    float[] v = interceptor.embed("hi");
    assertThat(v).containsExactly(2f, 42f);
    assertThat(interceptor.getRecordedCount()).isZero();
    assertThat(interceptor.getCacheHits()).isZero();
    assertThat(interceptor.getCacheMisses()).isZero();
  }

  @Test
  void recordModeCallsProviderAndRecords() {
    interceptor.setMode(VCRMode.RECORD);
    interceptor.embed("hello");
    interceptor.embed("world");
    assertThat(interceptor.singleCalls.get()).isEqualTo(2);
    assertThat(interceptor.getRecordedCount()).isEqualTo(2);
    assertThat(interceptor.getRecordedKeys())
        .containsExactly("vcr:embedding:MyTest.method:0001", "vcr:embedding:MyTest.method:0002");
    assertThat(interceptor.getCacheMisses()).isEqualTo(2);
    assertThat(interceptor.getCacheHits()).isZero();
  }

  @Test
  void batchRecordCountsAsSingleCassette() {
    interceptor.setMode(VCRMode.RECORD);
    List<float[]> out = interceptor.embedBatch(List.of("a", "bb", "ccc"));
    assertThat(out).hasSize(3);
    assertThat(interceptor.batchCalls.get()).isEqualTo(1);
    assertThat(interceptor.getRecordedCount()).isEqualTo(1);
    assertThat(interceptor.getRecordedKeys())
        .containsExactly("vcr:batch_embedding:MyTest.method:0001");
  }

  @Test
  void playbackReturnsPreloadedCassette() {
    interceptor.setMode(VCRMode.PLAYBACK);
    float[] expected = {0.1f, 0.2f, 0.3f};
    interceptor.preloadCassette("vcr:embedding:MyTest.method:0001", expected);

    float[] v = interceptor.embed("any");
    assertThat(v).containsExactly(expected);
    assertThat(interceptor.singleCalls.get()).isZero();
    assertThat(interceptor.getCacheHits()).isEqualTo(1);
  }

  @Test
  void playbackBatchReturnsPreloadedCassette() {
    interceptor.setMode(VCRMode.PLAYBACK);
    float[][] expected = {{0.1f, 0.2f}, {0.3f, 0.4f}};
    interceptor.preloadBatchCassette("vcr:batch_embedding:MyTest.method:0001", expected);

    List<float[]> v = interceptor.embedBatch(List.of("a", "b"));
    assertThat(v).hasSize(2);
    assertThat(v.get(0)).containsExactly(expected[0]);
    assertThat(v.get(1)).containsExactly(expected[1]);
    assertThat(interceptor.batchCalls.get()).isZero();
    assertThat(interceptor.getCacheHits()).isEqualTo(1);
  }

  @Test
  void playbackThrowsWhenCassetteMissing() {
    interceptor.setMode(VCRMode.PLAYBACK);
    assertThatThrownBy(() -> interceptor.embed("missing"))
        .isInstanceOf(VCRCassetteMissingException.class);
  }

  @Test
  void playbackOrRecordFillsCacheOnMiss() {
    interceptor.setMode(VCRMode.PLAYBACK_OR_RECORD);
    float[] first = interceptor.embed("once");
    assertThat(interceptor.singleCalls.get()).isEqualTo(1);
    assertThat(interceptor.getRecordedCount()).isEqualTo(1);
    assertThat(interceptor.getCacheHits()).isZero();
    assertThat(interceptor.getCacheMisses()).isEqualTo(1);

    interceptor.resetCallCounter();
    interceptor.resetStatistics();
    float[] second = interceptor.embed("once");
    assertThat(second).containsExactly(first);
    assertThat(interceptor.singleCalls.get()).isEqualTo(1);
    assertThat(interceptor.getCacheHits()).isEqualTo(1);
  }

  @Test
  void setTestIdResetsCounters() {
    interceptor.setMode(VCRMode.RECORD);
    interceptor.embed("a");
    interceptor.setTestId("Other.method");
    interceptor.embed("b");
    assertThat(interceptor.getRecordedKeys())
        .containsExactly("vcr:embedding:MyTest.method:0001", "vcr:embedding:Other.method:0001");
  }

  @Test
  void invalidPreloadKeyThrows() {
    assertThatThrownBy(() -> interceptor.preloadCassette("not-a-key", new float[] {1f}))
        .isInstanceOf(IllegalArgumentException.class);
  }
}
