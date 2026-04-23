package com.integrallis.vectors.vcr.junit5;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.integrallis.vectors.storage.backend.HeapStorageBackend;
import com.integrallis.vectors.vcr.CassetteStore;
import com.integrallis.vectors.vcr.ExactCassetteStore;
import com.integrallis.vectors.vcr.VCRCassetteMissingException;
import com.integrallis.vectors.vcr.VCRMode;
import com.integrallis.vectors.vcr.VCRModelWrapper;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Exercises the wrapper-provider pipeline end-to-end using the test-only {@link
 * FakeModelWrapperProvider} registered through {@code META-INF/services}.
 */
@Tag("unit")
class VCRModelWrapperUnitTest {

  private final CassetteStore store = new ExactCassetteStore(new HeapStorageBackend());

  @Test
  void wrapsFakeModelViaServiceLoader() {
    FakeEmbeddingModel raw = prompt -> new float[] {prompt.length(), 1f};
    Object wrapped = VCRModelWrapper.wrapModel(raw, "T:wrap", VCRMode.RECORD, "fake", store);
    assertInstanceOf(VCRFakeEmbeddingModel.class, wrapped);
  }

  @Test
  void recordStoresCassetteForEachCall() {
    FakeEmbeddingModel raw = prompt -> new float[] {prompt.length() * 1f, 2f};
    FakeEmbeddingModel wrapped =
        (FakeEmbeddingModel) VCRModelWrapper.wrapModel(raw, "T:rec", VCRMode.RECORD, "fake", store);

    float[] out1 = wrapped.embed("hi");
    float[] out2 = wrapped.embed("world");
    assertArrayEquals(new float[] {2f, 2f}, out1);
    assertArrayEquals(new float[] {5f, 2f}, out2);
  }

  @Test
  void playbackReturnsRecordedVectors() {
    FakeEmbeddingModel raw = prompt -> new float[] {prompt.length() * 1f, 2f};
    FakeEmbeddingModel recorder =
        (FakeEmbeddingModel) VCRModelWrapper.wrapModel(raw, "T:pb", VCRMode.RECORD, "fake", store);
    float[] recorded1 = recorder.embed("alpha");
    float[] recorded2 = recorder.embed("beta");

    FakeEmbeddingModel bad = prompt -> new float[] {-999f};
    FakeEmbeddingModel player =
        (FakeEmbeddingModel)
            VCRModelWrapper.wrapModel(bad, "T:pb", VCRMode.PLAYBACK, "fake", store);

    assertArrayEquals(recorded1, player.embed("alpha"));
    assertArrayEquals(recorded2, player.embed("beta"));
  }

  @Test
  void playbackThrowsWhenCassetteMissing() {
    FakeEmbeddingModel raw = prompt -> new float[] {0f};
    FakeEmbeddingModel player =
        (FakeEmbeddingModel)
            VCRModelWrapper.wrapModel(raw, "T:miss", VCRMode.PLAYBACK, "fake", store);
    assertThrows(VCRCassetteMissingException.class, () -> player.embed("anything"));
  }

  @Test
  void modelNameFallsBackToFieldNameWhenBlank() throws Exception {
    class Holder {
      @com.integrallis.vectors.vcr.VCRModel FakeEmbeddingModel fake;
    }
    Holder holder = new Holder();
    holder.fake = prompt -> new float[] {1f};
    boolean wrapped =
        VCRModelWrapper.wrapField(
            holder, Holder.class.getDeclaredField("fake"), "T:f", VCRMode.RECORD, "", store);
    assertEquals(true, wrapped);
    assertInstanceOf(VCRFakeEmbeddingModel.class, holder.fake);
  }
}
