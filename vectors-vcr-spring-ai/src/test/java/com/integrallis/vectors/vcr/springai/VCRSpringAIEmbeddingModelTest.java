package com.integrallis.vectors.vcr.springai;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.integrallis.vectors.storage.backend.HeapStorageBackend;
import com.integrallis.vectors.vcr.CassetteStore;
import com.integrallis.vectors.vcr.ExactCassetteStore;
import com.integrallis.vectors.vcr.VCRCassetteMissingException;
import com.integrallis.vectors.vcr.VCRMode;
import java.util.List;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.ai.embedding.EmbeddingModel;

@Tag("unit")
@ExtendWith(MockitoExtension.class)
class VCRSpringAIEmbeddingModelTest {

  @Mock EmbeddingModel delegate;

  CassetteStore store;

  @BeforeEach
  void setUp() {
    store = new ExactCassetteStore(new HeapStorageBackend());
  }

  @Test
  void recordingStoresAndReplaysSingleEmbedding() {
    when(delegate.embed(anyString())).thenReturn(new float[] {1f, 2f, 3f});
    VCRSpringAIEmbeddingModel recorder =
        new VCRSpringAIEmbeddingModel(delegate, "T:s", VCRMode.RECORD, "m", store);
    float[] recorded = recorder.embed("hello");
    assertThat(recorded).containsExactly(1f, 2f, 3f);

    VCRSpringAIEmbeddingModel player =
        new VCRSpringAIEmbeddingModel(delegate, "T:s", VCRMode.PLAYBACK, "m", store);
    float[] played = player.embed("hello");
    assertThat(played).containsExactly(1f, 2f, 3f);
    verify(delegate, times(1)).embed(anyString());
  }

  @Test
  void playbackThrowsWhenCassetteMissing() {
    VCRSpringAIEmbeddingModel player =
        new VCRSpringAIEmbeddingModel(delegate, "T:miss", VCRMode.PLAYBACK, "m", store);
    assertThatThrownBy(() -> player.embed("anything"))
        .isInstanceOf(VCRCassetteMissingException.class);
    verify(delegate, never()).embed(anyString());
  }

  @Test
  void offModeBypassesStore() {
    when(delegate.embed(anyString())).thenReturn(new float[] {9f});
    VCRSpringAIEmbeddingModel off =
        new VCRSpringAIEmbeddingModel(delegate, "T:off", VCRMode.OFF, "m", store);
    assertThat(off.embed("x")).containsExactly(9f);
    off.embed("y");
    verify(delegate, times(2)).embed(anyString());
  }

  @Test
  @SuppressWarnings("unchecked")
  void batchRecordAndReplay() {
    when(delegate.embed((List<String>) any(List.class)))
        .thenReturn(List.of(new float[] {1f, 2f}, new float[] {3f, 4f}));
    VCRSpringAIEmbeddingModel recorder =
        new VCRSpringAIEmbeddingModel(delegate, "T:b", VCRMode.RECORD, "m", store);
    List<float[]> recorded = recorder.embed(List.of("a", "b"));
    assertThat(recorded).hasSize(2);

    VCRSpringAIEmbeddingModel player =
        new VCRSpringAIEmbeddingModel(delegate, "T:b", VCRMode.PLAYBACK, "m", store);
    List<float[]> played = player.embed(List.of("a", "b"));
    assertThat(played).hasSize(2);
    assertThat(played.get(0)).containsExactly(1f, 2f);
    assertThat(played.get(1)).containsExactly(3f, 4f);
    verify(delegate, times(1)).embed((List<String>) any(List.class));
  }
}
