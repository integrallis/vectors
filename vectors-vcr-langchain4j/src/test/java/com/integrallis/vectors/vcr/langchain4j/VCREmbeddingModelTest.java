package com.integrallis.vectors.vcr.langchain4j;

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
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;
import java.util.List;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

@Tag("unit")
@ExtendWith(MockitoExtension.class)
class VCREmbeddingModelTest {

  @Mock EmbeddingModel delegate;

  CassetteStore store;

  @BeforeEach
  void setUp() {
    store = new ExactCassetteStore(new HeapStorageBackend());
  }

  @Test
  void recordsAndReplaysSingleEmbedding() {
    when(delegate.embed(anyString()))
        .thenReturn(Response.from(Embedding.from(new float[] {1f, 2f, 3f})));

    VCREmbeddingModel recorder = new VCREmbeddingModel(delegate, "T:s", VCRMode.RECORD, "m", store);
    float[] recorded = recorder.embed("hi").content().vector();
    assertThat(recorded).containsExactly(1f, 2f, 3f);

    VCREmbeddingModel player = new VCREmbeddingModel(delegate, "T:s", VCRMode.PLAYBACK, "m", store);
    float[] played = player.embed("hi").content().vector();
    assertThat(played).containsExactly(1f, 2f, 3f);
    verify(delegate, times(1)).embed(anyString());
  }

  @Test
  void playbackThrowsWhenMissing() {
    VCREmbeddingModel player =
        new VCREmbeddingModel(delegate, "T:miss", VCRMode.PLAYBACK, "m", store);
    assertThatThrownBy(() -> player.embed("x")).isInstanceOf(VCRCassetteMissingException.class);
    verify(delegate, never()).embed(anyString());
  }

  @Test
  @SuppressWarnings("unchecked")
  void recordsAndReplaysBatchEmbedding() {
    when(delegate.embedAll((List<TextSegment>) any(List.class)))
        .thenReturn(
            Response.from(
                List.of(
                    Embedding.from(new float[] {1f, 2f}), Embedding.from(new float[] {3f, 4f}))));

    VCREmbeddingModel recorder = new VCREmbeddingModel(delegate, "T:b", VCRMode.RECORD, "m", store);
    List<Embedding> recorded =
        recorder.embedAll(List.of(TextSegment.from("a"), TextSegment.from("b"))).content();
    assertThat(recorded).hasSize(2);

    VCREmbeddingModel player = new VCREmbeddingModel(delegate, "T:b", VCRMode.PLAYBACK, "m", store);
    List<Embedding> played =
        player.embedAll(List.of(TextSegment.from("a"), TextSegment.from("b"))).content();
    assertThat(played.get(0).vector()).containsExactly(1f, 2f);
    assertThat(played.get(1).vector()).containsExactly(3f, 4f);
    verify(delegate, times(1)).embedAll((List<TextSegment>) any(List.class));
  }

  @Test
  void offModeBypassesStore() {
    when(delegate.embed(anyString())).thenReturn(Response.from(Embedding.from(new float[] {9f})));
    VCREmbeddingModel off = new VCREmbeddingModel(delegate, "T:off", VCRMode.OFF, "m", store);
    assertThat(off.embed("x").content().vector()).containsExactly(9f);
  }
}
