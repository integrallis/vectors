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
import com.integrallis.vectors.vcr.CassetteKey;
import com.integrallis.vectors.vcr.CassetteRecord;
import com.integrallis.vectors.vcr.CassetteStore;
import com.integrallis.vectors.vcr.ExactCassetteStore;
import com.integrallis.vectors.vcr.VCRCassetteMissingException;
import com.integrallis.vectors.vcr.VCRMode;
import java.net.URI;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.ai.content.Media;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.embedding.EmbeddingRequest;
import org.springframework.util.MimeTypeUtils;

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

  @Test
  @SuppressWarnings("unchecked")
  void responseOverloadsExposeDelegateDimensionsAndOrderedResults() {
    when(delegate.dimensions()).thenReturn(2);
    when(delegate.embed((List<String>) any(List.class)))
        .thenReturn(List.of(new float[] {1f, 2f}, new float[] {3f, 4f}));
    VCRSpringAIEmbeddingModel model =
        new VCRSpringAIEmbeddingModel(delegate, "T:response", VCRMode.OFF, "m", store);

    var response = model.call(new EmbeddingRequest(List.of("a", "b"), null));
    var convenienceResponse = model.embedForResponse(List.of("c", "d"));

    assertThat(response.getResults()).hasSize(2);
    assertThat(response.getResults().get(0).getOutput()).containsExactly(1f, 2f);
    assertThat(response.getResults().get(1).getOutput()).containsExactly(3f, 4f);
    assertThat(convenienceResponse.getResults()).hasSize(2);
    assertThat(model.dimensions()).isEqualTo(2);
    assertThat(model.getDelegate()).isSameAs(delegate);
    verify(delegate, times(2)).embed((List<String>) any(List.class));
  }

  @Test
  void documentEmbeddingNormalizesNullText() {
    Media media =
        new Media(MimeTypeUtils.IMAGE_PNG, URI.create("https://example.invalid/test-document.png"));
    Document document = new Document(media, Map.of());
    when(delegate.embed("")).thenReturn(new float[] {7f});
    VCRSpringAIEmbeddingModel model =
        new VCRSpringAIEmbeddingModel(delegate, "T:document", VCRMode.OFF, "m", store);

    assertThat(document.getText()).isNull();
    assertThat(model.embed(document)).containsExactly(7f);
    verify(delegate).embed("");
  }

  @Test
  void playbackRejectsWrongSingleCassetteType() {
    store.store(
        new CassetteKey("embedding", "T:wrong-single", 1),
        new CassetteRecord.BatchEmbedding("T:wrong-single", "m", 1L, new float[][] {{1f}}));
    VCRSpringAIEmbeddingModel model =
        new VCRSpringAIEmbeddingModel(delegate, "T:wrong-single", VCRMode.PLAYBACK, "m", store);

    assertThatThrownBy(() -> model.embed("x"))
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("BatchEmbedding");
  }

  @Test
  void playbackRejectsWrongOrMissingBatchCassette() {
    store.store(
        new CassetteKey("batch_embedding", "T:wrong-batch", 1),
        new CassetteRecord.Embedding("T:wrong-batch", "m", 1L, new float[] {1f}));
    VCRSpringAIEmbeddingModel wrong =
        new VCRSpringAIEmbeddingModel(delegate, "T:wrong-batch", VCRMode.PLAYBACK, "m", store);
    VCRSpringAIEmbeddingModel missing =
        new VCRSpringAIEmbeddingModel(delegate, "T:missing-batch", VCRMode.PLAYBACK, "m", store);

    assertThatThrownBy(() -> wrong.embed(List.of("x")))
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("Embedding");
    assertThatThrownBy(() -> missing.embed(List.of("x")))
        .isInstanceOf(VCRCassetteMissingException.class);
  }

  @Test
  void dimensionDetectionFailureUsesUnknownSentinel() {
    when(delegate.dimensions()).thenThrow(new IllegalStateException("not initialized"));

    VCRSpringAIEmbeddingModel model =
        new VCRSpringAIEmbeddingModel(delegate, "T:dimensions", VCRMode.OFF, "m", store);

    assertThat(model.dimensions()).isEqualTo(-1);
  }
}
