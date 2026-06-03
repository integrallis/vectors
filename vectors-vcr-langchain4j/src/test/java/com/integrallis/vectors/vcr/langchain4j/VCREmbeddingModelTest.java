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
import com.integrallis.vectors.vcr.CassetteKey;
import com.integrallis.vectors.vcr.CassetteRecord;
import com.integrallis.vectors.vcr.CassetteStore;
import com.integrallis.vectors.vcr.ExactCassetteStore;
import com.integrallis.vectors.vcr.SimilarityCassetteStore;
import com.integrallis.vectors.vcr.VCRCassetteMissingException;
import com.integrallis.vectors.vcr.VCRMode;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
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
  void playbackOrRecordUsesSemanticSingleEmbeddingOnExactMiss() {
    when(delegate.embed(anyString()))
        .thenReturn(Response.from(Embedding.from(new float[] {1f, 0f})));
    SemanticLookupStore semanticStore =
        new SemanticLookupStore(
            new CassetteRecord.Embedding("T:sem", "m", 1L, new float[] {0.99f, 0.01f}));

    VCREmbeddingModel player =
        new VCREmbeddingModel(delegate, "T:sem", VCRMode.PLAYBACK_OR_RECORD, "m", semanticStore);
    float[] played = player.embed("near").content().vector();

    assertThat(played).containsExactly(0.99f, 0.01f);
    assertThat(semanticStore.similarQueries).hasSize(1);
    assertThat(semanticStore.similarQueries.get(0)).containsExactly(1f, 0f);
    assertThat(semanticStore.stored).isEmpty();
    verify(delegate, times(1)).embed(anyString());
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
  @SuppressWarnings("unchecked")
  void playbackOrRecordUsesSemanticBatchEmbeddingOnExactMiss() {
    when(delegate.embedAll((List<TextSegment>) any(List.class)))
        .thenReturn(
            Response.from(
                List.of(
                    Embedding.from(new float[] {1f, 0f}), Embedding.from(new float[] {0f, 1f}))));
    SemanticLookupStore semanticStore =
        new SemanticLookupStore(
            new CassetteRecord.BatchEmbedding(
                "T:sem-b", "m", 1L, new float[][] {{0.98f, 0.02f}, {0.02f, 0.98f}}));

    VCREmbeddingModel player =
        new VCREmbeddingModel(delegate, "T:sem-b", VCRMode.PLAYBACK_OR_RECORD, "m", semanticStore);
    List<Embedding> played =
        player.embedAll(List.of(TextSegment.from("near-a"), TextSegment.from("near-b"))).content();

    assertThat(played).hasSize(2);
    assertThat(played.get(0).vector()).containsExactly(0.98f, 0.02f);
    assertThat(played.get(1).vector()).containsExactly(0.02f, 0.98f);
    assertThat(semanticStore.similarQueries).hasSize(1);
    assertThat(semanticStore.similarQueries.get(0)).containsExactly(1f, 0f);
    assertThat(semanticStore.stored).isEmpty();
    verify(delegate, times(1)).embedAll((List<TextSegment>) any(List.class));
  }

  @Test
  void offModeBypassesStore() {
    when(delegate.embed(anyString())).thenReturn(Response.from(Embedding.from(new float[] {9f})));
    VCREmbeddingModel off = new VCREmbeddingModel(delegate, "T:off", VCRMode.OFF, "m", store);
    assertThat(off.embed("x").content().vector()).containsExactly(9f);
  }

  private static final class SemanticLookupStore implements SimilarityCassetteStore {
    final Map<CassetteKey, CassetteRecord> stored = new LinkedHashMap<>();
    final List<float[]> similarQueries = new ArrayList<>();
    private final CassetteRecord similarHit;

    SemanticLookupStore(CassetteRecord similarHit) {
      this.similarHit = similarHit;
    }

    @Override
    public void store(CassetteKey key, CassetteRecord record) {
      stored.put(key, record);
    }

    @Override
    public Optional<CassetteRecord> retrieve(CassetteKey key) {
      return Optional.ofNullable(stored.get(key));
    }

    @Override
    public boolean exists(CassetteKey key) {
      return stored.containsKey(key);
    }

    @Override
    public void delete(CassetteKey key) {
      stored.remove(key);
    }

    @Override
    public List<CassetteKey> listByTestId(String testId) {
      return stored.keySet().stream().filter(key -> key.testId().equals(testId)).toList();
    }

    @Override
    public Optional<CassetteRecord> retrieveSimilar(float[] queryEmbedding) {
      similarQueries.add(queryEmbedding);
      return Optional.of(similarHit);
    }
  }
}
