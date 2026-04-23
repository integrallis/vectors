package com.integrallis.vectors.vcr.springai;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
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
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.ai.chat.model.ChatModel;

@Tag("unit")
@ExtendWith(MockitoExtension.class)
class VCRSpringAIChatModelTest {

  @Mock ChatModel delegate;

  CassetteStore store;

  @BeforeEach
  void setUp() {
    store = new ExactCassetteStore(new HeapStorageBackend());
  }

  @Test
  void recordAndReplayStringCall() {
    when(delegate.call(anyString())).thenReturn("42");
    VCRSpringAIChatModel recorder =
        new VCRSpringAIChatModel(delegate, "T:c", VCRMode.RECORD, "m", store);
    assertThat(recorder.call("what?")).isEqualTo("42");

    VCRSpringAIChatModel player =
        new VCRSpringAIChatModel(delegate, "T:c", VCRMode.PLAYBACK, "m", store);
    assertThat(player.call("what?")).isEqualTo("42");
    verify(delegate, times(1)).call(anyString());
  }

  @Test
  void playbackThrowsWhenMissing() {
    VCRSpringAIChatModel player =
        new VCRSpringAIChatModel(delegate, "T:miss", VCRMode.PLAYBACK, "m", store);
    assertThatThrownBy(() -> player.call("anything"))
        .isInstanceOf(VCRCassetteMissingException.class);
    verify(delegate, never()).call(anyString());
  }

  @Test
  void offModeBypassesStore() {
    when(delegate.call(anyString())).thenReturn("live");
    VCRSpringAIChatModel off = new VCRSpringAIChatModel(delegate, "T:off", VCRMode.OFF, "m", store);
    assertThat(off.call("x")).isEqualTo("live");
  }
}
