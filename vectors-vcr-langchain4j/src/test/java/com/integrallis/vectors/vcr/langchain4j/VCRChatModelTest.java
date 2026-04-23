package com.integrallis.vectors.vcr.langchain4j;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.integrallis.vectors.storage.backend.HeapStorageBackend;
import com.integrallis.vectors.vcr.CassetteStore;
import com.integrallis.vectors.vcr.ExactCassetteStore;
import com.integrallis.vectors.vcr.VCRCassetteMissingException;
import com.integrallis.vectors.vcr.VCRMode;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatLanguageModel;
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
@SuppressWarnings("removal")
class VCRChatModelTest {

  @Mock ChatLanguageModel delegate;

  CassetteStore store;

  @BeforeEach
  void setUp() {
    store = new ExactCassetteStore(new HeapStorageBackend());
  }

  @Test
  @SuppressWarnings("unchecked")
  void recordsAndReplaysChatResponse() {
    when(delegate.generate((List<ChatMessage>) any(List.class)))
        .thenReturn(Response.from(AiMessage.from("42")));

    VCRChatModel recorder = new VCRChatModel(delegate, "T:c", VCRMode.RECORD, "m", store);
    Response<AiMessage> recorded = recorder.generate(List.of(UserMessage.from("meaning?")));
    assertThat(recorded.content().text()).isEqualTo("42");

    VCRChatModel player = new VCRChatModel(delegate, "T:c", VCRMode.PLAYBACK, "m", store);
    Response<AiMessage> played = player.generate(List.of(UserMessage.from("meaning?")));
    assertThat(played.content().text()).isEqualTo("42");
    verify(delegate, times(1)).generate((List<ChatMessage>) any(List.class));
  }

  @Test
  @SuppressWarnings("unchecked")
  void playbackThrowsWhenMissing() {
    VCRChatModel player = new VCRChatModel(delegate, "T:miss", VCRMode.PLAYBACK, "m", store);
    assertThatThrownBy(() -> player.generate(List.of(UserMessage.from("x"))))
        .isInstanceOf(VCRCassetteMissingException.class);
    verify(delegate, never()).generate((List<ChatMessage>) any(List.class));
  }
}
