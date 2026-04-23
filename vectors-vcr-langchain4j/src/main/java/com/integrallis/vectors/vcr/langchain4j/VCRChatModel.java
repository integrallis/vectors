package com.integrallis.vectors.vcr.langchain4j;

import com.integrallis.vectors.vcr.CassetteKey;
import com.integrallis.vectors.vcr.CassetteRecord;
import com.integrallis.vectors.vcr.CassetteStore;
import com.integrallis.vectors.vcr.VCRCassetteMissingException;
import com.integrallis.vectors.vcr.VCRMode;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.output.Response;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * LangChain4j {@link ChatLanguageModel} wrapper that records/replays chat responses through a
 * {@link CassetteStore}.
 */
@SuppressWarnings("removal")
public final class VCRChatModel implements ChatLanguageModel {

  private static final String TYPE_CHAT = "chat";

  private final ChatLanguageModel delegate;
  private final CassetteStore store;
  private final String testId;
  private final String modelName;
  private final VCRMode mode;
  private final AtomicInteger callCounter = new AtomicInteger();

  /**
   * @param delegate underlying LangChain4j chat model
   * @param testId test identifier
   * @param mode VCR mode
   * @param modelName model name for cassette metadata
   * @param store cassette store
   */
  public VCRChatModel(
      ChatLanguageModel delegate,
      String testId,
      VCRMode mode,
      String modelName,
      CassetteStore store) {
    this.delegate = delegate;
    this.testId = testId;
    this.mode = mode;
    this.modelName = modelName;
    this.store = store;
  }

  @Override
  public Response<AiMessage> generate(List<ChatMessage> messages) {
    if (mode == VCRMode.OFF) {
      return delegate.generate(messages);
    }
    CassetteKey key = new CassetteKey(TYPE_CHAT, testId, callCounter.incrementAndGet());
    if (mode.isPlaybackMode()) {
      Optional<CassetteRecord> hit = store.retrieve(key);
      if (hit.isPresent()) {
        String text = ((CassetteRecord.Chat) hit.get()).response();
        return Response.from(AiMessage.from(text));
      }
      if (mode == VCRMode.PLAYBACK) {
        throw new VCRCassetteMissingException(key.serializedKey(), testId);
      }
    }
    Response<AiMessage> response = delegate.generate(messages);
    String prompt = String.valueOf(messages);
    String text = response.content() == null ? "" : response.content().text();
    store.store(
        key,
        new CassetteRecord.Chat(
            testId, modelName, System.currentTimeMillis(), prompt, text, Map.of()));
    return response;
  }

  /**
   * @return the underlying delegate
   */
  public ChatLanguageModel getDelegate() {
    return delegate;
  }
}
