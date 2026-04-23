package com.integrallis.vectors.vcr.langchain4j;

import com.integrallis.vectors.vcr.CassetteStore;
import com.integrallis.vectors.vcr.ModelWrapperProvider;
import com.integrallis.vectors.vcr.VCRMode;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.EmbeddingModel;

/**
 * {@link ModelWrapperProvider} for LangChain4j {@link EmbeddingModel} and {@link ChatLanguageModel}
 * instances. Registered through {@code META-INF/services}.
 */
public final class LangChain4jModelWrapperProvider implements ModelWrapperProvider {

  @Override
  public Object wrap(
      Object model, String testId, VCRMode mode, String modelName, CassetteStore cassetteStore) {
    if (model instanceof EmbeddingModel em && !(model instanceof VCREmbeddingModel)) {
      return new VCREmbeddingModel(em, testId, mode, modelName, cassetteStore);
    }
    if (model instanceof ChatLanguageModel cm && !(model instanceof VCRChatModel)) {
      return new VCRChatModel(cm, testId, mode, modelName, cassetteStore);
    }
    return null;
  }
}
