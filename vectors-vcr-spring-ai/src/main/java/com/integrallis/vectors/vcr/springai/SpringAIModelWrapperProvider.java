package com.integrallis.vectors.vcr.springai;

import com.integrallis.vectors.vcr.CassetteStore;
import com.integrallis.vectors.vcr.ModelWrapperProvider;
import com.integrallis.vectors.vcr.VCRMode;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.embedding.EmbeddingModel;

/**
 * {@link ModelWrapperProvider} for Spring AI {@link EmbeddingModel} and {@link ChatModel}
 * instances. Registered through {@code META-INF/services} so the core module discovers it via
 * {@link java.util.ServiceLoader}.
 */
public final class SpringAIModelWrapperProvider implements ModelWrapperProvider {

  @Override
  public Object wrap(
      Object model, String testId, VCRMode mode, String modelName, CassetteStore cassetteStore) {
    if (model instanceof EmbeddingModel em && !(model instanceof VCRSpringAIEmbeddingModel)) {
      return new VCRSpringAIEmbeddingModel(em, testId, mode, modelName, cassetteStore);
    }
    if (model instanceof ChatModel cm && !(model instanceof VCRSpringAIChatModel)) {
      return new VCRSpringAIChatModel(cm, testId, mode, modelName, cassetteStore);
    }
    return null;
  }
}
