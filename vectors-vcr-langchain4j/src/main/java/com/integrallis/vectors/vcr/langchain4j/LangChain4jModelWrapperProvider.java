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

import com.integrallis.vectors.vcr.CassetteStore;
import com.integrallis.vectors.vcr.ModelWrapperProvider;
import com.integrallis.vectors.vcr.VCRMode;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;

/**
 * {@link ModelWrapperProvider} for LangChain4j {@link EmbeddingModel} and {@link ChatModel}
 * instances. Registered through {@code META-INF/services}.
 */
public final class LangChain4jModelWrapperProvider implements ModelWrapperProvider {

  @Override
  public Object wrap(
      Object model, String testId, VCRMode mode, String modelName, CassetteStore cassetteStore) {
    if (model instanceof EmbeddingModel em && !(model instanceof VCREmbeddingModel)) {
      return new VCREmbeddingModel(em, testId, mode, modelName, cassetteStore);
    }
    if (model instanceof ChatModel cm && !(model instanceof VCRChatModel)) {
      return new VCRChatModel(cm, testId, mode, modelName, cassetteStore);
    }
    return null;
  }
}
