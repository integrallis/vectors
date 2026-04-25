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
