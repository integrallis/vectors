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

import com.integrallis.vectors.storage.backend.HeapStorageBackend;
import com.integrallis.vectors.vcr.CassetteStore;
import com.integrallis.vectors.vcr.ExactCassetteStore;
import com.integrallis.vectors.vcr.VCRMode;
import com.integrallis.vectors.vcr.VCRModelWrapper;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

@Tag("unit")
@ExtendWith(MockitoExtension.class)
class LangChain4jModelWrapperProviderTest {

  @Mock EmbeddingModel embeddingDelegate;
  @Mock ChatLanguageModel chatDelegate;

  private final CassetteStore store = new ExactCassetteStore(new HeapStorageBackend());

  @Test
  void wrapsEmbeddingModelThroughServiceLoader() {
    Object wrapped =
        VCRModelWrapper.wrapModel(embeddingDelegate, "T:e", VCRMode.RECORD, "m", store);
    assertThat(wrapped).isInstanceOf(VCREmbeddingModel.class);
  }

  @Test
  void wrapsChatModelThroughServiceLoader() {
    Object wrapped = VCRModelWrapper.wrapModel(chatDelegate, "T:c", VCRMode.RECORD, "m", store);
    assertThat(wrapped).isInstanceOf(VCRChatModel.class);
  }
}
