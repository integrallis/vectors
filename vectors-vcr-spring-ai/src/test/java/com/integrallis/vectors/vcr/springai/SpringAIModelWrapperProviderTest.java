package com.integrallis.vectors.vcr.springai;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.storage.backend.HeapStorageBackend;
import com.integrallis.vectors.vcr.CassetteStore;
import com.integrallis.vectors.vcr.ExactCassetteStore;
import com.integrallis.vectors.vcr.VCRMode;
import com.integrallis.vectors.vcr.VCRModelWrapper;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.embedding.EmbeddingModel;

@Tag("unit")
@ExtendWith(MockitoExtension.class)
class SpringAIModelWrapperProviderTest {

  @Mock EmbeddingModel embeddingDelegate;
  @Mock ChatModel chatDelegate;

  private final CassetteStore store = new ExactCassetteStore(new HeapStorageBackend());

  @Test
  void wrapsEmbeddingModelThroughServiceLoader() {
    Object wrapped =
        VCRModelWrapper.wrapModel(embeddingDelegate, "T:e", VCRMode.RECORD, "m", store);
    assertThat(wrapped).isInstanceOf(VCRSpringAIEmbeddingModel.class);
  }

  @Test
  void wrapsChatModelThroughServiceLoader() {
    Object wrapped = VCRModelWrapper.wrapModel(chatDelegate, "T:c", VCRMode.RECORD, "m", store);
    assertThat(wrapped).isInstanceOf(VCRSpringAIChatModel.class);
  }
}
