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
