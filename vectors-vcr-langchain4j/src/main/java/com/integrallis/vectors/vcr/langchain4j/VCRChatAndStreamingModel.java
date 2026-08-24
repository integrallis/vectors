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
import com.integrallis.vectors.vcr.VCRMode;
import dev.langchain4j.model.ModelProvider;
import dev.langchain4j.model.chat.Capability;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.listener.ChatModelListener;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ChatRequestParameters;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import java.util.List;
import java.util.Set;

/** Preserves both interfaces when a LangChain4j model supports blocking and streaming chat. */
public final class VCRChatAndStreamingModel implements ChatModel, StreamingChatModel {

  private final ChatModel delegate;
  private final VCRChatModel blocking;
  private final VCRStreamingChatModel streaming;

  /** Creates a dual-interface VCR wrapper. */
  public VCRChatAndStreamingModel(
      ChatModel delegate,
      StreamingChatModel streamingDelegate,
      String testId,
      VCRMode mode,
      String modelName,
      CassetteStore store) {
    this.delegate = delegate;
    this.blocking = new VCRChatModel(delegate, testId, mode, modelName, store);
    this.streaming = new VCRStreamingChatModel(streamingDelegate, testId, mode, modelName, store);
  }

  @Override
  public ChatResponse doChat(ChatRequest request) {
    return blocking.doChat(request);
  }

  @Override
  public void doChat(ChatRequest request, StreamingChatResponseHandler handler) {
    streaming.doChat(request, handler);
  }

  @Override
  public ChatRequestParameters defaultRequestParameters() {
    return delegate.defaultRequestParameters();
  }

  @Override
  public List<ChatModelListener> listeners() {
    return delegate.listeners();
  }

  @Override
  public ModelProvider provider() {
    return delegate.provider();
  }

  @Override
  public Set<Capability> supportedCapabilities() {
    return delegate.supportedCapabilities();
  }
}
