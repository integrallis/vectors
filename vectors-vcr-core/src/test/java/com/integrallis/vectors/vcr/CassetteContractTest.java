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
package com.integrallis.vectors.vcr;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatNullPointerException;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class CassetteContractTest {

  @Test
  void serializerLoadsFromServiceProviderAndUsesJsonContentType() {
    CassetteSerializer serializer = CassetteSerializer.load();

    assertThat(serializer).isInstanceOf(TestSerializer.class);
    assertThat(serializer.contentType()).isEqualTo("application/json");
  }

  @Test
  void missingExceptionCarriesLookupContext() {
    VCRCassetteMissingException exception =
        new VCRCassetteMissingException("vcr:chat:T:missing:0001", "T:missing");

    assertThat(exception.getCassetteKey()).isEqualTo("vcr:chat:T:missing:0001");
    assertThat(exception.getTestId()).isEqualTo("T:missing");
    assertThat(exception).hasMessageContaining("VCRMode.RECORD");
  }

  @Test
  void structuredChatPayloadUsesDefensiveCollectionsAndDefaultMetadata() {
    List<CassetteRecord.ToolCall> tools =
        new ArrayList<>(List.of(new CassetteRecord.ToolCall("1", "lookup", "{}")));
    Map<String, Object> attributes = new HashMap<>(Map.of("provider", "test"));

    CassetteRecord.AiMessagePayload message =
        new CassetteRecord.AiMessagePayload("answer", "reasoning", tools, attributes);
    CassetteRecord.ChatPayload payload = new CassetteRecord.ChatPayload(message, null);
    tools.clear();
    attributes.clear();

    assertThat(message.toolExecutionRequests()).hasSize(1);
    assertThat(message.attributes()).containsEntry("provider", "test");
    assertThat(payload.metadata()).isEqualTo(CassetteRecord.ChatMetadata.empty());
  }

  @Test
  void toolCallRequiresAName() {
    assertThatNullPointerException()
        .isThrownBy(() -> new CassetteRecord.ToolCall("1", null, "{}"))
        .withMessage("name");
  }
}
