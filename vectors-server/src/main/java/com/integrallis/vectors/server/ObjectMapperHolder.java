/*
 * Copyright 2025-2026 Integrallis Software, LLC
 *
 * Licensed under the Functional Source License, Version 1.1, Apache 2.0 Future License
 * (the "License"); you may not use this file except in compliance with the License.
 *
 *     https://fsl.software/FSL-1.1-ALv2.txt
 *
 * Unless required by applicable law or agreed to in writing, software distributed under
 * the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
 * ANY KIND, either express or implied. See the License for the specific language
 * governing permissions and limitations under the License.
 *
 * Change Date: April 25, 2028
 * Change License: Apache License, Version 2.0
 */
package com.integrallis.vectors.server;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.StreamReadFeature;
import com.fasterxml.jackson.core.StreamWriteFeature;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;

/**
 * Holder for the shared Jackson {@link ObjectMapper} used by all server routes.
 *
 * <p>Configured with {@link JavaTimeModule} for ISO-8601 {@link java.time.Instant} serialization,
 * strict deserialization (unknown properties fail), and record-friendly defaults.
 */
public final class ObjectMapperHolder {

  private static final ObjectMapper MAPPER = buildDefault();

  private ObjectMapperHolder() {}

  /**
   * @return the shared server-wide mapper
   */
  public static ObjectMapper shared() {
    return MAPPER;
  }

  /**
   * Builds a fresh, standalone mapper with the server-wide defaults. Used by tests that need an
   * isolated mapper.
   *
   * @return a new mapper
   */
  public static ObjectMapper buildDefault() {
    JsonFactory factory =
        JsonFactory.builder()
            .enable(StreamReadFeature.USE_FAST_DOUBLE_PARSER)
            .enable(StreamWriteFeature.USE_FAST_DOUBLE_WRITER)
            .build();
    ObjectMapper m = new ObjectMapper(factory);
    m.registerModule(new JavaTimeModule());
    m.disable(SerializationFeature.WRITE_DATES_AS_TIMESTAMPS);
    m.disable(SerializationFeature.FAIL_ON_EMPTY_BEANS);
    m.enable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES);
    return m;
  }
}
