package com.integrallis.vectors.server;

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
    ObjectMapper m = new ObjectMapper();
    m.registerModule(new JavaTimeModule());
    m.disable(SerializationFeature.WRITE_DATES_AS_TIMESTAMPS);
    m.disable(SerializationFeature.FAIL_ON_EMPTY_BEANS);
    m.enable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES);
    return m;
  }
}
