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
package com.integrallis.vectors.vcr.serde.jackson;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import com.integrallis.vectors.vcr.CassetteRecord;
import com.integrallis.vectors.vcr.CassetteSerializer;
import com.integrallis.vectors.vcr.CassetteTreeCodec;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * {@link CassetteSerializer} backed by Jackson's streaming API.
 *
 * <p>The shared {@link CassetteTreeCodec} owns the cassette shape, while this adapter handles only
 * the Jackson JSON byte boundary.
 */
public final class JacksonCassetteSerializer implements CassetteSerializer {

  private static final JsonFactory FACTORY = new JsonFactory();

  /**
   * Serializes one cassette through the canonical object tree.
   *
   * @param record cassette record to encode
   * @return encoded JSON bytes
   * @throws UncheckedIOException if Jackson cannot write the JSON document
   */
  @Override
  public byte[] serialize(CassetteRecord record) {
    try (ByteArrayOutputStream output = new ByteArrayOutputStream();
        JsonGenerator generator = FACTORY.createGenerator(output)) {
      writeAny(generator, CassetteTreeCodec.toTree(record));
      generator.flush();
      return output.toByteArray();
    } catch (IOException exception) {
      throw new UncheckedIOException("Jackson cassette serialization failed", exception);
    }
  }

  /**
   * Deserializes a cassette written by either supported JSON adapter.
   *
   * @param bytes encoded cassette JSON
   * @return decoded cassette record
   * @throws UncheckedIOException if the JSON document or cassette shape is malformed
   */
  @Override
  public CassetteRecord deserialize(byte[] bytes) {
    try (JsonParser parser = FACTORY.createParser(new ByteArrayInputStream(bytes))) {
      JsonToken token = parser.nextToken();
      if (token != JsonToken.START_OBJECT) {
        throw new IOException("expected START_OBJECT");
      }
      Object parsed = readValue(parser, token);
      if (!(parsed instanceof Map<?, ?> tree)) {
        throw new IOException("expected object at top level");
      }
      try {
        return CassetteTreeCodec.fromTree(tree);
      } catch (IllegalArgumentException exception) {
        throw new IOException("invalid cassette object", exception);
      }
    } catch (IOException exception) {
      throw new UncheckedIOException("Jackson cassette deserialization failed", exception);
    }
  }

  private static void writeAny(JsonGenerator generator, Object value) throws IOException {
    if (value == null) {
      generator.writeNull();
    } else if (value instanceof String string) {
      generator.writeString(string);
    } else if (value instanceof Integer number) {
      generator.writeNumber(number);
    } else if (value instanceof Long number) {
      generator.writeNumber(number);
    } else if (value instanceof Float number) {
      generator.writeNumber(number);
    } else if (value instanceof Double number) {
      generator.writeNumber(number);
    } else if (value instanceof Number number) {
      generator.writeNumber(number.toString());
    } else if (value instanceof Boolean bool) {
      generator.writeBoolean(bool);
    } else if (value instanceof Enum<?> enumValue) {
      generator.writeString(enumValue.name());
    } else if (value instanceof Map<?, ?> map) {
      generator.writeStartObject();
      for (Map.Entry<?, ?> entry : map.entrySet()) {
        generator.writeFieldName(String.valueOf(entry.getKey()));
        writeAny(generator, entry.getValue());
      }
      generator.writeEndObject();
    } else if (value instanceof Iterable<?> iterable) {
      generator.writeStartArray();
      for (Object item : iterable) {
        writeAny(generator, item);
      }
      generator.writeEndArray();
    } else if (value.getClass().isArray()) {
      generator.writeStartArray();
      for (int i = 0; i < Array.getLength(value); i++) {
        writeAny(generator, Array.get(value, i));
      }
      generator.writeEndArray();
    } else {
      generator.writeString(value.toString());
    }
  }

  private static Object readValue(JsonParser parser, JsonToken token) throws IOException {
    return switch (token) {
      case VALUE_STRING -> parser.getValueAsString();
      case VALUE_NUMBER_INT -> parser.getLongValue();
      case VALUE_NUMBER_FLOAT -> parser.getDoubleValue();
      case VALUE_TRUE -> Boolean.TRUE;
      case VALUE_FALSE -> Boolean.FALSE;
      case VALUE_NULL -> null;
      case START_ARRAY -> {
        List<Object> values = new ArrayList<>();
        while (parser.nextToken() != JsonToken.END_ARRAY) {
          values.add(readValue(parser, parser.currentToken()));
        }
        yield values;
      }
      case START_OBJECT -> {
        Map<String, Object> values = new LinkedHashMap<>();
        while (parser.nextToken() != JsonToken.END_OBJECT) {
          String name = parser.currentName();
          values.put(name, readValue(parser, parser.nextToken()));
        }
        yield values;
      }
      default -> throw new IOException("unexpected token: " + token);
    };
  }
}
