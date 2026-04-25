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
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * {@link CassetteSerializer} implementation using Jackson's streaming API.
 *
 * <p>Produces the same compact JSON shape as the Avaje serializer in {@code
 * vectors-vcr-serde-avaje} so cassettes are interoperable across both implementations.
 */
public final class JacksonCassetteSerializer implements CassetteSerializer {

  private static final JsonFactory FACTORY = new JsonFactory();

  private static final String TYPE_EMBEDDING = "embedding";
  private static final String TYPE_BATCH_EMBEDDING = "batch_embedding";
  private static final String TYPE_CHAT = "chat";

  @Override
  public byte[] serialize(CassetteRecord record) {
    try (ByteArrayOutputStream baos = new ByteArrayOutputStream();
        JsonGenerator g = FACTORY.createGenerator(baos)) {
      g.writeStartObject();
      if (record instanceof CassetteRecord.Embedding e) {
        g.writeStringField("type", TYPE_EMBEDDING);
        writeCommon(g, e.testId(), e.model(), e.timestamp());
        writeFloatArray(g, "embedding", e.embedding());
      } else if (record instanceof CassetteRecord.BatchEmbedding b) {
        g.writeStringField("type", TYPE_BATCH_EMBEDDING);
        writeCommon(g, b.testId(), b.model(), b.timestamp());
        g.writeArrayFieldStart("embeddings");
        for (float[] v : b.embeddings()) {
          writeFloatArrayBody(g, v);
        }
        g.writeEndArray();
      } else if (record instanceof CassetteRecord.Chat c) {
        g.writeStringField("type", TYPE_CHAT);
        writeCommon(g, c.testId(), c.model(), c.timestamp());
        g.writeStringField("prompt", c.prompt());
        g.writeStringField("response", c.response());
        g.writeObjectFieldStart("metadata");
        for (Map.Entry<String, String> entry : c.metadata().entrySet()) {
          g.writeStringField(entry.getKey(), entry.getValue());
        }
        g.writeEndObject();
      } else {
        throw new IllegalArgumentException("unsupported record type: " + record.getClass());
      }
      g.writeEndObject();
      g.flush();
      return baos.toByteArray();
    } catch (IOException e) {
      throw new UncheckedIOException("Jackson cassette serialization failed", e);
    }
  }

  @Override
  public CassetteRecord deserialize(byte[] bytes) {
    try (JsonParser p = FACTORY.createParser(new ByteArrayInputStream(bytes))) {
      if (p.nextToken() != JsonToken.START_OBJECT) {
        throw new IOException("expected START_OBJECT");
      }
      Map<String, Object> fields = new LinkedHashMap<>();
      while (p.nextToken() != JsonToken.END_OBJECT) {
        String name = p.currentName();
        JsonToken tok = p.nextToken();
        fields.put(name, readValue(p, tok));
      }
      String type = (String) fields.get("type");
      String testId = (String) fields.get("testId");
      String model = (String) fields.get("model");
      long timestamp = ((Number) fields.get("timestamp")).longValue();
      return switch (type) {
        case TYPE_EMBEDDING ->
            new CassetteRecord.Embedding(
                testId, model, timestamp, toFloatArray((List<?>) fields.get("embedding")));
        case TYPE_BATCH_EMBEDDING -> {
          List<?> arr = (List<?>) fields.get("embeddings");
          float[][] embeddings = new float[arr.size()][];
          for (int i = 0; i < arr.size(); i++) {
            embeddings[i] = toFloatArray((List<?>) arr.get(i));
          }
          yield new CassetteRecord.BatchEmbedding(testId, model, timestamp, embeddings);
        }
        case TYPE_CHAT -> {
          @SuppressWarnings("unchecked")
          Map<String, String> metadata = (Map<String, String>) fields.get("metadata");
          Object promptObj = fields.get("prompt");
          String prompt = promptObj instanceof String s ? s : "";
          String response = (String) fields.get("response");
          yield new CassetteRecord.Chat(
              testId, model, timestamp, prompt, response, metadata == null ? Map.of() : metadata);
        }
        default -> throw new IOException("unknown cassette type: " + type);
      };
    } catch (IOException e) {
      throw new UncheckedIOException("Jackson cassette deserialization failed", e);
    }
  }

  private static void writeCommon(JsonGenerator g, String testId, String model, long timestamp)
      throws IOException {
    g.writeStringField("testId", testId);
    g.writeStringField("model", model);
    g.writeNumberField("timestamp", timestamp);
  }

  private static void writeFloatArray(JsonGenerator g, String fieldName, float[] values)
      throws IOException {
    g.writeArrayFieldStart(fieldName);
    for (float v : values) {
      g.writeNumber(v);
    }
    g.writeEndArray();
  }

  private static void writeFloatArrayBody(JsonGenerator g, float[] values) throws IOException {
    g.writeStartArray();
    for (float v : values) {
      g.writeNumber(v);
    }
    g.writeEndArray();
  }

  private static Object readValue(JsonParser p, JsonToken tok) throws IOException {
    return switch (tok) {
      case VALUE_STRING -> p.getValueAsString();
      case VALUE_NUMBER_INT -> p.getLongValue();
      case VALUE_NUMBER_FLOAT -> p.getDoubleValue();
      case VALUE_TRUE -> Boolean.TRUE;
      case VALUE_FALSE -> Boolean.FALSE;
      case VALUE_NULL -> null;
      case START_ARRAY -> {
        List<Object> list = new ArrayList<>();
        while (p.nextToken() != JsonToken.END_ARRAY) {
          list.add(readValue(p, p.currentToken()));
        }
        yield list;
      }
      case START_OBJECT -> {
        Map<String, Object> obj = new HashMap<>();
        while (p.nextToken() != JsonToken.END_OBJECT) {
          String n = p.currentName();
          obj.put(n, readValue(p, p.nextToken()));
        }
        yield obj;
      }
      default -> throw new IOException("unexpected token: " + tok);
    };
  }

  private static float[] toFloatArray(List<?> list) {
    float[] arr = new float[list.size()];
    for (int i = 0; i < list.size(); i++) {
      arr[i] = ((Number) list.get(i)).floatValue();
    }
    return arr;
  }
}
