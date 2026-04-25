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
package com.integrallis.vectors.vcr.semantic;

import com.integrallis.vectors.vcr.CassetteRecord;
import com.integrallis.vectors.vcr.CassetteSerializer;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.HashMap;
import java.util.Map;

/**
 * Test-only {@link CassetteSerializer} using raw Java {@code DataOutput}. Avoids depending on any
 * JSON library in this module's unit tests.
 */
final class TestSerializer implements CassetteSerializer {

  private static final byte TYPE_EMBEDDING = 1;
  private static final byte TYPE_BATCH_EMBEDDING = 2;
  private static final byte TYPE_CHAT = 3;

  @Override
  public byte[] serialize(CassetteRecord record) {
    try (ByteArrayOutputStream baos = new ByteArrayOutputStream();
        DataOutputStream out = new DataOutputStream(baos)) {
      if (record instanceof CassetteRecord.Embedding e) {
        out.writeByte(TYPE_EMBEDDING);
        writeCommon(out, e.testId(), e.model(), e.timestamp());
        writeFloatArray(out, e.embedding());
      } else if (record instanceof CassetteRecord.BatchEmbedding b) {
        out.writeByte(TYPE_BATCH_EMBEDDING);
        writeCommon(out, b.testId(), b.model(), b.timestamp());
        out.writeInt(b.embeddings().length);
        for (float[] v : b.embeddings()) {
          writeFloatArray(out, v);
        }
      } else if (record instanceof CassetteRecord.Chat c) {
        out.writeByte(TYPE_CHAT);
        writeCommon(out, c.testId(), c.model(), c.timestamp());
        out.writeUTF(c.prompt() == null ? "" : c.prompt());
        out.writeUTF(c.response());
        out.writeInt(c.metadata().size());
        for (Map.Entry<String, String> entry : c.metadata().entrySet()) {
          out.writeUTF(entry.getKey());
          out.writeUTF(entry.getValue());
        }
      } else {
        throw new IllegalArgumentException("unknown record type: " + record.getClass());
      }
      return baos.toByteArray();
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
  }

  @Override
  public CassetteRecord deserialize(byte[] bytes) {
    try (DataInputStream in = new DataInputStream(new ByteArrayInputStream(bytes))) {
      byte type = in.readByte();
      String testId = in.readUTF();
      String model = in.readUTF();
      long timestamp = in.readLong();
      switch (type) {
        case TYPE_EMBEDDING -> {
          return new CassetteRecord.Embedding(testId, model, timestamp, readFloatArray(in));
        }
        case TYPE_BATCH_EMBEDDING -> {
          int n = in.readInt();
          float[][] embeddings = new float[n][];
          for (int i = 0; i < n; i++) {
            embeddings[i] = readFloatArray(in);
          }
          return new CassetteRecord.BatchEmbedding(testId, model, timestamp, embeddings);
        }
        case TYPE_CHAT -> {
          String prompt = in.readUTF();
          String response = in.readUTF();
          int m = in.readInt();
          Map<String, String> metadata = new HashMap<>(m);
          for (int i = 0; i < m; i++) {
            metadata.put(in.readUTF(), in.readUTF());
          }
          return new CassetteRecord.Chat(testId, model, timestamp, prompt, response, metadata);
        }
        default -> throw new IllegalArgumentException("unknown type byte: " + type);
      }
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
  }

  private static void writeCommon(DataOutputStream out, String testId, String model, long timestamp)
      throws IOException {
    out.writeUTF(testId);
    out.writeUTF(model);
    out.writeLong(timestamp);
  }

  private static void writeFloatArray(DataOutputStream out, float[] values) throws IOException {
    out.writeInt(values.length);
    for (float v : values) {
      out.writeFloat(v);
    }
  }

  private static float[] readFloatArray(DataInputStream in) throws IOException {
    int n = in.readInt();
    float[] arr = new float[n];
    for (int i = 0; i < n; i++) {
      arr[i] = in.readFloat();
    }
    return arr;
  }
}
