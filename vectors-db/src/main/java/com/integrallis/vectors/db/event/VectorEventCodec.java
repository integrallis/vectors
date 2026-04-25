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
package com.integrallis.vectors.db.event;

import com.integrallis.vectors.core.MetadataValue;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Binary codec for {@link VectorEvent}. Converts events to/from compact byte arrays suitable for
 * writing to a {@link com.integrallis.vectors.storage.wal.SegmentedWriteAheadLog}.
 *
 * <p>Wire format (all integers big-endian):
 *
 * <pre>
 *   [1 byte: type tag]  1=Add 2=Delete 3=Upsert 4=Commit
 *   Per type:
 *     Add/Upsert: [UTF id][int dim][dim * float vector][UTF-or-null text][int metaCount][entries...]
 *     Delete:     [UTF id]
 *     Commit:     [long generationNumber]
 *   MetadataValue entry: [UTF key][byte valueType][payload]
 *     valueType 1=Str(UTF), 2=Num(double), 3=Bool(byte), 4=Tags(int n, n*UTF)
 * </pre>
 */
public final class VectorEventCodec {

  private static final byte TAG_ADD = 1;
  private static final byte TAG_DELETE = 2;
  private static final byte TAG_UPSERT = 3;
  private static final byte TAG_COMMIT = 4;

  private static final byte META_STR = 1;
  private static final byte META_NUM = 2;
  private static final byte META_BOOL = 3;
  private static final byte META_TAGS = 4;

  private VectorEventCodec() {}

  /** Encodes {@code event} to a byte array. Never returns {@code null}. */
  public static byte[] encode(VectorEvent event) {
    try (ByteArrayOutputStream bos = new ByteArrayOutputStream();
        DataOutputStream out = new DataOutputStream(bos)) {
      switch (event) {
        case VectorEvent.Add e -> {
          out.writeByte(TAG_ADD);
          writeDoc(out, e.id(), e.vector(), e.text(), e.metadata());
        }
        case VectorEvent.Delete e -> {
          out.writeByte(TAG_DELETE);
          out.writeUTF(e.id());
        }
        case VectorEvent.Upsert e -> {
          out.writeByte(TAG_UPSERT);
          writeDoc(out, e.id(), e.vector(), e.text(), e.metadata());
        }
        case VectorEvent.Commit e -> {
          out.writeByte(TAG_COMMIT);
          out.writeLong(e.generationNumber());
        }
      }
      out.flush();
      return bos.toByteArray();
    } catch (IOException ex) {
      throw new UncheckedIOException(ex);
    }
  }

  /** Decodes a byte array produced by {@link #encode}. */
  public static VectorEvent decode(byte[] bytes) {
    try (DataInputStream in = new DataInputStream(new ByteArrayInputStream(bytes))) {
      byte tag = in.readByte();
      return switch (tag) {
        case TAG_ADD -> readAdd(in);
        case TAG_DELETE -> new VectorEvent.Delete(in.readUTF());
        case TAG_UPSERT -> readUpsert(in);
        case TAG_COMMIT -> new VectorEvent.Commit(in.readLong());
        default -> throw new IllegalArgumentException("unknown event tag: " + tag);
      };
    } catch (IOException ex) {
      throw new UncheckedIOException(ex);
    }
  }

  // --- helpers ---

  private static void writeDoc(
      DataOutputStream out,
      String id,
      float[] vector,
      String text,
      Map<String, MetadataValue> metadata)
      throws IOException {
    out.writeUTF(id);
    out.writeInt(vector.length);
    for (float v : vector) out.writeFloat(v);
    out.writeBoolean(text != null);
    if (text != null) out.writeUTF(text);
    out.writeInt(metadata.size());
    for (Map.Entry<String, MetadataValue> e : metadata.entrySet()) {
      out.writeUTF(e.getKey());
      writeMetadataValue(out, e.getValue());
    }
  }

  private static VectorEvent.Add readAdd(DataInputStream in) throws IOException {
    String id = in.readUTF();
    float[] v = readVector(in);
    String text = in.readBoolean() ? in.readUTF() : null;
    Map<String, MetadataValue> meta = readMeta(in);
    return new VectorEvent.Add(id, v, text, meta);
  }

  private static VectorEvent.Upsert readUpsert(DataInputStream in) throws IOException {
    String id = in.readUTF();
    float[] v = readVector(in);
    String text = in.readBoolean() ? in.readUTF() : null;
    Map<String, MetadataValue> meta = readMeta(in);
    return new VectorEvent.Upsert(id, v, text, meta);
  }

  private static float[] readVector(DataInputStream in) throws IOException {
    int dim = in.readInt();
    float[] v = new float[dim];
    for (int i = 0; i < dim; i++) v[i] = in.readFloat();
    return v;
  }

  private static Map<String, MetadataValue> readMeta(DataInputStream in) throws IOException {
    int n = in.readInt();
    Map<String, MetadataValue> meta = new HashMap<>(n);
    for (int i = 0; i < n; i++) {
      String key = in.readUTF();
      meta.put(key, readMetadataValue(in));
    }
    return meta;
  }

  private static void writeMetadataValue(DataOutputStream out, MetadataValue v) throws IOException {
    switch (v) {
      case MetadataValue.Str s -> {
        out.writeByte(META_STR);
        out.writeUTF(s.value());
      }
      case MetadataValue.Num n -> {
        out.writeByte(META_NUM);
        out.writeDouble(n.value());
      }
      case MetadataValue.Bool b -> {
        out.writeByte(META_BOOL);
        out.writeBoolean(b.value());
      }
      case MetadataValue.Tags t -> {
        out.writeByte(META_TAGS);
        out.writeInt(t.values().size());
        for (String s : t.values()) out.writeUTF(s);
      }
    }
  }

  private static MetadataValue readMetadataValue(DataInputStream in) throws IOException {
    byte type = in.readByte();
    return switch (type) {
      case META_STR -> new MetadataValue.Str(in.readUTF());
      case META_NUM -> new MetadataValue.Num(in.readDouble());
      case META_BOOL -> new MetadataValue.Bool(in.readBoolean());
      case META_TAGS -> {
        int n = in.readInt();
        List<String> tags = new ArrayList<>(n);
        for (int i = 0; i < n; i++) tags.add(in.readUTF());
        yield new MetadataValue.Tags(tags);
      }
      default -> throw new IllegalArgumentException("unknown metadata type: " + type);
    };
  }
}
