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
package com.integrallis.vectors.vcr.serde.avaje;

import com.integrallis.vectors.vcr.CassetteRecord;
import com.integrallis.vectors.vcr.CassetteSerializer;
import com.integrallis.vectors.vcr.CassetteTreeCodec;
import io.avaje.jsonb.JsonType;
import io.avaje.jsonb.Jsonb;
import java.util.Map;

/**
 * {@link CassetteSerializer} backed by Avaje {@code Jsonb}.
 *
 * <p>The shared {@link CassetteTreeCodec} owns the cassette shape, while this adapter handles only
 * the Avaje JSON byte boundary. No annotation processor is required.
 */
public final class AvajeCassetteSerializer implements CassetteSerializer {

  private final JsonType<Object> anyType = Jsonb.builder().build().type(Object.class);

  /**
   * Serializes one cassette through the canonical object tree.
   *
   * @param record cassette record to encode
   * @return encoded JSON bytes
   */
  @Override
  public byte[] serialize(CassetteRecord record) {
    return anyType.toJsonBytes(CassetteTreeCodec.toTree(record));
  }

  /**
   * Deserializes a cassette written by either supported JSON adapter.
   *
   * @param bytes encoded cassette JSON
   * @return decoded cassette record
   * @throws IllegalArgumentException if the JSON root or cassette shape is malformed
   */
  @Override
  public CassetteRecord deserialize(byte[] bytes) {
    Object parsed = anyType.fromJson(bytes);
    if (!(parsed instanceof Map<?, ?> tree)) {
      throw new IllegalArgumentException("expected JSON object at top level");
    }
    return CassetteTreeCodec.fromTree(tree);
  }
}
