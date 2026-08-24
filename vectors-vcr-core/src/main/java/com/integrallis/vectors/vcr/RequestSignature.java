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

import java.lang.reflect.Array;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

/** Produces stable SHA-256 signatures from framework adapters' canonical request trees. */
public final class RequestSignature {

  private RequestSignature() {}

  /**
   * Hashes a request tree composed of nulls, strings, numbers, booleans, enums, maps, iterables,
   * and arrays. Map keys are sorted so insertion order does not affect the result.
   *
   * @param request canonical request tree
   * @return lowercase {@code sha256:...} signature
   */
  public static String create(Object request) {
    StringBuilder canonical = new StringBuilder();
    append(canonical, request);
    try {
      byte[] digest =
          MessageDigest.getInstance("SHA-256")
              .digest(canonical.toString().getBytes(StandardCharsets.UTF_8));
      return "sha256:" + java.util.HexFormat.of().formatHex(digest);
    } catch (NoSuchAlgorithmException e) {
      throw new IllegalStateException("SHA-256 is unavailable", e);
    }
  }

  private static void append(StringBuilder out, Object value) {
    if (value == null) {
      out.append("null");
    } else if (value instanceof CharSequence || value instanceof Character) {
      appendString(out, String.valueOf(value));
    } else if (value instanceof Number number) {
      out.append(number.getClass().getName()).append(':').append(number);
    } else if (value instanceof Boolean bool) {
      out.append(bool);
    } else if (value instanceof Enum<?> enumValue) {
      out.append(enumValue.getDeclaringClass().getName()).append(':').append(enumValue.name());
    } else if (value instanceof Map<?, ?> map) {
      out.append('{');
      List<Map.Entry<?, ?>> entries = new ArrayList<>(map.entrySet());
      entries.sort(Comparator.comparing(entry -> String.valueOf(entry.getKey())));
      for (Map.Entry<?, ?> entry : entries) {
        appendString(out, String.valueOf(entry.getKey()));
        out.append('=');
        append(out, entry.getValue());
        out.append(';');
      }
      out.append('}');
    } else if (value instanceof Iterable<?> iterable) {
      out.append('[');
      for (Object item : iterable) {
        append(out, item);
        out.append(';');
      }
      out.append(']');
    } else if (value.getClass().isArray()) {
      out.append('[');
      for (int i = 0; i < Array.getLength(value); i++) {
        append(out, Array.get(value, i));
        out.append(';');
      }
      out.append(']');
    } else {
      throw new IllegalArgumentException(
          "Unsupported request-signature value: " + value.getClass().getName());
    }
  }

  private static void appendString(StringBuilder out, String value) {
    out.append(value.length()).append(':').append(value);
  }
}
