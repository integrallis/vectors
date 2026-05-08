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
package com.integrallis.vectors.studio.sidecart.sources;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Minimal JSON encoder / decoder shaped to D1's HTTP wire format. Avoids pulling Jackson into
 * {@code vectors-studio-sidecart} for a single endpoint. Supports nested objects / arrays, strings,
 * booleans, nulls, longs, and doubles — sufficient for the {@code /query} request and response
 * shape.
 */
final class D1Json {

  private D1Json() {}

  // ────────────────────────── encoding ──────────────────────────

  static String encode(Object value) {
    StringBuilder out = new StringBuilder();
    writeValue(out, value);
    return out.toString();
  }

  private static void writeValue(StringBuilder out, Object value) {
    if (value == null) {
      out.append("null");
    } else if (value instanceof String s) {
      writeString(out, s);
    } else if (value instanceof Boolean b) {
      out.append(b ? "true" : "false");
    } else if (value instanceof Number n) {
      out.append(n.toString());
    } else if (value instanceof Map<?, ?> m) {
      out.append('{');
      boolean first = true;
      for (Map.Entry<?, ?> e : m.entrySet()) {
        if (!first) out.append(',');
        writeString(out, String.valueOf(e.getKey()));
        out.append(':');
        writeValue(out, e.getValue());
        first = false;
      }
      out.append('}');
    } else if (value instanceof List<?> l) {
      out.append('[');
      for (int i = 0; i < l.size(); i++) {
        if (i > 0) out.append(',');
        writeValue(out, l.get(i));
      }
      out.append(']');
    } else if (value instanceof Object[] arr) {
      out.append('[');
      for (int i = 0; i < arr.length; i++) {
        if (i > 0) out.append(',');
        writeValue(out, arr[i]);
      }
      out.append(']');
    } else {
      writeString(out, value.toString());
    }
  }

  private static void writeString(StringBuilder out, String s) {
    out.append('"');
    for (int i = 0; i < s.length(); i++) {
      char c = s.charAt(i);
      switch (c) {
        case '"' -> out.append("\\\"");
        case '\\' -> out.append("\\\\");
        case '\n' -> out.append("\\n");
        case '\r' -> out.append("\\r");
        case '\t' -> out.append("\\t");
        case '\b' -> out.append("\\b");
        case '\f' -> out.append("\\f");
        default -> {
          if (c < 0x20) {
            out.append(String.format("\\u%04x", (int) c));
          } else {
            out.append(c);
          }
        }
      }
    }
    out.append('"');
  }

  // ────────────────────────── decoding ──────────────────────────

  static Object decode(String src) {
    Parser p = new Parser(src);
    p.skipWs();
    Object v = p.readValue();
    p.skipWs();
    if (p.pos != p.src.length()) {
      throw new IllegalArgumentException("trailing junk at " + p.pos);
    }
    return v;
  }

  /** Convenience: when the top level is known to be an object, return it as a Map. */
  @SuppressWarnings("unchecked")
  static Map<String, Object> decodeObject(String src) {
    Object v = decode(src);
    if (!(v instanceof Map)) {
      throw new IllegalArgumentException(
          "expected object, got " + (v == null ? "null" : v.getClass()));
    }
    return (Map<String, Object>) v;
  }

  static final class Parser {
    final String src;
    int pos;

    Parser(String src) {
      this.src = src;
    }

    void skipWs() {
      while (pos < src.length()) {
        char c = src.charAt(pos);
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') pos++;
        else break;
      }
    }

    Object readValue() {
      skipWs();
      if (pos >= src.length()) throw new IllegalArgumentException("unexpected EOF");
      char c = src.charAt(pos);
      return switch (c) {
        case '{' -> readObject();
        case '[' -> readArray();
        case '"' -> readString();
        case 't', 'f' -> readBool();
        case 'n' -> readNull();
        default -> readNumber();
      };
    }

    Map<String, Object> readObject() {
      expect('{');
      Map<String, Object> out = new LinkedHashMap<>();
      skipWs();
      if (peek() == '}') {
        pos++;
        return out;
      }
      while (true) {
        skipWs();
        String key = readString();
        skipWs();
        expect(':');
        out.put(key, readValue());
        skipWs();
        char c = src.charAt(pos);
        if (c == ',') {
          pos++;
          continue;
        }
        if (c == '}') {
          pos++;
          return out;
        }
        throw new IllegalArgumentException("expected , or } at " + pos);
      }
    }

    List<Object> readArray() {
      expect('[');
      List<Object> out = new ArrayList<>();
      skipWs();
      if (peek() == ']') {
        pos++;
        return out;
      }
      while (true) {
        out.add(readValue());
        skipWs();
        char c = src.charAt(pos);
        if (c == ',') {
          pos++;
          continue;
        }
        if (c == ']') {
          pos++;
          return out;
        }
        throw new IllegalArgumentException("expected , or ] at " + pos);
      }
    }

    String readString() {
      expect('"');
      StringBuilder sb = new StringBuilder();
      while (pos < src.length()) {
        char c = src.charAt(pos++);
        if (c == '"') return sb.toString();
        if (c == '\\') {
          char esc = src.charAt(pos++);
          switch (esc) {
            case '"' -> sb.append('"');
            case '\\' -> sb.append('\\');
            case '/' -> sb.append('/');
            case 'n' -> sb.append('\n');
            case 'r' -> sb.append('\r');
            case 't' -> sb.append('\t');
            case 'b' -> sb.append('\b');
            case 'f' -> sb.append('\f');
            case 'u' -> {
              String hex = src.substring(pos, pos + 4);
              pos += 4;
              sb.append((char) Integer.parseInt(hex, 16));
            }
            default -> throw new IllegalArgumentException("bad escape: \\" + esc);
          }
        } else {
          sb.append(c);
        }
      }
      throw new IllegalArgumentException("unterminated string");
    }

    Boolean readBool() {
      if (src.startsWith("true", pos)) {
        pos += 4;
        return Boolean.TRUE;
      }
      if (src.startsWith("false", pos)) {
        pos += 5;
        return Boolean.FALSE;
      }
      throw new IllegalArgumentException("bad literal at " + pos);
    }

    Object readNull() {
      if (src.startsWith("null", pos)) {
        pos += 4;
        return null;
      }
      throw new IllegalArgumentException("bad literal at " + pos);
    }

    Number readNumber() {
      int start = pos;
      if (peek() == '-') pos++;
      while (pos < src.length()) {
        char c = src.charAt(pos);
        if ((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') {
          pos++;
        } else break;
      }
      String num = src.substring(start, pos);
      if (num.indexOf('.') >= 0 || num.indexOf('e') >= 0 || num.indexOf('E') >= 0) {
        return Double.parseDouble(num);
      }
      try {
        return Long.parseLong(num);
      } catch (NumberFormatException e) {
        return Double.parseDouble(num);
      }
    }

    char peek() {
      return src.charAt(pos);
    }

    void expect(char c) {
      if (pos >= src.length() || src.charAt(pos) != c) {
        throw new IllegalArgumentException("expected '" + c + "' at " + pos);
      }
      pos++;
    }
  }
}
