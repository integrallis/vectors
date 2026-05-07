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
package com.integrallis.vectors.storage.wal;

/**
 * Minimal cursor-based JSON reader covering exactly the subset emitted by {@link Manifest}: nested
 * objects, arrays, integers, booleans, and double-quoted ASCII strings. Throws {@link
 * RuntimeException} on any malformed input; callers wrap into an {@code IOException}.
 */
final class Cursor {
  private final String s;
  private int p;

  Cursor(String s) {
    this.s = s;
    this.p = 0;
  }

  char peek() {
    if (p >= s.length()) throw new RuntimeException("unexpected EOF");
    return s.charAt(p);
  }

  char next() {
    if (p >= s.length()) throw new RuntimeException("unexpected EOF");
    return s.charAt(p++);
  }

  void expect(char c) {
    char got = next();
    if (got != c) throw new RuntimeException("expected '" + c + "' got '" + got + "' at " + (p - 1));
  }

  void skipWs() {
    while (p < s.length()) {
      char c = s.charAt(p);
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r') p++;
      else break;
    }
  }

  /** Reads a string up to the closing double-quote. The opening quote must already be consumed. */
  String readString() {
    StringBuilder sb = new StringBuilder();
    while (true) {
      char c = next();
      if (c == '"') return sb.toString();
      if (c == '\\') sb.append(next());
      else sb.append(c);
    }
  }

  long readLong() {
    int start = p;
    if (p < s.length() && s.charAt(p) == '-') p++;
    while (p < s.length() && Character.isDigit(s.charAt(p))) p++;
    if (start == p) throw new RuntimeException("expected number at " + p);
    return Long.parseLong(s.substring(start, p));
  }

  boolean readBool() {
    if (s.startsWith("true", p)) {
      p += 4;
      return true;
    }
    if (s.startsWith("false", p)) {
      p += 5;
      return false;
    }
    throw new RuntimeException("expected boolean at " + p);
  }

  /** Skips a single JSON value (object/array/string/number/literal). */
  void skipValue() {
    skipWs();
    char c = peek();
    if (c == '{' || c == '[') {
      char open = next();
      char close = open == '{' ? '}' : ']';
      int depth = 1;
      while (depth > 0) {
        char x = next();
        if (x == '"') readString();
        else if (x == open) depth++;
        else if (x == close) depth--;
      }
    } else if (c == '"') {
      next();
      readString();
    } else if (c == 't' || c == 'f') {
      readBool();
    } else if (c == '-' || Character.isDigit(c)) {
      readLong();
    } else {
      throw new RuntimeException("unexpected '" + c + "' at " + p);
    }
  }
}
