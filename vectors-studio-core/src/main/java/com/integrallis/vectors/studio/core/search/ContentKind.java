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
package com.integrallis.vectors.studio.core.search;

import java.util.Map;

/**
 * Coarse classification of the payload carried by a {@link DocumentView}, used by the UI to pick a
 * dedicated viewer (truncated text, formatted JSON, inline image, opaque binary placeholder) and by
 * downstream tooling to decide whether to fetch the associated blob.
 *
 * <p>Detection is intentionally cheap and based on metadata hints first, falling back to content
 * sniffing on {@code text}. The order matters: a metadata {@code type=IMAGE} or {@code
 * mime=image/*} always wins over text shape.
 */
public enum ContentKind {
  TEXT,
  JSON,
  IMAGE,
  AUDIO,
  BINARY,
  EMPTY;

  /** Resolves the kind for a Studio {@link DocumentView}. */
  public static ContentKind detect(DocumentView view) {
    if (view == null) return EMPTY;
    return detect(view.text(), view.metadata());
  }

  /** Resolves the kind from raw text + metadata, both of which may be {@code null}. */
  public static ContentKind detect(String text, Map<String, Object> metadata) {
    ContentKind hinted = fromMetadata(metadata);
    if (hinted != null) return hinted;
    if (text == null || text.isEmpty()) return EMPTY;
    if (looksLikeJson(text)) return JSON;
    return TEXT;
  }

  private static ContentKind fromMetadata(Map<String, Object> metadata) {
    if (metadata == null || metadata.isEmpty()) return null;
    Object type = metadata.get("type");
    if (type instanceof String s) {
      String upper = s.toUpperCase();
      if (upper.equals("IMAGE")) return IMAGE;
      if (upper.equals("AUDIO")) return AUDIO;
      if (upper.equals("BINARY") || upper.equals("BLOB")) return BINARY;
      if (upper.equals("JSON")) return JSON;
      if (upper.equals("TEXT")) return TEXT;
    }
    Object mime = metadata.get("mime");
    if (mime == null) mime = metadata.get("contentType");
    if (mime instanceof String m) {
      String lower = m.toLowerCase();
      if (lower.startsWith("image/")) return IMAGE;
      if (lower.startsWith("audio/")) return AUDIO;
      if (lower.equals("application/json") || lower.endsWith("+json")) return JSON;
      if (lower.startsWith("text/")) return TEXT;
      if (lower.startsWith("application/octet-stream")) return BINARY;
    }
    return null;
  }

  private static boolean looksLikeJson(String text) {
    int i = 0;
    int n = text.length();
    while (i < n && Character.isWhitespace(text.charAt(i))) i++;
    if (i >= n) return false;
    char first = text.charAt(i);
    if (first != '{' && first != '[') return false;
    int j = n - 1;
    while (j > i && Character.isWhitespace(text.charAt(j))) j--;
    char last = text.charAt(j);
    return (first == '{' && last == '}') || (first == '[' && last == ']');
  }
}
