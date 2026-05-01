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
package com.integrallis.vectors.studio.web.view;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * Small static helpers exposed to JTE templates (truncation, JSON pretty-print). Kept side-effect
 * free so templates can call them inline as {@code TemplateSupport.truncate(s, 240)}.
 */
public final class TemplateSupport {

  private static final ObjectMapper MAPPER =
      new ObjectMapper().enable(com.fasterxml.jackson.databind.SerializationFeature.INDENT_OUTPUT);

  private TemplateSupport() {}

  /**
   * Returns {@code s} clipped to {@code max} characters with a trailing ellipsis when truncated.
   * {@code null} becomes the empty string; {@code max <= 0} disables truncation.
   */
  public static String truncate(String s, int max) {
    if (s == null) return "";
    if (max <= 0 || s.length() <= max) return s;
    return s.substring(0, max) + "…";
  }

  /** Returns the number of pages required for {@code total} items at {@code limit} per page. */
  public static int pageCount(long total, int limit) {
    if (limit <= 0 || total <= 0) return 1;
    return (int) ((total + limit - 1) / limit);
  }

  /** Zero-based page index of {@code offset} for the given page size. */
  public static int currentPage(int offset, int limit) {
    if (limit <= 0) return 0;
    return offset / limit;
  }

  /**
   * Pretty-prints any value as indented JSON. {@code null} renders as the literal string "null".
   */
  public static String prettyJson(Object value) {
    if (value == null) return "null";
    try {
      return MAPPER.writeValueAsString(value);
    } catch (JsonProcessingException e) {
      return String.valueOf(value);
    }
  }
}
