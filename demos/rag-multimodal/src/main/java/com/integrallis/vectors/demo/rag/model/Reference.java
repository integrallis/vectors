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
package com.integrallis.vectors.demo.rag.model;

/**
 * A reference to a source location in the PDF document.
 *
 * @param page Page number (1-indexed)
 * @param type Content type (TEXT or IMAGE)
 * @param preview Short preview of the content
 */
public record Reference(int page, String type, String preview) {

  /**
   * Creates a reference with a truncated preview.
   *
   * @param page Page number
   * @param type Content type
   * @param content Full content text
   * @param maxLength Maximum preview length
   * @return Reference with truncated preview
   */
  public static Reference of(int page, String type, String content, int maxLength) {
    String preview = content;
    if (preview != null && preview.length() > maxLength) {
      preview = preview.substring(0, maxLength) + "...";
    }
    return new Reference(page, type, preview);
  }
}
