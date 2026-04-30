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
 * Represents a chunk of a document with multimodal content.
 *
 * @param id Unique chunk identifier
 * @param documentId Parent document identifier
 * @param pageNumber Page number (for PDFs)
 * @param textSummary Text summary for embedding/search
 * @param imageData Raw image data (can be null)
 * @param imageFormat Image format string e.g. "JPEG", "PNG" (null for non-image chunks)
 * @param chunkType Type of chunk (TEXT, IMAGE, TABLE)
 */
public record DocumentChunk(
    String id,
    String documentId,
    int pageNumber,
    String textSummary,
    byte[] imageData,
    String imageFormat,
    ChunkType chunkType) {

  public enum ChunkType {
    TEXT,
    IMAGE,
    /** Full-page render used as visual context for the LLM, not a user-displayable image. */
    PAGE_RENDER,
    TABLE
  }

  /**
   * Creates a text chunk with a deterministic ID based on documentId, page, and type. Deterministic
   * IDs ensure that re-uploading the same PDF replaces existing chunks via upsert rather than
   * creating duplicates.
   *
   * @param documentId Document ID
   * @param pageNumber Page number
   * @param textContent Text content
   * @return Text chunk
   */
  public static DocumentChunk text(String documentId, int pageNumber, String textContent) {
    String id = documentId + "_text_p" + pageNumber;
    return new DocumentChunk(id, documentId, pageNumber, textContent, null, null, ChunkType.TEXT);
  }

  /**
   * Creates an image chunk with a deterministic ID.
   *
   * @param documentId Document ID
   * @param pageNumber Page number
   * @param imageIndex 0-based image index on the page
   * @param summary Text summary for search
   * @param imageData Raw image bytes
   * @return Image chunk
   */
  public static DocumentChunk image(
      String documentId,
      int pageNumber,
      int imageIndex,
      String summary,
      byte[] imageData,
      String imageFormat) {
    String id = documentId + "_img_p" + pageNumber + "_" + imageIndex;
    return new DocumentChunk(
        id, documentId, pageNumber, summary, imageData, imageFormat, ChunkType.IMAGE);
  }

  /**
   * Creates a page-render chunk with a deterministic ID. Page renders are full-page raster images
   * used as visual context for the LLM but not listed or displayable as individual images.
   *
   * @param documentId Document ID
   * @param pageNumber Page number
   * @param summary Text summary for search
   * @param imageData Raw image bytes
   * @param imageFormat Image format (e.g. "PNG")
   * @return Page-render chunk
   */
  public static DocumentChunk pageRender(
      String documentId, int pageNumber, String summary, byte[] imageData, String imageFormat) {
    String id = documentId + "_render_p" + pageNumber;
    return new DocumentChunk(
        id, documentId, pageNumber, summary, imageData, imageFormat, ChunkType.PAGE_RENDER);
  }

  /**
   * Creates a metadata chunk with a deterministic ID. Stores document-level metadata (title,
   * author, page count, etc.) as searchable text.
   *
   * @param documentId Document ID
   * @param metadataText Formatted metadata text
   * @return Metadata chunk
   */
  public static DocumentChunk metadata(String documentId, String metadataText) {
    String id = documentId + "_metadata";
    return new DocumentChunk(id, documentId, 0, metadataText, null, null, ChunkType.TEXT);
  }

  /**
   * Checks if this chunk has image data.
   *
   * @return true if image data present
   */
  public boolean hasImage() {
    return imageData != null && imageData.length > 0;
  }
}
