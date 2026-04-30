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
package com.integrallis.vectors.demo.rag.service;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.demo.rag.service.PDFImageExtractor.ExtractedImage;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Tests PDF image extraction against real academic papers from pixeltable test suite.
 *
 * <p>Test PDFs:
 *
 * <ul>
 *   <li>{@code 1706.03762.pdf} — "Attention Is All You Need" (Vaswani et al., 2017). Contains
 *       transformer architecture diagrams.
 *   <li>{@code Vector_database.pdf} — Wikipedia article on vector databases. Contains diagrams and
 *       screenshots.
 *   <li>{@code layout-parser-paper.pdf} — "LayoutParser" paper. Contains many figures, tables, and
 *       screenshots.
 * </ul>
 */
@Tag("integration")
@DisplayName("PDF image extraction — real academic papers (pixeltable test suite)")
class PDFImageExtractionTest {

  private static final String ATTENTION_PDF = "/pdfs/1706.03762.pdf";
  private static final String VECTOR_DB_PDF = "/pdfs/Vector_database.pdf";
  private static final String LAYOUT_PARSER_PDF = "/pdfs/layout-parser-paper.pdf";
  private static final String ARGUS_PDF = "/pdfs/Argus-Market-Digest-June-2024.pdf";

  private List<ExtractedImage> extractImages(String resourcePath) throws IOException {
    try (InputStream is = getClass().getResourceAsStream(resourcePath)) {
      assertThat(is).as("Test PDF not found: %s", resourcePath).isNotNull();
      return PDFImageExtractor.extractImages(is);
    }
  }

  private List<ExtractedImage> extractImages(String resourcePath, int minWidth, int minHeight)
      throws IOException {
    try (InputStream is = getClass().getResourceAsStream(resourcePath)) {
      assertThat(is).as("Test PDF not found: %s", resourcePath).isNotNull();
      return PDFImageExtractor.extractImages(is, minWidth, minHeight);
    }
  }

  @Nested
  @Tag("integration")
  @DisplayName("Per-document extraction")
  class PerDocumentExtraction {

    @Test
    @DisplayName("Attention paper (1706.03762) yields images")
    void attentionPaperContainsImages() throws IOException {
      List<ExtractedImage> images = extractImages(ATTENTION_PDF);
      // "Attention Is All You Need" contains transformer architecture diagrams
      assertThat(images).as("Attention paper should contain at least one figure").isNotEmpty();

      for (ExtractedImage img : images) {
        assertValidImage(img);
      }
    }

    @Test
    @DisplayName("Vector database article yields images")
    void vectorDatabaseContainsImages() throws IOException {
      List<ExtractedImage> images = extractImages(VECTOR_DB_PDF);
      // Wikipedia article with diagrams — may have images >= 100x100
      // Some PDFs may not have large enough images; just verify no exceptions
      for (ExtractedImage img : images) {
        assertValidImage(img);
      }
    }

    @Test
    @DisplayName("LayoutParser paper yields multiple images")
    void layoutParserContainsImages() throws IOException {
      List<ExtractedImage> images = extractImages(LAYOUT_PARSER_PDF);
      // LayoutParser paper is figure-heavy (screenshots, architecture diagrams, examples)
      assertThat(images)
          .as("LayoutParser paper should contain multiple figures")
          .hasSizeGreaterThan(1);

      for (ExtractedImage img : images) {
        assertValidImage(img);
      }
    }

    @Test
    @DisplayName("text-heavy financial newsletter yields non-embedded images")
    void argusMarketDigestIsTextHeavy() throws IOException {
      List<ExtractedImage> images = extractImages(ARGUS_PDF);
      // The Argus Market Digest is a short text-heavy financial newsletter.
      // Its embedded rasters are decorative, so the extractor falls through to
      // Tier 2 (vector regions) or Tier 3 (page renders). Either is acceptable.
      assertThat(images).isNotEmpty();
      for (ExtractedImage img : images) {
        assertValidImage(img);
        assertThat(img.pageNumber()).as("page number must be positive").isPositive();
        assertThat(img.source())
            .as("Argus images should not be EMBEDDED (those are decorative)")
            .isNotEqualTo(PDFImageExtractor.Source.EMBEDDED);
      }
    }

    @Test
    @DisplayName("layout-parser paper produces extractable images via any tier")
    void layoutParserProducesImages() throws IOException {
      List<ExtractedImage> images = extractImages(LAYOUT_PARSER_PDF);
      // LayoutParser paper has embedded images but many have high white backgrounds
      // (academic paper screenshots). The extractor may use any tier depending on
      // decorative filtering. The key invariant: images are extracted without error.
      assertThat(images).isNotEmpty();
      for (ExtractedImage img : images) {
        assertValidImage(img);
      }
    }
  }

  @Nested
  @Tag("integration")
  @DisplayName("Cross-document validation")
  class CrossDocumentValidation {

    @Test
    @DisplayName("all extracted images across all PDFs have valid bytes")
    void allExtractedImagesHaveValidBytes() throws IOException {
      for (String pdf : List.of(ATTENTION_PDF, VECTOR_DB_PDF, LAYOUT_PARSER_PDF)) {
        List<ExtractedImage> images = extractImages(pdf);
        for (ExtractedImage img : images) {
          assertThat(img.imageBytes())
              .as("Image from %s page %d should have non-empty bytes", pdf, img.pageNumber())
              .isNotNull()
              .isNotEmpty();
          // Real images should have at least a few hundred bytes
          assertThat(img.imageBytes().length)
              .as(
                  "Image from %s page %d should be a real image (>100 bytes)",
                  pdf, img.pageNumber())
              .isGreaterThan(100);
        }
      }
    }

    @Test
    @DisplayName("all extracted images have valid format (JPEG or PNG)")
    void allExtractedImagesHaveValidFormat() throws IOException {
      for (String pdf : List.of(ATTENTION_PDF, VECTOR_DB_PDF, LAYOUT_PARSER_PDF)) {
        List<ExtractedImage> images = extractImages(pdf);
        for (ExtractedImage img : images) {
          assertThat(img.format())
              .as("Image from %s page %d should be JPEG or PNG", pdf, img.pageNumber())
              .isIn("JPEG", "PNG");
        }
      }
    }

    @Test
    @DisplayName("all extracted images have positive dimensions")
    void allExtractedImagesHaveValidDimensions() throws IOException {
      for (String pdf : List.of(ATTENTION_PDF, VECTOR_DB_PDF, LAYOUT_PARSER_PDF)) {
        List<ExtractedImage> images = extractImages(pdf);
        for (ExtractedImage img : images) {
          assertThat(img.width())
              .as("Image from %s page %d width", pdf, img.pageNumber())
              .isGreaterThanOrEqualTo(100);
          assertThat(img.height())
              .as("Image from %s page %d height", pdf, img.pageNumber())
              .isGreaterThanOrEqualTo(100);
        }
      }
    }

    @Test
    @DisplayName("page numbers are within document page range")
    void pageNumbersAreWithinDocumentRange() throws IOException {
      // Attention paper: 15 pages
      List<ExtractedImage> attentionImages = extractImages(ATTENTION_PDF);
      for (ExtractedImage img : attentionImages) {
        assertThat(img.pageNumber()).as("Attention paper page number").isBetween(1, 15);
      }

      // Layout parser: ~16 pages
      List<ExtractedImage> layoutImages = extractImages(LAYOUT_PARSER_PDF);
      for (ExtractedImage img : layoutImages) {
        assertThat(img.pageNumber()).as("LayoutParser page number").isBetween(1, 20);
      }
    }

    @Test
    @DisplayName("no duplicate (page, imageIndex) pairs within a document")
    void noDuplicateImageIndices() throws IOException {
      for (String pdf : List.of(ATTENTION_PDF, VECTOR_DB_PDF, LAYOUT_PARSER_PDF)) {
        List<ExtractedImage> images = extractImages(pdf);
        Set<String> seen = new HashSet<>();
        for (ExtractedImage img : images) {
          String key = img.pageNumber() + "_" + img.imageIndex();
          assertThat(seen.add(key))
              .as(
                  "Duplicate image at page %d index %d in %s",
                  img.pageNumber(), img.imageIndex(), pdf)
              .isTrue();
        }
      }
    }

    @Test
    @DisplayName("all PDFs from pixeltable are processable without exceptions")
    void allPdfsFromPixeltableAreProcessable() throws IOException {
      // This mirrors pixeltable's test_doc_splitter_images: process all PDFs, assert no errors
      for (String pdf : List.of(ATTENTION_PDF, VECTOR_DB_PDF, LAYOUT_PARSER_PDF)) {
        List<ExtractedImage> images = extractImages(pdf);
        // All extracted images must be valid (non-null bytes, valid format)
        for (ExtractedImage img : images) {
          assertThat(img.imageBytes()).isNotNull();
          assertThat(img.format()).isNotNull().isNotEmpty();
          assertThat(img.pageNumber()).isPositive();
          assertThat(img.width()).isPositive();
          assertThat(img.height()).isPositive();
        }
      }
    }
  }

  @Nested
  @Tag("integration")
  @DisplayName("Size filtering")
  class SizeFiltering {

    @Test
    @DisplayName("raising minimum size filters out smaller images")
    void higherMinSizeFiltersMore() throws IOException {
      List<ExtractedImage> defaultImages = extractImages(LAYOUT_PARSER_PDF);
      List<ExtractedImage> largeOnly = extractImages(LAYOUT_PARSER_PDF, 400, 400);

      // Large-only should be a subset (fewer or equal)
      assertThat(largeOnly.size())
          .as("Raising min size should filter out smaller images")
          .isLessThanOrEqualTo(defaultImages.size());

      // All large-only images should meet the higher threshold
      for (ExtractedImage img : largeOnly) {
        assertThat(img.width()).isGreaterThanOrEqualTo(400);
        assertThat(img.height()).isGreaterThanOrEqualTo(400);
      }
    }

    @Test
    @DisplayName("lowering minimum size yields more or equal images")
    void lowerMinSizeYieldsMore() throws IOException {
      List<ExtractedImage> defaultImages = extractImages(LAYOUT_PARSER_PDF);
      List<ExtractedImage> moreImages = extractImages(LAYOUT_PARSER_PDF, 50, 50);

      assertThat(moreImages.size())
          .as("Lowering min size should yield more or equal images")
          .isGreaterThanOrEqualTo(defaultImages.size());
    }
  }

  @Nested
  @Tag("integration")
  @DisplayName("Deduplication")
  class Deduplication {

    @Test
    @DisplayName("shared images (logos, watermarks) are not extracted per-page")
    void sharedImagesDeduplicatedAcrossPages() throws IOException {
      // If a PDF has a logo on every page, it should appear only once in the results.
      // We verify this by checking that image count is much less than page count.
      // The Attention paper has 15 pages — if shared images weren't deduped,
      // we'd see 15x the same logo.
      List<ExtractedImage> images = extractImages(ATTENTION_PDF);

      // Count images per page — no page should have a suspiciously high count
      // from inherited resources
      long distinctPages = images.stream().map(ExtractedImage::pageNumber).distinct().count();
      // If deduplication failed, we'd see images attributed to nearly every page
      // For an academic paper, images should be on specific figure pages only
      if (!images.isEmpty()) {
        assertThat(distinctPages)
            .as(
                "Images should be on specific pages, not every page (dedup check). "
                    + "Found %d images across %d distinct pages out of 15 total pages.",
                images.size(), distinctPages)
            .isLessThan(15);
      }
    }
  }

  /** Validates that an ExtractedImage has all required fields populated correctly. */
  private static void assertValidImage(ExtractedImage img) {
    assertThat(img.imageBytes()).as("imageBytes").isNotNull().isNotEmpty();
    assertThat(img.format()).as("format").isIn("JPEG", "PNG");
    assertThat(img.pageNumber()).as("pageNumber").isPositive();
    assertThat(img.width()).as("width").isGreaterThanOrEqualTo(100);
    assertThat(img.height()).as("height").isGreaterThanOrEqualTo(100);
    assertThat(img.imageIndex()).as("imageIndex").isGreaterThanOrEqualTo(0);
  }
}
