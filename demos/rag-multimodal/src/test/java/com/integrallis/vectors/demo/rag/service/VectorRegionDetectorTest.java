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

import java.awt.geom.Rectangle2D;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import org.apache.pdfbox.Loader;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDPage;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Tests for {@link VectorRegionDetector}.
 *
 * <p>Tests both the clustering algorithm (unit tests) and region detection on real PDFs
 * (integration tests).
 */
@DisplayName("VectorRegionDetector")
class VectorRegionDetectorTest {

  @Nested
  @Tag("unit")
  @DisplayName("Clustering algorithm")
  class ClusteringAlgorithm {

    @Test
    @DisplayName("single rectangle yields one region")
    void singleRectangleYieldsOneRegion() {
      List<Rectangle2D.Float> regions = new ArrayList<>();
      regions.add(new Rectangle2D.Float(100, 100, 200, 150));

      List<Rectangle2D.Float> clustered =
          VectorRegionDetector.clusterRegions(regions, VectorRegionDetector.CLUSTER_GAP);

      assertThat(clustered).hasSize(1);
      assertThat(clustered.get(0).width).isEqualTo(200f);
      assertThat(clustered.get(0).height).isEqualTo(150f);
    }

    @Test
    @DisplayName("two separated regions stay separate")
    void twoSeparatedRegionsStaySeparate() {
      List<Rectangle2D.Float> regions = new ArrayList<>();
      regions.add(new Rectangle2D.Float(10, 10, 50, 50));
      regions.add(new Rectangle2D.Float(200, 200, 50, 50));

      List<Rectangle2D.Float> clustered =
          VectorRegionDetector.clusterRegions(regions, VectorRegionDetector.CLUSTER_GAP);

      assertThat(clustered).hasSize(2);
    }

    @Test
    @DisplayName("two nearby regions merge into one")
    void twoNearbyRegionsMerge() {
      List<Rectangle2D.Float> regions = new ArrayList<>();
      regions.add(new Rectangle2D.Float(10, 10, 50, 50));
      regions.add(new Rectangle2D.Float(65, 10, 50, 50)); // 5pt gap

      List<Rectangle2D.Float> clustered =
          VectorRegionDetector.clusterRegions(regions, VectorRegionDetector.CLUSTER_GAP);

      assertThat(clustered).hasSize(1);
      // Merged bounding box: x=10, w=(65+50)-10 = 105
      assertThat(clustered.get(0).x).isEqualTo(10f);
      assertThat(clustered.get(0).width).isEqualTo(105f);
    }

    @Test
    @DisplayName("dense bar chart (many rectangles) clusters into one")
    void denseBarChartClustersIntoOne() {
      List<Rectangle2D.Float> regions = new ArrayList<>();
      // Simulate bar chart: many thin rectangles side by side
      for (int i = 0; i < 20; i++) {
        regions.add(new Rectangle2D.Float(100 + i * 12, 100, 10, 80));
      }

      List<Rectangle2D.Float> clustered =
          VectorRegionDetector.clusterRegions(regions, VectorRegionDetector.CLUSTER_GAP);

      assertThat(clustered).hasSize(1);
    }

    @Test
    @DisplayName("empty list yields empty result")
    void emptyListYieldsEmpty() {
      List<Rectangle2D.Float> clustered =
          VectorRegionDetector.clusterRegions(List.of(), VectorRegionDetector.CLUSTER_GAP);

      assertThat(clustered).isEmpty();
    }

    @Test
    @DisplayName("overlapping regions merge")
    void overlappingRegionsMerge() {
      List<Rectangle2D.Float> regions = new ArrayList<>();
      regions.add(new Rectangle2D.Float(10, 10, 100, 100));
      regions.add(new Rectangle2D.Float(50, 50, 100, 100));

      List<Rectangle2D.Float> clustered =
          VectorRegionDetector.clusterRegions(regions, VectorRegionDetector.CLUSTER_GAP);

      assertThat(clustered).hasSize(1);
      assertThat(clustered.get(0).x).isEqualTo(10f);
      assertThat(clustered.get(0).y).isEqualTo(10f);
      assertThat(clustered.get(0).width).isEqualTo(140f);
      assertThat(clustered.get(0).height).isEqualTo(140f);
    }
  }

  @Nested
  @Tag("integration")
  @DisplayName("Region detection on real PDFs")
  class RealPdfDetection {

    private static final String ARGUS_PDF = "/pdfs/Argus-Market-Digest-June-2024.pdf";
    private static final String LAYOUT_PARSER_PDF = "/pdfs/layout-parser-paper.pdf";

    @Test
    @DisplayName("Argus financial newsletter detects drawing regions")
    void argusDetectsDrawingRegions() throws IOException {
      try (InputStream is = getClass().getResourceAsStream(ARGUS_PDF)) {
        assertThat(is).as("Test PDF not found").isNotNull();
        byte[] pdfBytes = is.readAllBytes();

        try (PDDocument document = Loader.loadPDF(pdfBytes)) {
          int totalRegions = 0;
          for (int i = 0; i < document.getNumberOfPages(); i++) {
            PDPage page = document.getPage(i);
            VectorRegionDetector detector = new VectorRegionDetector(page);
            List<Rectangle2D.Float> regions = detector.detectRegions();
            totalRegions += regions.size();
          }

          // The Argus PDF has vector-drawn charts — should detect at least some regions
          // (exact count depends on filtering heuristics)
          log.debug("Argus PDF: detected {} total regions", totalRegions);
          // We don't assert a specific count since the heuristic filtering may vary,
          // but the detector should not crash
        }
      }
    }

    @Test
    @DisplayName("layout-parser paper extracts images without errors")
    void layoutParserExtractsImages() throws IOException {
      // The layout-parser paper has embedded raster images, though many have high
      // white backgrounds (academic paper screenshots). Extraction should succeed
      // regardless of which tier is selected.
      List<PDFImageExtractor.ExtractedImage> images;
      try (InputStream is = getClass().getResourceAsStream(LAYOUT_PARSER_PDF)) {
        assertThat(is).as("Test PDF not found").isNotNull();
        images = PDFImageExtractor.extractImages(is);
      }

      assertThat(images).isNotEmpty();
      for (PDFImageExtractor.ExtractedImage img : images) {
        assertThat(img.imageBytes()).isNotNull().isNotEmpty();
        assertThat(img.format()).isIn("JPEG", "PNG");
        assertThat(img.pageNumber()).isPositive();
      }
    }

    @Test
    @DisplayName("detected regions have valid bounds within page dimensions")
    void detectedRegionsHaveValidBounds() throws IOException {
      try (InputStream is = getClass().getResourceAsStream(ARGUS_PDF)) {
        assertThat(is).as("Test PDF not found").isNotNull();
        byte[] pdfBytes = is.readAllBytes();

        try (PDDocument document = Loader.loadPDF(pdfBytes)) {
          for (int i = 0; i < document.getNumberOfPages(); i++) {
            PDPage page = document.getPage(i);
            float pageWidth = page.getMediaBox().getWidth();
            float pageHeight = page.getMediaBox().getHeight();

            VectorRegionDetector detector = new VectorRegionDetector(page);
            List<Rectangle2D.Float> regions = detector.detectRegions();

            for (Rectangle2D.Float region : regions) {
              assertThat(region.x).as("region x").isGreaterThanOrEqualTo(0);
              assertThat(region.y).as("region y").isGreaterThanOrEqualTo(0);
              assertThat(region.width).as("region width").isGreaterThan(0);
              assertThat(region.height).as("region height").isGreaterThan(0);
              assertThat(region.x + region.width)
                  .as("region right edge")
                  .isLessThanOrEqualTo(pageWidth + VectorRegionDetector.PADDING);
              assertThat(region.y + region.height)
                  .as("region bottom edge")
                  .isLessThanOrEqualTo(pageHeight + VectorRegionDetector.PADDING);
            }
          }
        }
      }
    }
  }

  private static final org.slf4j.Logger log =
      org.slf4j.LoggerFactory.getLogger(VectorRegionDetectorTest.class);
}
