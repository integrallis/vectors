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

import java.awt.geom.Rectangle2D;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import org.apache.pdfbox.Loader;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDPage;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

/** Diagnostic test for Jefferson-Amazon.pdf — not part of the regular suite. */
class JeffersonAmazonDiagnosticTest {

  private static final Path PDF =
      Path.of(System.getProperty("user.home"), "Downloads", "Jefferson-Amazon.pdf");

  static boolean pdfExists() {
    return PDF.toFile().exists();
  }

  @Test
  @EnabledIf("pdfExists")
  void diagnoseExtraction() throws IOException {
    byte[] pdfBytes = java.nio.file.Files.readAllBytes(PDF);

    System.out.println("=== TIER 1: Embedded images ===");
    var embedded = PDFImageExtractor.extractEmbeddedImages(pdfBytes, 100, 100);
    System.out.println("  Raw embedded: " + embedded.size());
    for (var img : embedded) {
      System.out.printf(
          "    page=%d idx=%d %dx%d %s informative=%b%n",
          img.pageNumber(),
          img.imageIndex(),
          img.width(),
          img.height(),
          img.format(),
          PDFImageExtractor.isInformative(img));
    }

    var informative = PDFImageExtractor.filterDecorative(embedded);
    System.out.println("  Informative after filter: " + informative.size());

    System.out.println("\n=== TIER 2: Vector region detection ===");
    try (PDDocument doc = Loader.loadPDF(pdfBytes)) {
      for (int i = 0; i < doc.getNumberOfPages(); i++) {
        PDPage page = doc.getPage(i);
        VectorRegionDetector detector = new VectorRegionDetector(page);
        List<Rectangle2D.Float> regions = detector.detectRegions();
        if (!regions.isEmpty()) {
          System.out.printf("  Page %d: %d regions%n", i + 1, regions.size());
          for (Rectangle2D.Float r : regions) {
            System.out.printf("    x=%.1f y=%.1f w=%.1f h=%.1f%n", r.x, r.y, r.width, r.height);
          }
        } else {
          System.out.printf("  Page %d: 0 regions%n", i + 1);
        }
      }
    }

    System.out.println("\n=== TIER 2: extractVectorRegions() ===");
    var vectorRegions = PDFImageExtractor.extractVectorRegions(pdfBytes);
    System.out.println("  Vector region images: " + vectorRegions.size());
    for (var img : vectorRegions) {
      System.out.printf(
          "    page=%d idx=%d %dx%d caption=%s%n",
          img.pageNumber(), img.imageIndex(), img.width(), img.height(), img.caption());
    }

    System.out.println("\n=== Full extraction (three-tier) ===");
    try (var fis = new FileInputStream(PDF.toFile())) {
      var all = PDFImageExtractor.extractImages(fis);
      System.out.println("  Total images: " + all.size());
      for (var img : all) {
        System.out.printf(
            "    page=%d idx=%d %dx%d source=%s caption=%s%n",
            img.pageNumber(),
            img.imageIndex(),
            img.width(),
            img.height(),
            img.source(),
            img.caption());
      }
    }
  }
}
