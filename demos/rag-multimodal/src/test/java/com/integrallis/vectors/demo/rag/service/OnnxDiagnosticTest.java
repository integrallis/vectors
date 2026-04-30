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

import java.awt.image.BufferedImage;
import java.io.FileInputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.apache.pdfbox.Loader;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.rendering.ImageType;
import org.apache.pdfbox.rendering.PDFRenderer;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

/**
 * Diagnostic test: runs DocLayout-YOLO detector on each page of Jefferson-Amazon.pdf and dumps ALL
 * raw detections (no filtering) so we can see exactly what the model produces.
 */
@Tag("slow")
class OnnxDiagnosticTest {

  private static final Path MODEL_PATH =
      Path.of(
          System.getProperty("user.home"),
          ".java-vectors",
          "models",
          "doclayout_yolo_docstructbench_imgsz1024.onnx");
  private static final Path PDF_PATH =
      Path.of(System.getProperty("user.home"), "Downloads", "Jefferson-Amazon.pdf");

  static boolean canRun() {
    return Files.exists(MODEL_PATH) && Files.exists(PDF_PATH);
  }

  /** Dump ALL raw detections from ONNX on every page — no class/confidence filtering. */
  @Test
  @EnabledIf("canRun")
  void dumpAllDetectionsPerPage() throws Exception {
    String[] classLabels = {
      "Title",
      "Plain Text",
      "Abandoned Text",
      "Figure",
      "Figure Caption",
      "Table",
      "Table Caption",
      "Table Footnote",
      "Isolated Formula",
      "Formula Caption"
    };

    try (OnnxLayoutDetector detector = new OnnxLayoutDetector(MODEL_PATH);
        PDDocument doc = Loader.loadPDF(PDF_PATH.toFile())) {

      PDFRenderer renderer = new PDFRenderer(doc);
      int numPages = doc.getNumberOfPages();

      System.out.println("=== ONNX DIAGNOSTIC: " + numPages + " pages ===\n");

      for (int page = 0; page < numPages; page++) {
        BufferedImage img = renderer.renderImageWithDPI(page, 150, ImageType.RGB);
        System.out.printf("--- Page %d (%dx%d) ---%n", page + 1, img.getWidth(), img.getHeight());

        // Run detection with VERY low threshold to see everything
        List<OnnxLayoutDetector.Detection> dets = detector.detect(img, 0.1f);
        if (dets.isEmpty()) {
          System.out.println("  (no detections at all)");
        }
        for (OnnxLayoutDetector.Detection d : dets) {
          String label =
              (d.classId() >= 0 && d.classId() < classLabels.length)
                  ? classLabels[d.classId()]
                  : "?";
          float areaPercent = 100f * d.width() * d.height() / (img.getWidth() * img.getHeight());
          System.out.printf(
              "  class=%d (%s)  conf=%.3f  bbox=[%d,%d %dx%d]  area=%.1f%%%n",
              d.classId(), label, d.confidence(), d.x(), d.y(), d.width(), d.height(), areaPercent);
        }
        System.out.println();
      }
    }
  }

  /** Also dump what PDFImageExtractor produces for comparison. */
  @Test
  @EnabledIf("canRun")
  void dumpExtractedImages() throws Exception {
    try (FileInputStream fis = new FileInputStream(PDF_PATH.toFile())) {
      List<PDFImageExtractor.ExtractedImage> images = PDFImageExtractor.extractImages(fis);
      System.out.println("=== PDFImageExtractor: " + images.size() + " images ===\n");
      for (PDFImageExtractor.ExtractedImage img : images) {
        System.out.printf(
            "  page=%d  source=%s  %dx%d  format=%s  idx=%d  caption=%s%n",
            img.pageNumber(),
            img.source(),
            img.width(),
            img.height(),
            img.format(),
            img.imageIndex(),
            img.caption() != null ? "\"" + img.caption() + "\"" : "null");
      }
    }
  }
}
