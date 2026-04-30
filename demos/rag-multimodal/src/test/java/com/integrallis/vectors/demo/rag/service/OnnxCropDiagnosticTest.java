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

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import javax.imageio.ImageIO;
import org.apache.pdfbox.Loader;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.rendering.ImageType;
import org.apache.pdfbox.rendering.PDFRenderer;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

/**
 * Diagnostic test that saves annotated page images and individual crops from ONNX detection. Output
 * goes to build/onnx-diag/ for visual inspection.
 */
@Tag("slow")
class OnnxCropDiagnosticTest {

  private static final Path MODEL_PATH =
      Path.of(
          System.getProperty("user.home"),
          ".java-vectors",
          "models",
          "doclayout_yolo_docstructbench_imgsz1024.onnx");
  private static final Path PDF_PATH =
      Path.of(System.getProperty("user.home"), "Downloads", "Jefferson-Amazon.pdf");
  private static final Path OUTPUT_DIR = Path.of("build", "onnx-diag");

  static boolean canRun() {
    return Files.exists(MODEL_PATH) && Files.exists(PDF_PATH);
  }

  /**
   * For each page: saves an annotated image with bounding boxes drawn, plus individual crops. This
   * lets us visually verify that detections are correct and well-bounded.
   */
  @Test
  @EnabledIf("canRun")
  void saveAnnotatedPagesAndCrops() throws Exception {
    Files.createDirectories(OUTPUT_DIR);

    try (OnnxLayoutDetector detector = new OnnxLayoutDetector(MODEL_PATH);
        PDDocument doc = Loader.loadPDF(PDF_PATH.toFile())) {

      PDFRenderer renderer = new PDFRenderer(doc);
      int numPages = doc.getNumberOfPages();
      int totalCrops = 0;

      System.out.println("=== ONNX CROP DIAGNOSTIC: " + numPages + " pages ===");
      System.out.println("Output: " + OUTPUT_DIR.toAbsolutePath() + "\n");

      for (int page = 0; page < numPages; page++) {
        BufferedImage img = renderer.renderImageWithDPI(page, 150, ImageType.RGB);
        List<OnnxLayoutDetector.Detection> dets = detector.detect(img); // default 0.25 threshold

        System.out.printf(
            "Page %d: %d detections (%dx%d)%n",
            page + 1, dets.size(), img.getWidth(), img.getHeight());

        // Save annotated page with bounding boxes
        BufferedImage annotated =
            new BufferedImage(img.getWidth(), img.getHeight(), BufferedImage.TYPE_INT_RGB);
        Graphics2D g = annotated.createGraphics();
        g.drawImage(img, 0, 0, null);
        g.setStroke(new BasicStroke(3));
        g.setFont(new Font("SansSerif", Font.BOLD, 16));

        for (int i = 0; i < dets.size(); i++) {
          OnnxLayoutDetector.Detection d = dets.get(i);
          float areaPercent = 100f * d.width() * d.height() / (img.getWidth() * img.getHeight());

          // Color code: Figure=blue, Table=green
          g.setColor(d.classId() == OnnxLayoutDetector.CLASS_FIGURE ? Color.BLUE : Color.GREEN);
          g.drawRect(d.x(), d.y(), d.width(), d.height());

          String label =
              String.format("%s %.0f%% (%.1f%%)", d.label(), d.confidence() * 100, areaPercent);
          g.drawString(label, d.x() + 4, d.y() + 18);

          // Save individual crop
          int cx = Math.max(0, d.x());
          int cy = Math.max(0, d.y());
          int cw = Math.min(d.width(), img.getWidth() - cx);
          int ch = Math.min(d.height(), img.getHeight() - cy);
          if (cw > 0 && ch > 0) {
            BufferedImage crop = img.getSubimage(cx, cy, cw, ch);
            File cropFile =
                OUTPUT_DIR
                    .resolve(
                        String.format(
                            "p%02d_crop%d_%s_%.0f.png",
                            page + 1, i, d.label(), d.confidence() * 100))
                    .toFile();
            ImageIO.write(crop, "PNG", cropFile);
            totalCrops++;

            System.out.printf(
                "  [%d] %s conf=%.2f  bbox=[%d,%d %dx%d]  area=%.1f%%  → %s%n",
                i,
                d.label(),
                d.confidence(),
                d.x(),
                d.y(),
                d.width(),
                d.height(),
                areaPercent,
                cropFile.getName());
          }
        }

        g.dispose();
        File annotatedFile =
            OUTPUT_DIR.resolve(String.format("p%02d_annotated.png", page + 1)).toFile();
        ImageIO.write(annotated, "PNG", annotatedFile);
      }

      System.out.printf(
          "%n=== Total: %d crops saved to %s ===%n", totalCrops, OUTPUT_DIR.toAbsolutePath());
    }
  }
}
