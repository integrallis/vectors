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

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.assertj.core.data.Offset;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

/** Tests for OnnxLayoutDetector preprocessing, coordinate transforms, and scale_boxes. */
class OnnxLayoutDetectorTest {

  @Nested
  @Tag("unit")
  class PreprocessingTests {

    @Test
    void preprocessOutputHasCorrectSize() {
      BufferedImage testImage = new BufferedImage(800, 600, BufferedImage.TYPE_3BYTE_BGR);
      OnnxLayoutDetector.PreprocessResult result = OnnxLayoutDetector.preprocess(testImage);

      // Should be 3 channels * tensorH * tensorW
      assertThat(result.data()).hasSize(3 * result.tensorH() * result.tensorW());
    }

    @Test
    void preprocessTensorDimensionsAreStrideAligned() {
      // 612x792 typical PDF page
      BufferedImage testImage = new BufferedImage(612, 792, BufferedImage.TYPE_3BYTE_BGR);
      OnnxLayoutDetector.PreprocessResult result = OnnxLayoutDetector.preprocess(testImage);

      // Both dimensions must be divisible by STRIDE (32)
      assertThat(result.tensorH() % OnnxLayoutDetector.STRIDE).isEqualTo(0);
      assertThat(result.tensorW() % OnnxLayoutDetector.STRIDE).isEqualTo(0);
    }

    @Test
    void preprocessTensorDimensionsDoNotExceedModelSize() {
      BufferedImage testImage = new BufferedImage(1280, 720, BufferedImage.TYPE_3BYTE_BGR);
      OnnxLayoutDetector.PreprocessResult result = OnnxLayoutDetector.preprocess(testImage);

      assertThat(result.tensorH()).isLessThanOrEqualTo(OnnxLayoutDetector.MODEL_SIZE);
      assertThat(result.tensorW()).isLessThanOrEqualTo(OnnxLayoutDetector.MODEL_SIZE);
    }

    @Test
    void preprocessSquareImageProducesModelSizeTensor() {
      BufferedImage testImage = new BufferedImage(800, 800, BufferedImage.TYPE_3BYTE_BGR);
      OnnxLayoutDetector.PreprocessResult result = OnnxLayoutDetector.preprocess(testImage);

      // Square image scaled to 1024x1024 — no padding needed since 1024 % 32 == 0
      assertThat(result.tensorH()).isEqualTo(OnnxLayoutDetector.MODEL_SIZE);
      assertThat(result.tensorW()).isEqualTo(OnnxLayoutDetector.MODEL_SIZE);
    }

    @Test
    void preprocessValuesAreNormalized() {
      // Create a white image
      BufferedImage testImage = new BufferedImage(100, 100, BufferedImage.TYPE_3BYTE_BGR);
      Graphics2D g = testImage.createGraphics();
      g.setColor(Color.WHITE);
      g.fillRect(0, 0, 100, 100);
      g.dispose();

      OnnxLayoutDetector.PreprocessResult result = OnnxLayoutDetector.preprocess(testImage);

      // Check that values are in [0, 1] range
      for (float v : result.data()) {
        assertThat(v).isBetween(0.0f, 1.0f);
      }

      // The white region should have values close to 1.0
      float max = 0;
      for (float v : result.data()) {
        if (v > max) max = v;
      }
      assertThat(max).isCloseTo(1.0f, Offset.offset(0.01f));
    }

    @Test
    void preprocessBlackImageProducesPadValueForPadArea() {
      // Use a non-square image (612x792 typical PDF page) so the letterbox produces pad columns.
      // A square image would scale to fill the entire 1024x1024 tensor with no padding.
      BufferedImage testImage = new BufferedImage(612, 792, BufferedImage.TYPE_3BYTE_BGR);
      // Default is already black
      OnnxLayoutDetector.PreprocessResult result = OnnxLayoutDetector.preprocess(testImage);

      // The image area should have values close to 0.0 (black)
      // The pad area should have values close to 114/255 ≈ 0.447
      float padNormalized = OnnxLayoutDetector.PAD_VALUE / 255.0f;
      boolean hasBlack = false;
      boolean hasPad = false;
      for (float v : result.data()) {
        if (v < 0.01f) hasBlack = true;
        if (Math.abs(v - padNormalized) < 0.02f) hasPad = true;
      }
      // Black image pixels exist
      assertThat(hasBlack).isTrue();
      // Pad area pixels exist (non-square image produces padding columns)
      assertThat(hasPad).isTrue();
    }

    @Test
    void preprocessUsesBgrChannelOrder() {
      // Create a pure red image (R=255, G=0, B=0)
      BufferedImage redImage = new BufferedImage(32, 32, BufferedImage.TYPE_3BYTE_BGR);
      Graphics2D g = redImage.createGraphics();
      g.setColor(Color.RED);
      g.fillRect(0, 0, 32, 32);
      g.dispose();

      OnnxLayoutDetector.PreprocessResult result = OnnxLayoutDetector.preprocess(redImage);
      int planeSize = result.tensorH() * result.tensorW();

      // In BGR order: channel 0 = B (should be ~0), channel 2 = R (should be ~1.0)
      // Sample a pixel in the image area (not pad)
      // The image is 32x32 and gets scaled; find center pixel
      float r = Math.min(1024.0f / 32, 1024.0f / 32);
      int resizedW = Math.round(32 * r);
      int padW = (1024 - resizedW) % 32;
      int left = padW / 2;
      // Sample at offset (left + resizedW/2, top + resizedH/2)
      int padH = (1024 - Math.round(32 * r)) % 32;
      int top = padH / 2;
      int centerX = left + resizedW / 2;
      int centerY = top + Math.round(32 * r) / 2;
      int idx = centerY * result.tensorW() + centerX;

      // Channel 0 (B) should be ~0 for a red pixel
      assertThat(result.data()[idx]).isCloseTo(0.0f, Offset.offset(0.05f));
      // Channel 2 (R) should be ~1.0 for a red pixel
      assertThat(result.data()[2 * planeSize + idx]).isCloseTo(1.0f, Offset.offset(0.05f));
    }
  }

  @Nested
  @Tag("unit")
  class ScaleBoxesTests {

    @Test
    void scaleBoxesReversesLetterboxForPortraitPage() {
      // Original: 612x792 (portrait PDF page)
      int origW = 612;
      int origH = 792;

      // Compute what preprocess would produce
      float r = Math.min(1024.0f / origH, 1024.0f / origW);
      int resizedW = Math.round(origW * r);
      int resizedH = Math.round(origH * r);
      int padW = (1024 - resizedW) % 32;
      int padH = (1024 - resizedH) % 32;
      int tensorW = resizedW + padW;
      int tensorH = resizedH + padH;

      // A detection in the center of the tensor
      float[] det = {tensorW / 2f - 50, tensorH / 2f - 50, tensorW / 2f + 50, tensorH / 2f + 50};
      float[] scaled = OnnxLayoutDetector.scaleBoxes(det, tensorH, tensorW, origW, origH);

      // Scaled coords should be within original image bounds
      assertThat(scaled[0]).isGreaterThanOrEqualTo(0);
      assertThat(scaled[1]).isGreaterThanOrEqualTo(0);
      assertThat(scaled[2]).isLessThanOrEqualTo(origW);
      assertThat(scaled[3]).isLessThanOrEqualTo(origH);
    }

    @Test
    void scaleBoxesPreservesAspectRatio() {
      int origW = 1000;
      int origH = 500;
      float r = Math.min(1024.0f / origH, 1024.0f / origW);
      int resizedW = Math.round(origW * r);
      int resizedH = Math.round(origH * r);
      int padW = (1024 - resizedW) % 32;
      int padH = (1024 - resizedH) % 32;
      int tensorW = resizedW + padW;
      int tensorH = resizedH + padH;

      float gain = Math.min((float) tensorH / origH, (float) tensorW / origW);
      int padX = Math.round((tensorW - origW * gain) / 2.0f - 0.1f);
      int padY = Math.round((tensorH - origH * gain) / 2.0f - 0.1f);

      // Square detection in tensor space
      float[] det = {padX + 100, padY + 100, padX + 300, padY + 300};
      float[] scaled = OnnxLayoutDetector.scaleBoxes(det, tensorH, tensorW, origW, origH);

      float w = scaled[2] - scaled[0];
      float h = scaled[3] - scaled[1];
      // Square in tensor → square in original
      assertThat(Math.abs(w - h)).isLessThan(2.0f);
    }

    @Test
    void scaleBoxesClampsToImageBounds() {
      int origW = 800;
      int origH = 600;
      int tensorW = 1024;
      int tensorH = 1024;

      // Detection that extends outside the image region
      float[] det = {-10, -10, 1050, 1050};
      float[] scaled = OnnxLayoutDetector.scaleBoxes(det, tensorH, tensorW, origW, origH);

      assertThat(scaled[0]).isGreaterThanOrEqualTo(0);
      assertThat(scaled[1]).isGreaterThanOrEqualTo(0);
      assertThat(scaled[2]).isLessThanOrEqualTo(origW);
      assertThat(scaled[3]).isLessThanOrEqualTo(origH);
    }

    @Test
    void scaleBoxesUsesReferenceFormula() {
      // Verify the -0.1 offset in padding calculation matches reference
      int origW = 612;
      int origH = 792;
      int tensorW = 800;
      int tensorH = 1024;

      float gain = Math.min((float) tensorH / origH, (float) tensorW / origW);
      int expectedPadX = Math.round((tensorW - origW * gain) / 2.0f - 0.1f);
      int expectedPadY = Math.round((tensorH - origH * gain) / 2.0f - 0.1f);

      // A detection at (expectedPadX, expectedPadY) should map to ~(0, 0)
      float[] det = {expectedPadX, expectedPadY, expectedPadX + 100 * gain, expectedPadY};
      float[] scaled = OnnxLayoutDetector.scaleBoxes(det, tensorH, tensorW, origW, origH);

      assertThat(scaled[0]).isCloseTo(0.0f, Offset.offset(1.5f));
      assertThat(scaled[1]).isCloseTo(0.0f, Offset.offset(1.5f));
    }
  }

  @Nested
  @Tag("unit")
  class ClassLabelTests {

    @Test
    void classLabelsHaveTenEntries() {
      assertThat(OnnxLayoutDetector.CLASS_LABELS).hasSize(10);
    }

    @Test
    void figureClassIdIsCorrect() {
      assertThat(OnnxLayoutDetector.CLASS_LABELS[OnnxLayoutDetector.CLASS_FIGURE])
          .isEqualTo("Figure");
    }

    @Test
    void tableClassIdIsCorrect() {
      assertThat(OnnxLayoutDetector.CLASS_LABELS[OnnxLayoutDetector.CLASS_TABLE])
          .isEqualTo("Table");
    }
  }

  @Nested
  @Tag("unit")
  class OverlapSuppressionTests {

    private static OnnxLayoutDetector.Detection det(
        int x, int y, int w, int h, float conf, int classId) {
      String label = OnnxLayoutDetector.CLASS_LABELS[classId];
      return new OnnxLayoutDetector.Detection(x, y, w, h, label, conf, classId);
    }

    @Test
    void identicalBoxesHaveIoUOfOne() {
      var a = det(100, 100, 200, 200, 0.9f, OnnxLayoutDetector.CLASS_FIGURE);
      var b = det(100, 100, 200, 200, 0.8f, OnnxLayoutDetector.CLASS_FIGURE);
      assertThat(OnnxLayoutDetector.computeIoU(a, b)).isCloseTo(1.0f, Offset.offset(0.001f));
    }

    @Test
    void nonOverlappingBoxesHaveIoUOfZero() {
      var a = det(0, 0, 100, 100, 0.9f, OnnxLayoutDetector.CLASS_FIGURE);
      var b = det(200, 200, 100, 100, 0.8f, OnnxLayoutDetector.CLASS_FIGURE);
      assertThat(OnnxLayoutDetector.computeIoU(a, b)).isEqualTo(0.0f);
    }

    @Test
    void partialOverlapComputesCorrectly() {
      // Two 100x100 boxes overlapping by 50x100 (right half of A = left half of B)
      var a = det(0, 0, 100, 100, 0.9f, OnnxLayoutDetector.CLASS_FIGURE);
      var b = det(50, 0, 100, 100, 0.8f, OnnxLayoutDetector.CLASS_FIGURE);
      // Intersection = 50*100 = 5000, Union = 10000+10000-5000 = 15000, IoU = 1/3
      assertThat(OnnxLayoutDetector.computeIoU(a, b)).isCloseTo(1.0f / 3, Offset.offset(0.01f));
    }

    @Test
    void containedBoxHasLowIoUButHighContainment() {
      // Small box fully inside large box
      var large = det(0, 0, 400, 400, 0.7f, OnnxLayoutDetector.CLASS_FIGURE);
      var small = det(100, 100, 100, 100, 0.5f, OnnxLayoutDetector.CLASS_FIGURE);
      // IoU is low (1/16) because union is dominated by the large box
      float iou = OnnxLayoutDetector.computeIoU(large, small);
      assertThat(iou).isLessThan(OnnxLayoutDetector.IOU_THRESHOLD);
      // But containment is 1.0 — the small box is fully inside the large one
      float containment = OnnxLayoutDetector.computeContainment(large, small);
      assertThat(containment).isCloseTo(1.0f, Offset.offset(0.001f));
    }

    @Test
    void containedDetectionIsSuppressedByContainmentCheck() {
      // Page 7 scenario: large aggregate box (66% conf, 43.7% area) contains
      // a specific chart detection (36% conf, 9.7% area)
      var aggregate = det(66, 276, 1139, 806, 0.66f, OnnxLayoutDetector.CLASS_FIGURE);
      var specific = det(644, 310, 560, 364, 0.36f, OnnxLayoutDetector.CLASS_FIGURE);

      // The specific detection is fully contained in the aggregate
      float containment = OnnxLayoutDetector.computeContainment(aggregate, specific);
      assertThat(containment).isGreaterThan(OnnxLayoutDetector.CONTAINMENT_THRESHOLD);

      // Suppression should keep only the higher-confidence aggregate
      List<OnnxLayoutDetector.Detection> result =
          OnnxLayoutDetector.suppressOverlaps(List.of(specific, aggregate));
      assertThat(result).hasSize(1);
      assertThat(result.getFirst().confidence()).isEqualTo(0.66f);
    }

    @Test
    void nonContainedBoxesNotSuppressedByContainment() {
      // Two adjacent charts in a 2x2 grid — no containment
      var left = det(70, 583, 560, 365, 0.65f, OnnxLayoutDetector.CLASS_FIGURE);
      var right = det(644, 580, 560, 368, 0.68f, OnnxLayoutDetector.CLASS_FIGURE);
      float containment = OnnxLayoutDetector.computeContainment(left, right);
      assertThat(containment).isLessThan(OnnxLayoutDetector.CONTAINMENT_THRESHOLD);
    }

    @Test
    void nearDuplicateIsSuppressed() {
      // Two nearly identical detections of the same chart (page 8 scenario)
      var high = det(644, 1330, 559, 160, 0.46f, OnnxLayoutDetector.CLASS_FIGURE);
      var low = det(644, 1306, 559, 185, 0.34f, OnnxLayoutDetector.CLASS_FIGURE);
      float iou = OnnxLayoutDetector.computeIoU(high, low);
      // These overlap heavily — IoU should be above threshold
      assertThat(iou).isGreaterThan(OnnxLayoutDetector.IOU_THRESHOLD);

      // Suppression should keep only the higher-confidence one
      List<OnnxLayoutDetector.Detection> result =
          OnnxLayoutDetector.suppressOverlaps(List.of(low, high));
      assertThat(result).hasSize(1);
      assertThat(result.getFirst().confidence()).isEqualTo(0.46f);
    }

    @Test
    void nonOverlappingDetectionsAreAllKept() {
      var a = det(0, 0, 100, 100, 0.9f, OnnxLayoutDetector.CLASS_FIGURE);
      var b = det(200, 0, 100, 100, 0.8f, OnnxLayoutDetector.CLASS_TABLE);
      var c = det(400, 0, 100, 100, 0.7f, OnnxLayoutDetector.CLASS_FIGURE);
      List<OnnxLayoutDetector.Detection> result =
          OnnxLayoutDetector.suppressOverlaps(List.of(a, b, c));
      assertThat(result).hasSize(3);
    }

    @Test
    void emptyListReturnsEmpty() {
      List<OnnxLayoutDetector.Detection> result = OnnxLayoutDetector.suppressOverlaps(List.of());
      assertThat(result).isEmpty();
    }

    @Test
    void singleDetectionReturnsSame() {
      var a = det(0, 0, 100, 100, 0.9f, OnnxLayoutDetector.CLASS_FIGURE);
      List<OnnxLayoutDetector.Detection> result = OnnxLayoutDetector.suppressOverlaps(List.of(a));
      assertThat(result).containsExactly(a);
    }
  }

  @Nested
  @Tag("slow")
  class IntegrationTests {

    private static final Path MODEL_PATH =
        Path.of(
            System.getProperty("user.home"),
            ".java-vectors",
            "models",
            "doclayout_yolo_docstructbench_imgsz1024.onnx");

    static boolean modelAvailable() {
      return Files.exists(MODEL_PATH);
    }

    @Test
    @EnabledIf("modelAvailable")
    void detectOnSyntheticPageReturnsValidDetections() throws Exception {
      // Create a synthetic 612x792 "page" with a colored rectangle (simulating a chart)
      BufferedImage page = new BufferedImage(612, 792, BufferedImage.TYPE_3BYTE_BGR);
      Graphics2D g = page.createGraphics();
      g.setColor(Color.WHITE);
      g.fillRect(0, 0, 612, 792);
      // Draw a "chart" area with bars
      g.setColor(Color.BLUE);
      g.fillRect(100, 200, 50, 150);
      g.fillRect(170, 250, 50, 100);
      g.fillRect(240, 220, 50, 130);
      g.fillRect(310, 270, 50, 80);
      g.setColor(Color.BLACK);
      g.drawRect(80, 180, 300, 200);
      g.dispose();

      try (OnnxLayoutDetector detector = new OnnxLayoutDetector(MODEL_PATH)) {
        List<OnnxLayoutDetector.Detection> detections = detector.detect(page);
        // Synthetic charts may or may not be detected — just verify no crash and valid format
        for (OnnxLayoutDetector.Detection det : detections) {
          assertThat(det.x()).isGreaterThanOrEqualTo(0);
          assertThat(det.y()).isGreaterThanOrEqualTo(0);
          assertThat(det.width()).isGreaterThan(0);
          assertThat(det.height()).isGreaterThan(0);
          assertThat(det.x() + det.width()).isLessThanOrEqualTo(612);
          assertThat(det.y() + det.height()).isLessThanOrEqualTo(792);
          assertThat(det.confidence()).isBetween(0.0f, 1.0f);
          assertThat(det.classId()).isBetween(0, 9);
          assertThat(det.label()).isNotEmpty();
        }
      }
    }
  }
}
