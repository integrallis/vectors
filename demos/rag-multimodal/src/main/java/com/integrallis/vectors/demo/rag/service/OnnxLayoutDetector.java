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

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Detects document layout elements (figures, tables) using DocLayout-YOLO trained on DocStructBench
 * via ONNX Runtime.
 *
 * <p>The model was trained on DocStructBench (diverse document types including financial reports)
 * and returns pixel-level bounding boxes for 10 element classes. This detector filters for Figure
 * (class 3) and Table (class 5) by default.
 *
 * <p>Preprocessing is ported line-for-line from the reference {@code inference.py} at {@code
 * wybxc/DocLayout-YOLO-DocStructBench-onnx}.
 *
 * <p>Model specs:
 *
 * <ul>
 *   <li>Input: {@code "images"} tensor, shape (1, 3, H, W), float32, values [0, 1], BGR order
 *   <li>Output: {@code "output0"} tensor, shape (1, N, 6) — [x1, y1, x2, y2, score, classId]
 *   <li>No NMS needed (YOLOv10 uses consistent dual assignments)
 *   <li>Input size: 1024x1024 with stride-32-aligned letterbox padding (pad value 114)
 * </ul>
 */
public class OnnxLayoutDetector implements AutoCloseable {

  private static final Logger log = LoggerFactory.getLogger(OnnxLayoutDetector.class);

  /** Model input size (square). */
  static final int MODEL_SIZE = 1024;

  /** Stride for padding alignment. */
  static final int STRIDE = 32;

  /** Letterbox pad value (gray, matching OpenCV default). */
  static final int PAD_VALUE = 114;

  /** Default confidence threshold from reference inference.py. */
  private static final float DEFAULT_CONFIDENCE = 0.25f;

  /** DocStructBench class: Figure. */
  static final int CLASS_FIGURE = 3;

  /** DocStructBench class: Table. */
  static final int CLASS_TABLE = 5;

  /** DocStructBench class labels (10 classes). */
  static final String[] CLASS_LABELS = {
    "Title", // 0
    "Plain Text", // 1
    "Abandoned Text", // 2
    "Figure", // 3
    "Figure Caption", // 4
    "Table", // 5
    "Table Caption", // 6
    "Table Footnote", // 7
    "Isolated Formula", // 8
    "Formula Caption" // 9
  };

  /**
   * A detected document layout element.
   *
   * @param x left edge in original image pixels
   * @param y top edge in original image pixels
   * @param width detection width in pixels
   * @param height detection height in pixels
   * @param label human-readable class label (e.g. "Figure", "Table")
   * @param confidence model confidence score (0-1)
   * @param classId DocStructBench class ID (0-9)
   */
  public record Detection(
      int x, int y, int width, int height, String label, float confidence, int classId) {}

  /** Preprocessing result holding the tensor data and actual tensor dimensions. */
  record PreprocessResult(float[] data, int tensorH, int tensorW) {}

  private final OrtEnvironment env;
  private final OrtSession session;

  /**
   * Creates a new detector by loading the ONNX model from the given path.
   *
   * @param modelPath path to the DocLayout-YOLO DocStructBench ONNX model file
   * @throws OrtException if the model cannot be loaded
   */
  public OnnxLayoutDetector(Path modelPath) throws OrtException {
    this.env = OrtEnvironment.getEnvironment();
    this.session = env.createSession(modelPath.toString());
    log.info("ONNX layout detector loaded from {}", modelPath);
  }

  /**
   * Detects document layout elements (Figures and Tables) in a page image.
   *
   * @param pageImage rendered PDF page as a BufferedImage
   * @return list of detections with coordinates in the original image's pixel space
   * @throws OrtException if inference fails
   */
  public List<Detection> detect(BufferedImage pageImage) throws OrtException {
    return detect(pageImage, DEFAULT_CONFIDENCE);
  }

  /**
   * Detects document layout elements with a custom confidence threshold.
   *
   * @param pageImage rendered PDF page as a BufferedImage
   * @param confidenceThreshold minimum confidence to keep a detection (0-1)
   * @return list of detections with coordinates in the original image's pixel space
   * @throws OrtException if inference fails
   */
  public List<Detection> detect(BufferedImage pageImage, float confidenceThreshold)
      throws OrtException {
    int origW = pageImage.getWidth();
    int origH = pageImage.getHeight();

    // Preprocess: letterbox resize + BGR + normalize + HWC→CHW
    PreprocessResult pp = preprocess(pageImage);

    // Run inference
    long[] inputShape = {1, 3, pp.tensorH(), pp.tensorW()};
    try (OnnxTensor inputTensor =
        OnnxTensor.createTensor(env, FloatBuffer.wrap(pp.data()), inputShape)) {
      var results = session.run(Map.of("images", inputTensor));
      float[][][] output = (float[][][]) results.get(0).getValue();

      // Postprocess: filter and reverse letterbox transform
      return postprocess(output[0], pp.tensorH(), pp.tensorW(), origW, origH, confidenceThreshold);
    }
  }

  /**
   * Preprocesses a BufferedImage for DocLayout-YOLO input.
   *
   * <p>Pipeline ported from reference inference.py:
   *
   * <ol>
   *   <li>Compute scale: {@code r = min(1024/origH, 1024/origW)}
   *   <li>Resize to (resizedW, resizedH) with bilinear interpolation
   *   <li>Compute stride-aligned padding: {@code padW = (MODEL_SIZE - resizedW) % STRIDE}
   *   <li>Center padding with value 114 (gray)
   *   <li>Convert to float CHW array in <b>BGR order</b>, normalize /255
   * </ol>
   *
   * @param image input image
   * @return PreprocessResult with float data and actual tensor dimensions
   */
  static PreprocessResult preprocess(BufferedImage image) {
    int origW = image.getWidth();
    int origH = image.getHeight();

    // Compute scale factor (longest edge maps to MODEL_SIZE)
    float r = Math.min((float) MODEL_SIZE / origH, (float) MODEL_SIZE / origW);

    // Compute resized dimensions
    int resizedW = Math.round(origW * r);
    int resizedH = Math.round(origH * r);

    // Compute stride-aligned padding
    int padW = (MODEL_SIZE - resizedW) % STRIDE;
    int padH = (MODEL_SIZE - resizedH) % STRIDE;

    // Ensure non-negative padding
    if (padW < 0) padW += STRIDE;
    if (padH < 0) padH += STRIDE;

    // Center padding
    int top = padH / 2;
    int bottom = padH - top;
    int left = padW / 2;
    int right = padW - left;

    // Final tensor dimensions
    int tensorW = resizedW + padW;
    int tensorH = resizedH + padH;

    // Create canvas filled with pad value (114 gray)
    BufferedImage canvas = new BufferedImage(tensorW, tensorH, BufferedImage.TYPE_3BYTE_BGR);
    Graphics2D g = canvas.createGraphics();
    g.setColor(new java.awt.Color(PAD_VALUE, PAD_VALUE, PAD_VALUE));
    g.fillRect(0, 0, tensorW, tensorH);
    // Draw resized image at offset (left, top)
    g.setRenderingHint(
        RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
    g.drawImage(image, left, top, left + resizedW, top + resizedH, 0, 0, origW, origH, null);
    g.dispose();

    // Convert to CHW float array in BGR order, normalize [0,255] → [0,1]
    float[] data = new float[3 * tensorH * tensorW];
    int planeSize = tensorH * tensorW;

    for (int y = 0; y < tensorH; y++) {
      for (int x = 0; x < tensorW; x++) {
        int rgb = canvas.getRGB(x, y);
        int r2 = (rgb >> 16) & 0xFF;
        int g2 = (rgb >> 8) & 0xFF;
        int b = rgb & 0xFF;
        int idx = y * tensorW + x;
        // BGR order: channel 0 = B, channel 1 = G, channel 2 = R
        data[idx] = b / 255.0f;
        data[planeSize + idx] = g2 / 255.0f;
        data[2 * planeSize + idx] = r2 / 255.0f;
      }
    }

    return new PreprocessResult(data, tensorH, tensorW);
  }

  /**
   * Scales detection boxes from tensor space back to original image space. Ported from reference
   * {@code scale_boxes()} in inference.py:
   *
   * <pre>
   * gain = min(tensorH / origH, tensorW / origW)
   * padX = round((tensorW - origW * gain) / 2 - 0.1)
   * padY = round((tensorH - origH * gain) / 2 - 0.1)
   * box = (box - [padX, padY, padX, padY]) / gain
   * </pre>
   *
   * @param det raw detection [x1, y1, x2, y2, ...]
   * @param tensorH actual tensor height
   * @param tensorW actual tensor width
   * @param origW original image width
   * @param origH original image height
   * @return scaled [x1, y1, x2, y2] in original image pixels
   */
  static float[] scaleBoxes(float[] det, int tensorH, int tensorW, int origW, int origH) {
    float gain = Math.min((float) tensorH / origH, (float) tensorW / origW);
    int padX = Math.round((tensorW - origW * gain) / 2.0f - 0.1f);
    int padY = Math.round((tensorH - origH * gain) / 2.0f - 0.1f);

    float x1 = (det[0] - padX) / gain;
    float y1 = (det[1] - padY) / gain;
    float x2 = (det[2] - padX) / gain;
    float y2 = (det[3] - padY) / gain;

    // Clamp to image bounds
    x1 = Math.max(0, Math.min(x1, origW));
    y1 = Math.max(0, Math.min(y1, origH));
    x2 = Math.max(0, Math.min(x2, origW));
    y2 = Math.max(0, Math.min(y2, origH));

    return new float[] {x1, y1, x2, y2};
  }

  private List<Detection> postprocess(
      float[][] detections,
      int tensorH,
      int tensorW,
      int origW,
      int origH,
      float confidenceThreshold) {
    List<Detection> results = new ArrayList<>();

    for (float[] det : detections) {
      // det = [x1, y1, x2, y2, score, classId]
      if (det.length < 6) continue;

      float score = det[4];
      int classId = Math.round(det[5]);

      // Filter by confidence
      if (score < confidenceThreshold) continue;

      // Filter by class: Figure (3) or Table (5)
      if (classId != CLASS_FIGURE && classId != CLASS_TABLE) continue;

      // Reverse letterbox transform
      float[] scaled = scaleBoxes(det, tensorH, tensorW, origW, origH);

      int ix1 = Math.round(scaled[0]);
      int iy1 = Math.round(scaled[1]);
      int ix2 = Math.round(scaled[2]);
      int iy2 = Math.round(scaled[3]);

      int w = ix2 - ix1;
      int h = iy2 - iy1;
      if (w <= 0 || h <= 0) continue;

      String label =
          (classId >= 0 && classId < CLASS_LABELS.length) ? CLASS_LABELS[classId] : "Unknown";

      results.add(new Detection(ix1, iy1, w, h, label, score, classId));
    }

    // Greedy IoU-based overlap suppression: when two detections overlap significantly
    // (IoU > 0.5), keep only the higher-confidence one. This handles both containment
    // overlaps (aggregate box containing individual charts) and near-duplicate detections.
    List<Detection> suppressed = suppressOverlaps(results);

    log.info(
        "Postprocessed {} raw detections → {} after class/conf filter → {} after IoU suppression",
        detections.length,
        results.size(),
        suppressed.size());
    return suppressed;
  }

  /** IoU threshold above which the lower-confidence detection is suppressed. */
  static final float IOU_THRESHOLD = 0.5f;

  /**
   * Containment threshold: if the intersection covers this fraction of the smaller detection's
   * area, it is considered redundant regardless of IoU. This catches cases where a small chart
   * detection is fully inside a much larger aggregate box (IoU is low because the union is
   * dominated by the large box, but the small detection is 100% contained).
   */
  static final float CONTAINMENT_THRESHOLD = 0.8f;

  /**
   * Greedy NMS with containment check: sort by confidence (descending), keep a detection only if it
   * is not overlapping or contained by an already-kept detection. Two overlap criteria:
   *
   * <ul>
   *   <li><b>IoU &gt; 0.5</b> — near-duplicates (same object detected twice with slight bbox shift)
   *   <li><b>IoMin &gt; 0.8</b> — containment (small specific chart inside a large aggregate box)
   * </ul>
   *
   * @param detections candidate detections, already filtered by class and confidence
   * @return detections with overlaps suppressed
   */
  static List<Detection> suppressOverlaps(List<Detection> detections) {
    if (detections.size() <= 1) {
      return detections;
    }

    // Sort by confidence descending (higher confidence kept first)
    List<Detection> sorted = new ArrayList<>(detections);
    sorted.sort((a, b) -> Float.compare(b.confidence(), a.confidence()));

    List<Detection> kept = new ArrayList<>();
    for (Detection candidate : sorted) {
      boolean suppressed = false;
      for (Detection already : kept) {
        if (computeIoU(candidate, already) > IOU_THRESHOLD
            || computeContainment(candidate, already) > CONTAINMENT_THRESHOLD) {
          suppressed = true;
          break;
        }
      }
      if (!suppressed) {
        kept.add(candidate);
      }
    }
    return kept;
  }

  /**
   * Computes Intersection-over-Union between two detections.
   *
   * @return IoU value in [0, 1]
   */
  static float computeIoU(Detection a, Detection b) {
    int ax2 = a.x() + a.width();
    int ay2 = a.y() + a.height();
    int bx2 = b.x() + b.width();
    int by2 = b.y() + b.height();

    int interX1 = Math.max(a.x(), b.x());
    int interY1 = Math.max(a.y(), b.y());
    int interX2 = Math.min(ax2, bx2);
    int interY2 = Math.min(ay2, by2);

    int interW = Math.max(0, interX2 - interX1);
    int interH = Math.max(0, interY2 - interY1);
    long interArea = (long) interW * interH;

    long areaA = (long) a.width() * a.height();
    long areaB = (long) b.width() * b.height();
    long unionArea = areaA + areaB - interArea;

    if (unionArea <= 0) return 0.0f;
    return (float) interArea / unionArea;
  }

  /**
   * Computes the containment ratio: intersection area divided by the smaller detection's area. A
   * value of 1.0 means the smaller detection is fully contained within the larger one.
   *
   * @return containment ratio in [0, 1]
   */
  static float computeContainment(Detection a, Detection b) {
    int ax2 = a.x() + a.width();
    int ay2 = a.y() + a.height();
    int bx2 = b.x() + b.width();
    int by2 = b.y() + b.height();

    int interX1 = Math.max(a.x(), b.x());
    int interY1 = Math.max(a.y(), b.y());
    int interX2 = Math.min(ax2, bx2);
    int interY2 = Math.min(ay2, by2);

    int interW = Math.max(0, interX2 - interX1);
    int interH = Math.max(0, interY2 - interY1);
    long interArea = (long) interW * interH;

    long areaA = (long) a.width() * a.height();
    long areaB = (long) b.width() * b.height();
    long minArea = Math.min(areaA, areaB);

    if (minArea <= 0) return 0.0f;
    return (float) interArea / minArea;
  }

  @Override
  public void close() {
    try {
      session.close();
    } catch (OrtException e) {
      log.warn("Error closing ONNX session: {}", e.getMessage());
    }
  }
}
