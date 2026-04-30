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

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import dev.langchain4j.data.message.ImageContent;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.Objects;
import javax.imageio.ImageIO;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Uses GPT-4o Vision to detect chart and diagram regions in rendered PDF page images.
 *
 * <p>Sends each page render to the vision model with a prompt asking it to identify bounding boxes
 * of charts, graphs, and data visualizations. Returns cropped chart images that can be stored as
 * individually displayable IMAGE chunks.
 *
 * <p>Bounding boxes use the OpenAI-recommended 0-999 normalized coordinate system (top-left
 * origin), which gives consistent results regardless of image dimensions.
 */
public class VisionChartDetector {

  private static final Logger log = LoggerFactory.getLogger(VisionChartDetector.class);
  private static final ObjectMapper MAPPER = new ObjectMapper();

  /** Minimum crop size in pixels — skip tiny false-positive detections. */
  private static final int MIN_CROP_SIZE = 80;

  private static final String DETECTION_PROMPT =
      """
      Analyze this PDF page image. Identify ALL visual elements that are data \
      visualizations, including:
      - Pie charts, donut charts
      - Bar charts (vertical or horizontal), grouped bar charts, stacked bar charts
      - Line charts, area charts, stock price charts
      - Scatter plots, bubble charts
      - Any chart or graph that visualizes numerical data

      IMPORTANT: Include bar charts even when they appear in a grid layout with \
      multiple small charts side by side. Each group of bars that forms a distinct \
      chart should be identified separately. Also include charts that have data \
      labels or numbers overlaid on them.

      Do NOT include:
      - Pure text paragraphs or headings (no visual data representation)
      - Tables that contain ONLY text and numbers in rows/columns with no bars or graphics
      - Page headers, footers, logos, or decorative borders

      For each visualization found, return a JSON object with:
      - "label": A descriptive name (e.g., "Revenue pie chart", "EBIT margin bar chart")
      - "bbox": Bounding box as [x_min, y_min, x_max, y_max] in normalized 0-999 \
        coordinates (top-left origin, spanning the full image dimensions). \
        IMPORTANT: y_min must be LESS than y_max (top before bottom).

      Return ONLY valid JSON: {"charts": [...]}
      If no charts or visualizations are found, return: {"charts": []}
      """;

  private static final String CLASSIFICATION_PROMPT =
      """
      Analyze this PDF page image. Identify ALL visual elements that are data \
      visualizations, including:
      - Pie charts, donut charts
      - Bar charts (vertical or horizontal), grouped bar charts, stacked bar charts
      - Line charts, area charts, stock price charts
      - Scatter plots, bubble charts
      - Diagrams, flowcharts, or any other graphical data representation

      Do NOT include:
      - Pure text paragraphs or headings
      - Tables with only text/numbers and no graphical elements
      - Page headers, footers, logos, or decorative borders

      For each visualization found, return a JSON object with:
      - "label": A concise descriptive name (e.g., "Revenue growth bar chart", \
        "Market share pie chart", "Stock price line chart 2020-2024")

      Return ONLY valid JSON: {"charts": [{"label": "..."}]}
      If no charts or visualizations are found, return: {"charts": []}
      """;

  private final ChatModel chatModel;

  /**
   * Creates a new VisionChartDetector.
   *
   * @param chatModel LangChain4j ChatModel with vision capabilities (e.g., GPT-4o)
   */
  public VisionChartDetector(ChatModel chatModel) {
    this.chatModel = Objects.requireNonNull(chatModel, "chatModel");
  }

  /**
   * Classifies a page render by identifying charts/visualizations and returning their descriptions.
   * Does NOT attempt bounding-box detection or cropping — GPT-4o is excellent at identifying and
   * describing charts but unreliable at providing precise pixel coordinates.
   *
   * @param pageImageBytes PNG bytes of the rendered page
   * @param pageNumber 1-indexed page number
   * @return list of chart labels found on the page, empty if none detected or on error
   */
  public List<String> classifyCharts(byte[] pageImageBytes, int pageNumber) {
    try {
      String base64 = Base64.getEncoder().encodeToString(pageImageBytes);
      UserMessage message =
          UserMessage.from(
              TextContent.from(CLASSIFICATION_PROMPT), ImageContent.from(base64, "image/png"));

      ChatRequest request = ChatRequest.builder().messages(List.of(message)).build();

      log.info("Page {}: sending to vision model for chart classification...", pageNumber);
      ChatResponse response = chatModel.chat(request);
      String responseText = response.aiMessage().text();

      List<String> labels = parseClassificationResponse(responseText, pageNumber);
      log.info("Page {}: vision model identified {} chart(s)", pageNumber, labels.size());
      return labels;
    } catch (Exception e) {
      log.warn("Page {}: vision chart classification failed: {}", pageNumber, e.getMessage());
      return List.of();
    }
  }

  /**
   * Detects chart regions in a page render and returns cropped images.
   *
   * @param pageImageBytes PNG bytes of the rendered page
   * @param pageNumber 1-indexed page number
   * @return list of cropped chart images, empty if none detected or on error
   */
  public List<CroppedChart> detectAndCrop(byte[] pageImageBytes, int pageNumber) {
    try {
      // Decode page render
      BufferedImage pageImage = ImageIO.read(new ByteArrayInputStream(pageImageBytes));
      if (pageImage == null) {
        log.warn("Page {}: could not decode page render image", pageNumber);
        return List.of();
      }

      // Send to vision model
      String base64 = Base64.getEncoder().encodeToString(pageImageBytes);
      UserMessage message =
          UserMessage.from(
              TextContent.from(DETECTION_PROMPT), ImageContent.from(base64, "image/png"));

      ChatRequest request = ChatRequest.builder().messages(List.of(message)).build();

      log.info("Page {}: sending to vision model for chart detection...", pageNumber);
      ChatResponse response = chatModel.chat(request);
      String responseText = response.aiMessage().text();

      // Parse bounding boxes
      List<DetectedChart> charts =
          parseResponse(responseText, pageImage.getWidth(), pageImage.getHeight(), pageNumber);

      if (charts.isEmpty()) {
        log.info("Page {}: no charts detected by vision model", pageNumber);
        return List.of();
      }

      // Crop each detected chart
      List<CroppedChart> results = new ArrayList<>();
      for (int i = 0; i < charts.size(); i++) {
        DetectedChart chart = charts.get(i);
        CroppedChart cropped = cropChart(pageImage, chart, pageNumber, i);
        if (cropped != null) {
          results.add(cropped);
        }
      }

      log.info("Page {}: cropped {} chart(s)", pageNumber, results.size());
      return results;

    } catch (Exception e) {
      log.warn("Page {}: vision chart detection failed: {}", pageNumber, e.getMessage());
      return List.of();
    }
  }

  private CroppedChart cropChart(
      BufferedImage pageImage, DetectedChart chart, int pageNumber, int index) {
    try {
      int x = Math.max(0, chart.x());
      int y = Math.max(0, chart.y());
      int w = Math.min(pageImage.getWidth() - x, chart.width());
      int h = Math.min(pageImage.getHeight() - y, chart.height());

      if (w < MIN_CROP_SIZE || h < MIN_CROP_SIZE) {
        log.debug("Page {}: skipping tiny detection '{}' ({}x{})", pageNumber, chart.label(), w, h);
        return null;
      }

      BufferedImage cropped = pageImage.getSubimage(x, y, w, h);
      byte[] croppedBytes = toPng(cropped);

      if (croppedBytes.length == 0) {
        return null;
      }

      return new CroppedChart(croppedBytes, chart.label(), pageNumber, index, w, h);
    } catch (Exception e) {
      log.warn("Page {}: failed to crop chart '{}': {}", pageNumber, chart.label(), e.getMessage());
      return null;
    }
  }

  private List<DetectedChart> parseResponse(
      String json, int imgWidth, int imgHeight, int pageNumber) {
    List<DetectedChart> results = new ArrayList<>();
    try {
      String cleanJson = extractJson(json);
      JsonNode root = MAPPER.readTree(cleanJson);
      JsonNode charts = root.get("charts");

      if (charts == null || !charts.isArray()) {
        return results;
      }

      for (JsonNode chart : charts) {
        String label = chart.has("label") ? chart.get("label").asText() : "Chart";
        JsonNode bbox = chart.get("bbox");
        if (bbox == null || !bbox.isArray() || bbox.size() != 4) {
          continue;
        }

        // Convert normalized 0-999 coords to pixel coords
        int x0 = (int) Math.round(bbox.get(0).asDouble() * (imgWidth - 1) / 999.0);
        int y0 = (int) Math.round(bbox.get(1).asDouble() * (imgHeight - 1) / 999.0);
        int x1 = (int) Math.round(bbox.get(2).asDouble() * (imgWidth - 1) / 999.0);
        int y1 = (int) Math.round(bbox.get(3).asDouble() * (imgHeight - 1) / 999.0);

        // Fix inverted coordinates (model sometimes swaps min/max)
        int xMin = Math.min(x0, x1);
        int yMin = Math.min(y0, y1);
        int xMax = Math.max(x0, x1);
        int yMax = Math.max(y0, y1);

        int w = xMax - xMin;
        int h = yMax - yMin;

        if (w < MIN_CROP_SIZE || h < MIN_CROP_SIZE) {
          continue;
        }

        results.add(new DetectedChart(label, xMin, yMin, w, h));
      }

      log.info("Page {}: vision model detected {} chart(s)", pageNumber, results.size());
    } catch (Exception e) {
      log.warn("Page {}: failed to parse vision response: {}", pageNumber, e.getMessage());
    }
    return results;
  }

  /** Strips markdown code fences if the model wraps its JSON response. */
  static String extractJson(String text) {
    text = text.trim();
    if (text.startsWith("```json")) {
      text = text.substring(7);
    } else if (text.startsWith("```")) {
      text = text.substring(3);
    }
    if (text.endsWith("```")) {
      text = text.substring(0, text.length() - 3);
    }
    return text.trim();
  }

  private static byte[] toPng(BufferedImage image) throws IOException {
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    ImageIO.write(image, "PNG", baos);
    return baos.toByteArray();
  }

  private List<String> parseClassificationResponse(String json, int pageNumber) {
    List<String> labels = new ArrayList<>();
    try {
      String cleanJson = extractJson(json);
      JsonNode root = MAPPER.readTree(cleanJson);
      JsonNode charts = root.get("charts");

      if (charts == null || !charts.isArray()) {
        return labels;
      }

      for (JsonNode chart : charts) {
        if (chart.has("label")) {
          String label = chart.get("label").asText().trim();
          if (!label.isEmpty()) {
            labels.add(label);
          }
        }
      }
    } catch (Exception e) {
      log.warn("Page {}: failed to parse classification response: {}", pageNumber, e.getMessage());
    }
    return labels;
  }

  /** A chart region detected by the vision model, in pixel coordinates. */
  public record DetectedChart(String label, int x, int y, int width, int height) {}

  /** A cropped chart image ready for storage. */
  public record CroppedChart(
      byte[] imageBytes, String label, int pageNumber, int chartIndex, int width, int height) {}
}
