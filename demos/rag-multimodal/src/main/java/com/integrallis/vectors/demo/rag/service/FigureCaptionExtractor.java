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

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.apache.pdfbox.Loader;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.text.PDFTextStripper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Utility class for extracting figure captions from PDF documents.
 *
 * <p>Identifies figure references like "Figure 1:", "Fig. 2:", "Chart 3:", "Table 1:", etc. and
 * extracts their captions.
 */
public class FigureCaptionExtractor {

  private static final Logger log = LoggerFactory.getLogger(FigureCaptionExtractor.class);

  // Patterns to match figure/chart/graph/exhibit/table references
  // Matches: "Figure 1:", "Fig. 1:", "Chart 3:", "Graph 2.", "Exhibit A:", "Table 1:", etc.
  private static final Pattern FIGURE_PATTERN =
      Pattern.compile(
          "(?i)(Figure|Fig\\.?|Chart|Graph|Exhibit|Table)\\s+(\\d+)\\s*[:\\.)]\\s*([^\n]+)",
          Pattern.CASE_INSENSITIVE);

  /**
   * Represents a figure caption with its number, page, and text.
   *
   * @param figureNumber Figure number (e.g., 1, 2, 3)
   * @param pageNumber Page number where caption appears
   * @param captionText Caption text
   */
  public record FigureCaption(int figureNumber, int pageNumber, String captionText) {
    public String getFullCaption() {
      return "Figure " + figureNumber + ": " + captionText;
    }
  }

  /**
   * Extracts all figure captions from a PDF file.
   *
   * @param pdfFile PDF file to process
   * @return Map of page number to list of figure captions on that page
   * @throws IOException if PDF processing fails
   */
  public static Map<Integer, List<FigureCaption>> extractCaptions(File pdfFile) throws IOException {
    try (PDDocument document = Loader.loadPDF(pdfFile)) {
      return extractCaptions(document);
    }
  }

  /**
   * Extracts all figure captions from an already-loaded PDF document. Avoids re-opening the PDF
   * when the document is already loaded (e.g., during vector region extraction).
   *
   * @param document loaded PDF document
   * @return Map of page number to list of figure captions on that page
   * @throws IOException if text extraction fails
   */
  public static Map<Integer, List<FigureCaption>> extractCaptions(PDDocument document)
      throws IOException {
    Map<Integer, List<FigureCaption>> captionsByPage = new HashMap<>();
    PDFTextStripper textStripper = new PDFTextStripper();
    int numPages = document.getNumberOfPages();

    for (int pageNum = 0; pageNum < numPages; pageNum++) {
      int pageIndex = pageNum + 1; // 1-indexed
      textStripper.setStartPage(pageIndex);
      textStripper.setEndPage(pageIndex);
      String pageText = textStripper.getText(document);

      if (pageText == null || pageText.trim().isEmpty()) {
        continue;
      }

      List<FigureCaption> pageCaptions = extractCaptionsFromText(pageText, pageIndex);
      if (!pageCaptions.isEmpty()) {
        captionsByPage.put(pageIndex, pageCaptions);
        log.info("Found {} figure caption(s) on page {}", pageCaptions.size(), pageIndex);
      }
    }

    log.info(
        "Extracted {} total figure captions from {} pages",
        captionsByPage.values().stream().mapToInt(List::size).sum(),
        captionsByPage.size());

    return captionsByPage;
  }

  /**
   * Extracts figure captions from text.
   *
   * @param text Text to search
   * @param pageNumber Page number for metadata
   * @return List of figure captions found
   */
  private static List<FigureCaption> extractCaptionsFromText(String text, int pageNumber) {
    List<FigureCaption> captions = new ArrayList<>();
    Matcher matcher = FIGURE_PATTERN.matcher(text);

    while (matcher.find()) {
      try {
        int figNum = Integer.parseInt(matcher.group(2));
        String captionText = matcher.group(3).trim();

        // Clean up caption text (remove extra whitespace, newlines)
        captionText = captionText.replaceAll("\\s+", " ");

        // Limit caption length to first sentence or reasonable length
        if (captionText.length() > 200) {
          int periodIndex = captionText.indexOf('.');
          if (periodIndex > 0 && periodIndex < 200) {
            captionText = captionText.substring(0, periodIndex + 1);
          } else {
            captionText = captionText.substring(0, 200) + "...";
          }
        }

        FigureCaption caption = new FigureCaption(figNum, pageNumber, captionText);
        captions.add(caption);

        log.debug("Found caption on page {}: Figure {} - {}", pageNumber, figNum, captionText);
      } catch (NumberFormatException e) {
        log.warn("Failed to parse figure number: {}", matcher.group(2));
      }
    }

    return captions;
  }

  /**
   * Finds the best matching caption for an image on a given page.
   *
   * @param imageIndex Index of image on the page (0-based)
   * @param pageNumber Page number where image appears
   * @param captionsByPage Map of captions by page
   * @return Best matching caption, or null if none found
   */
  public static FigureCaption findCaptionForImage(
      int imageIndex, int pageNumber, Map<Integer, List<FigureCaption>> captionsByPage) {

    // First, check the same page
    List<FigureCaption> pageCaptions = captionsByPage.get(pageNumber);
    if (pageCaptions != null && !pageCaptions.isEmpty()) {
      // Match by image index (first image -> first caption, etc.)
      if (imageIndex < pageCaptions.size()) {
        return pageCaptions.get(imageIndex);
      }
      // If more images than captions, return first caption
      return pageCaptions.get(0);
    }

    // Check previous page (figures sometimes appear before their captions)
    if (pageNumber > 1) {
      List<FigureCaption> prevPageCaptions = captionsByPage.get(pageNumber - 1);
      if (prevPageCaptions != null && !prevPageCaptions.isEmpty()) {
        return prevPageCaptions.get(prevPageCaptions.size() - 1);
      }
    }

    // Check next page (figures sometimes appear after their captions)
    List<FigureCaption> nextPageCaptions = captionsByPage.get(pageNumber + 1);
    if (nextPageCaptions != null && !nextPageCaptions.isEmpty()) {
      return nextPageCaptions.get(0);
    }

    return null;
  }
}
