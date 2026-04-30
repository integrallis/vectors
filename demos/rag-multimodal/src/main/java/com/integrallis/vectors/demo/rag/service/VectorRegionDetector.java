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

import java.awt.geom.Point2D;
import java.awt.geom.Rectangle2D;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.apache.pdfbox.contentstream.PDFGraphicsStreamEngine;
import org.apache.pdfbox.cos.COSName;
import org.apache.pdfbox.pdmodel.PDPage;
import org.apache.pdfbox.pdmodel.graphics.image.PDImage;
import org.apache.pdfbox.text.PDFTextStripper;
import org.apache.pdfbox.text.TextPosition;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Detects vector-drawn graphical regions (charts, diagrams, figures) in a PDF page by walking the
 * content stream via {@link PDFGraphicsStreamEngine}.
 *
 * <p>Intercepts all path operations (moveTo, lineTo, curveTo, appendRectangle) to build bounding
 * boxes of graphical regions. PDFBox delivers all coordinates already transformed to device space.
 * Text glyph positions are tracked separately to filter out text-dominated regions.
 *
 * <p>The detection algorithm:
 *
 * <ol>
 *   <li>Accumulate bounding boxes of all stroked/filled paths
 *   <li>Cluster nearby regions with single-linkage clustering (gap threshold {@value #CLUSTER_GAP})
 *   <li>Filter out noise: too small, full-page backgrounds, thin rules, text-dominated regions
 *   <li>Pad surviving regions and return as device-space rectangles
 * </ol>
 */
public class VectorRegionDetector extends PDFGraphicsStreamEngine {

  private static final Logger log = LoggerFactory.getLogger(VectorRegionDetector.class);

  /** Minimum region size in points to be considered a chart/diagram. */
  static final float MIN_REGION_SIZE = 50f;

  /** Gap threshold in points for single-linkage clustering. */
  static final float CLUSTER_GAP = 15f;

  /** Regions spanning more than this fraction of the page in both dimensions are backgrounds. */
  static final float PAGE_SPAN_THRESHOLD = 0.85f;

  /** Regions with more than this fraction of text overlap are considered text, not graphics. */
  static final float TEXT_OVERLAP_THRESHOLD = 0.70f;

  /** Padding in points added around detected regions. */
  static final float PADDING = 10f;

  /** Width threshold for thin horizontal rule detection (relative to page width). */
  private static final float RULE_WIDTH_THRESHOLD = 0.85f;

  /** Height threshold for thin horizontal rules in points. */
  private static final float RULE_HEIGHT_THRESHOLD = 5f;

  /**
   * Regions spanning more than this fraction of the page in a single dimension are column/sidebar
   * background fills, not charts. Unlike {@link #PAGE_SPAN_THRESHOLD} which requires both
   * dimensions to be large, this catches tall sidebars and wide header/footer strips.
   */
  private static final float STRIP_SPAN_THRESHOLD = 0.90f;

  private final List<Rectangle2D.Float> drawingRegions = new ArrayList<>();
  private List<Rectangle2D.Float> textRegions = List.of();

  // Current path accumulator
  private float pathMinX = Float.MAX_VALUE;
  private float pathMinY = Float.MAX_VALUE;
  private float pathMaxX = -Float.MAX_VALUE;
  private float pathMaxY = -Float.MAX_VALUE;
  private boolean pathActive = false;

  /**
   * Creates a new detector for the given page.
   *
   * @param page the PDF page to analyze
   */
  public VectorRegionDetector(PDPage page) {
    super(page);
  }

  /**
   * Detects graphical regions on the page by processing its content stream and clustering drawing
   * operations.
   *
   * @return list of device-space rectangles representing detected chart/diagram regions
   * @throws IOException if content stream processing fails
   */
  public List<Rectangle2D.Float> detectRegions() throws IOException {
    processPage(getPage());

    // Extract text positions for text-overlap filtering
    textRegions = extractTextRegions(getPage());

    if (drawingRegions.isEmpty()) {
      return List.of();
    }

    // Cluster nearby drawing regions
    List<Rectangle2D.Float> clustered = clusterRegions(drawingRegions, CLUSTER_GAP);

    // Get page dimensions for filtering
    float pageWidth = getPage().getMediaBox().getWidth();
    float pageHeight = getPage().getMediaBox().getHeight();

    // Filter out noise
    List<Rectangle2D.Float> filtered = new ArrayList<>();
    for (Rectangle2D.Float region : clustered) {
      if (region.width < MIN_REGION_SIZE || region.height < MIN_REGION_SIZE) {
        continue; // Too small
      }

      float widthFraction = region.width / pageWidth;
      float heightFraction = region.height / pageHeight;

      // Full-page background
      if (widthFraction > PAGE_SPAN_THRESHOLD && heightFraction > PAGE_SPAN_THRESHOLD) {
        log.debug(
            "Filtered full-page background: {:.0f}x{:.0f} ({:.0%}x{:.0%})",
            region.width, region.height, widthFraction, heightFraction);
        continue;
      }

      // Column/sidebar/strip background: spans >90% of one dimension
      if (widthFraction > STRIP_SPAN_THRESHOLD || heightFraction > STRIP_SPAN_THRESHOLD) {
        log.debug(
            "Filtered strip/sidebar background: {:.0f}x{:.0f} ({:.0%}x{:.0%})",
            region.width, region.height, widthFraction, heightFraction);
        continue;
      }

      // Thin horizontal rule
      if (widthFraction > RULE_WIDTH_THRESHOLD && region.height < RULE_HEIGHT_THRESHOLD) {
        continue;
      }

      // Text-dominated region check
      if (isTextDominated(region)) {
        continue;
      }

      // Pad and clamp to page bounds
      float padX = Math.max(0, region.x - PADDING);
      float padY = Math.max(0, region.y - PADDING);
      float padW = Math.min(pageWidth - padX, region.width + 2 * PADDING);
      float padH = Math.min(pageHeight - padY, region.height + 2 * PADDING);

      filtered.add(new Rectangle2D.Float(padX, padY, padW, padH));
    }

    log.debug(
        "Detected {} vector regions (from {} raw, {} clustered)",
        filtered.size(),
        drawingRegions.size(),
        clustered.size());

    return filtered;
  }

  // --- Path operations: accumulate current path bounds ---

  @Override
  public void moveTo(float x, float y) throws IOException {
    extendPath(x, y);
  }

  @Override
  public void lineTo(float x, float y) throws IOException {
    extendPath(x, y);
  }

  @Override
  public void curveTo(float x1, float y1, float x2, float y2, float x3, float y3)
      throws IOException {
    extendPath(x1, y1);
    extendPath(x2, y2);
    extendPath(x3, y3);
  }

  @Override
  public void appendRectangle(Point2D p0, Point2D p1, Point2D p2, Point2D p3) throws IOException {
    extendPath((float) p0.getX(), (float) p0.getY());
    extendPath((float) p1.getX(), (float) p1.getY());
    extendPath((float) p2.getX(), (float) p2.getY());
    extendPath((float) p3.getX(), (float) p3.getY());
  }

  // --- Path commit operations ---

  @Override
  public void strokePath() throws IOException {
    commitPath();
  }

  @Override
  public void fillPath(int windingRule) throws IOException {
    commitPath();
  }

  @Override
  public void fillAndStrokePath(int windingRule) throws IOException {
    commitPath();
  }

  @Override
  public void endPath() throws IOException {
    // Discard uncommitted path (invisible)
    resetPath();
  }

  // --- No-op overrides required by contract ---

  @Override
  public void drawImage(PDImage pdImage) throws IOException {
    // Handled by existing Tier 1 (ImageCollector)
  }

  @Override
  public void clip(int windingRule) throws IOException {}

  @Override
  public Point2D getCurrentPoint() throws IOException {
    return new Point2D.Float(0, 0);
  }

  @Override
  public void closePath() throws IOException {}

  @Override
  public void shadingFill(COSName shadingName) throws IOException {}

  // --- Internal helpers ---

  /**
   * Extracts text glyph bounding boxes from a single page using PDFTextStripper. The stripper
   * processes the page's content stream and collects TextPosition objects.
   */
  private static List<Rectangle2D.Float> extractTextRegions(PDPage page) throws IOException {
    List<Rectangle2D.Float> regions = new ArrayList<>();
    // Create a temporary single-page document for the stripper
    // PDFTextStripper operates on PDDocument, so we wrap the page
    try (org.apache.pdfbox.pdmodel.PDDocument tempDoc =
        new org.apache.pdfbox.pdmodel.PDDocument()) {
      tempDoc.addPage(page);

      TextPositionCollector collector = new TextPositionCollector();
      collector.setStartPage(1);
      collector.setEndPage(1);
      collector.getText(tempDoc);
      regions.addAll(collector.textRects);

      // Remove the page before closing to avoid double-close
      tempDoc.removePage(0);
    }
    return regions;
  }

  /** PDFTextStripper subclass that collects text position rectangles. */
  private static final class TextPositionCollector extends PDFTextStripper {
    final List<Rectangle2D.Float> textRects = new ArrayList<>();

    @Override
    protected void writeString(String text, List<TextPosition> textPositions) throws IOException {
      for (TextPosition tp : textPositions) {
        textRects.add(
            new Rectangle2D.Float(tp.getXDirAdj(), tp.getYDirAdj(), tp.getWidth(), tp.getHeight()));
      }
      super.writeString(text, textPositions);
    }
  }

  private void extendPath(float x, float y) {
    pathMinX = Math.min(pathMinX, x);
    pathMinY = Math.min(pathMinY, y);
    pathMaxX = Math.max(pathMaxX, x);
    pathMaxY = Math.max(pathMaxY, y);
    pathActive = true;
  }

  private void commitPath() {
    if (pathActive && pathMaxX > pathMinX && pathMaxY > pathMinY) {
      drawingRegions.add(
          new Rectangle2D.Float(pathMinX, pathMinY, pathMaxX - pathMinX, pathMaxY - pathMinY));
    }
    resetPath();
  }

  private void resetPath() {
    pathMinX = Float.MAX_VALUE;
    pathMinY = Float.MAX_VALUE;
    pathMaxX = -Float.MAX_VALUE;
    pathMaxY = -Float.MAX_VALUE;
    pathActive = false;
  }

  /**
   * Checks whether a region is dominated by text glyphs. A region is text-dominated if the total
   * area of text glyphs overlapping it exceeds {@value #TEXT_OVERLAP_THRESHOLD} of the region area.
   */
  private boolean isTextDominated(Rectangle2D.Float region) {
    double regionArea = region.width * (double) region.height;
    if (regionArea <= 0) {
      return false;
    }

    double textOverlapArea = 0;
    for (Rectangle2D.Float textRect : textRegions) {
      Rectangle2D intersection = region.createIntersection(textRect);
      if (intersection.getWidth() > 0 && intersection.getHeight() > 0) {
        textOverlapArea += intersection.getWidth() * intersection.getHeight();
      }
    }

    return (textOverlapArea / regionArea) > TEXT_OVERLAP_THRESHOLD;
  }

  /**
   * Single-linkage clustering: merge rectangles whose edges are within {@code gap} points of each
   * other.
   */
  static List<Rectangle2D.Float> clusterRegions(List<Rectangle2D.Float> regions, float gap) {
    if (regions.isEmpty()) {
      return List.of();
    }

    // Work on mutable copies
    List<Rectangle2D.Float> clusters = new ArrayList<>();
    for (Rectangle2D.Float r : regions) {
      clusters.add(new Rectangle2D.Float(r.x, r.y, r.width, r.height));
    }

    boolean merged = true;
    while (merged) {
      merged = false;
      for (int i = 0; i < clusters.size(); i++) {
        for (int j = i + 1; j < clusters.size(); j++) {
          if (isNear(clusters.get(i), clusters.get(j), gap)) {
            // Merge j into i
            Rectangle2D.Float a = clusters.get(i);
            Rectangle2D.Float b = clusters.remove(j);
            float minX = Math.min(a.x, b.x);
            float minY = Math.min(a.y, b.y);
            float maxX = Math.max(a.x + a.width, b.x + b.width);
            float maxY = Math.max(a.y + a.height, b.y + b.height);
            a.setRect(minX, minY, maxX - minX, maxY - minY);
            merged = true;
            break;
          }
        }
        if (merged) {
          break;
        }
      }
    }

    return clusters;
  }

  /** Two rectangles are "near" if the gap between their closest edges is within threshold. */
  private static boolean isNear(Rectangle2D.Float a, Rectangle2D.Float b, float gap) {
    float dx = Math.max(0, Math.max(a.x - (b.x + b.width), b.x - (a.x + a.width)));
    float dy = Math.max(0, Math.max(a.y - (b.y + b.height), b.y - (a.y + a.height)));
    return dx <= gap && dy <= gap;
  }
}
