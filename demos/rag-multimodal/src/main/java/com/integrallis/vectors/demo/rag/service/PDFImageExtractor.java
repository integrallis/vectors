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
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.imageio.ImageIO;
import org.apache.pdfbox.Loader;
import org.apache.pdfbox.contentstream.PDFGraphicsStreamEngine;
import org.apache.pdfbox.cos.COSDictionary;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDPage;
import org.apache.pdfbox.pdmodel.graphics.image.PDImage;
import org.apache.pdfbox.pdmodel.graphics.image.PDImageXObject;
import org.apache.pdfbox.rendering.ImageType;
import org.apache.pdfbox.rendering.PDFRenderer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Extracts visual content from PDF documents using a three-tier strategy:
 *
 * <ol>
 *   <li><b>Embedded raster images</b> — photos, screenshots, scanned diagrams that are stored as
 *       image XObjects in the PDF. Uses {@link PDFGraphicsStreamEngine} to walk each page's content
 *       stream so only images that are <em>actually rendered</em> on a page are attributed to it.
 *   <li><b>Vector region detection</b> — detects charts, diagrams, and figures drawn with PDF path
 *       commands via {@link VectorRegionDetector}, renders the page, and crops individual regions
 *       into separate images.
 *   <li><b>Page rendering</b> — renders pages as raster images via {@link PDFRenderer} as a final
 *       fallback when neither embedded images nor vector regions are found.
 * </ol>
 *
 * <p>The {@link #extractImages(InputStream)} method automatically selects the best strategy:
 * embedded images that carry real information content (photos, complex diagrams) are preferred;
 * then vector-drawn regions are detected and cropped; finally, full pages are rendered as fallback.
 */
public class PDFImageExtractor {

  private static final Logger log = LoggerFactory.getLogger(PDFImageExtractor.class);

  // Minimum image dimensions to filter out small icons/decorations
  private static final int DEFAULT_MIN_WIDTH = 100;
  private static final int DEFAULT_MIN_HEIGHT = 100;

  /** DPI for page rendering (200 gives good quality for cropped regions). */
  private static final float RENDER_DPI = 200f;

  /**
   * Fraction of near-white pixels above which an embedded image is considered decorative.
   * Decorative images (gradient triangles, solid fills on white backgrounds) typically have &gt;
   * 40% white pixels. This filter only applies to Tier 1 (embedded raster images); Tier 2 (vector
   * regions) relies on the {@link VectorRegionDetector}'s own heuristics instead.
   */
  private static final double DECORATIVE_WHITE_THRESHOLD = 0.40;

  /** Near-white pixel threshold: all of R, G, B must be above this value. */
  private static final int WHITE_CHANNEL_THRESHOLD = 240;

  /**
   * Extracts visual content from a PDF using the three-tier strategy.
   *
   * <p>First attempts embedded raster extraction. If all extracted images are decorative, tries
   * vector region detection. Falls back to page rendering as a last resort.
   *
   * @param pdfStream PDF input stream
   * @return List of extracted images with metadata
   * @throws IOException if PDF processing fails
   */
  public static List<ExtractedImage> extractImages(InputStream pdfStream) throws IOException {
    return extractImages(pdfStream, DEFAULT_MIN_WIDTH, DEFAULT_MIN_HEIGHT);
  }

  /**
   * Extracts visual content from a PDF with custom size filter for embedded images.
   *
   * @param pdfStream PDF input stream
   * @param minWidth Minimum image width in pixels
   * @param minHeight Minimum image height in pixels
   * @return List of extracted images with metadata
   * @throws IOException if PDF processing fails
   */
  public static List<ExtractedImage> extractImages(
      InputStream pdfStream, int minWidth, int minHeight) throws IOException {

    byte[] pdfBytes = pdfStream.readAllBytes();

    // Tier 1: Try embedded raster extraction
    List<ExtractedImage> embedded = extractEmbeddedImages(pdfBytes, minWidth, minHeight);

    // Evaluate information content: filter out decorative images
    List<ExtractedImage> informative = filterDecorative(embedded);

    List<ExtractedImage> result;

    if (!informative.isEmpty()) {
      log.info(
          "Using {} informative embedded images (filtered {} decorative)",
          informative.size(),
          embedded.size() - informative.size());
      result = informative;
    } else {
      // Tier 2: Try vector region detection (charts drawn with PDF path commands)
      log.info(
          "No informative embedded images found ({} decorative filtered). "
              + "Trying vector region detection...",
          embedded.size());

      List<ExtractedImage> vectorRegions = extractVectorRegions(pdfBytes);
      if (!vectorRegions.isEmpty()) {
        log.info("Vector region detection found {} chart region(s)", vectorRegions.size());
        result = vectorRegions;
      } else {
        // Tier 3: Render each page as a displayable image (final fallback).
        log.info("No vector regions found. Rendering pages as images.");
        result = renderPages(pdfBytes);
      }
    }

    // Apply size filter to final result (all tiers)
    if (minWidth > DEFAULT_MIN_WIDTH || minHeight > DEFAULT_MIN_HEIGHT) {
      result =
          result.stream()
              .filter(img -> img.width() >= minWidth && img.height() >= minHeight)
              .toList();
    }

    return result;
  }

  /**
   * Extracts only embedded raster images from the PDF (Tier 1).
   *
   * @param pdfBytes Raw PDF bytes
   * @param minWidth Minimum image width in pixels
   * @param minHeight Minimum image height in pixels
   * @return List of extracted embedded images
   * @throws IOException if PDF processing fails
   */
  static List<ExtractedImage> extractEmbeddedImages(byte[] pdfBytes, int minWidth, int minHeight)
      throws IOException {

    List<ExtractedImage> extractedImages = new ArrayList<>();
    Set<COSDictionary> globalSeen = Collections.newSetFromMap(new IdentityHashMap<>());

    try (PDDocument document = Loader.loadPDF(pdfBytes)) {
      int pageNumber = 1;

      for (PDPage page : document.getPages()) {
        ImageCollector collector = new ImageCollector(page, minWidth, minHeight, globalSeen);
        try {
          collector.processPage(page);
        } catch (IOException e) {
          log.warn("Error processing page {}: {}", pageNumber, e.getMessage());
        }

        int imageIndex = 0;
        for (PDImageXObject image : collector.images) {
          ExtractedImage extracted = extractImage(image, pageNumber, imageIndex);
          if (extracted != null) {
            extractedImages.add(extracted);
            imageIndex++;
          }
        }
        pageNumber++;
      }
    }

    log.info("Extracted {} embedded images from PDF", extractedImages.size());
    return extractedImages;
  }

  /**
   * Detects vector-drawn graphical regions (charts, diagrams, figures) and crops them into
   * individual images (Tier 2).
   *
   * <p>For each page, runs {@link VectorRegionDetector} to find drawing regions, renders the page
   * at {@value #RENDER_DPI} DPI, and crops each detected region. Correlates regions with figure
   * captions from {@link FigureCaptionExtractor}.
   *
   * @param pdfBytes Raw PDF bytes
   * @return List of cropped region images
   * @throws IOException if processing fails
   */
  static List<ExtractedImage> extractVectorRegions(byte[] pdfBytes) throws IOException {
    List<ExtractedImage> results = new ArrayList<>();

    try (PDDocument document = Loader.loadPDF(pdfBytes)) {
      PDFRenderer renderer = new PDFRenderer(document);
      Map<Integer, List<FigureCaptionExtractor.FigureCaption>> captionsByPage =
          FigureCaptionExtractor.extractCaptions(document);

      int numPages = document.getNumberOfPages();
      int globalImageIndex = 0;

      for (int pageIdx = 0; pageIdx < numPages; pageIdx++) {
        int pageNumber = pageIdx + 1;
        PDPage page = document.getPage(pageIdx);

        VectorRegionDetector detector = new VectorRegionDetector(page);
        List<Rectangle2D.Float> regions;
        try {
          regions = detector.detectRegions();
        } catch (IOException e) {
          log.warn("Vector region detection failed on page {}: {}", pageNumber, e.getMessage());
          continue;
        }

        if (regions.isEmpty()) {
          continue;
        }

        // Render the page once at high DPI
        BufferedImage pageImage;
        try {
          pageImage = renderer.renderImageWithDPI(pageIdx, RENDER_DPI, ImageType.RGB);
        } catch (IOException e) {
          log.warn("Failed to render page {} for cropping: {}", pageNumber, e.getMessage());
          continue;
        }

        // Scale factor: PDF points → pixels at RENDER_DPI
        float scale = RENDER_DPI / 72f;

        for (int regionIdx = 0; regionIdx < regions.size(); regionIdx++) {
          Rectangle2D.Float region = regions.get(regionIdx);

          // Convert device-space points to pixel coordinates
          int px = Math.max(0, (int) (region.x * scale));
          int py = Math.max(0, (int) (region.y * scale));
          int pw = Math.min(pageImage.getWidth() - px, (int) (region.width * scale));
          int ph = Math.min(pageImage.getHeight() - py, (int) (region.height * scale));

          if (pw <= 0 || ph <= 0) {
            continue;
          }

          BufferedImage cropped = pageImage.getSubimage(px, py, pw, ph);
          byte[] croppedBytes = convertImageToBytes(cropped, "PNG");
          if (croppedBytes.length == 0) {
            continue;
          }

          // Correlate with figure caption
          String caption = null;
          FigureCaptionExtractor.FigureCaption figCaption =
              FigureCaptionExtractor.findCaptionForImage(regionIdx, pageNumber, captionsByPage);
          if (figCaption != null) {
            caption = figCaption.getFullCaption();
          }

          results.add(
              new ExtractedImage(
                  croppedBytes,
                  "PNG",
                  pageNumber,
                  pw,
                  ph,
                  globalImageIndex,
                  Source.VECTOR_REGION,
                  caption));
          globalImageIndex++;
        }
      }
    }

    return results;
  }

  /**
   * Renders each page of the PDF as a raster image (Tier 3).
   *
   * @param pdfBytes Raw PDF bytes
   * @return List of page-rendered images
   * @throws IOException if rendering fails
   */
  static List<ExtractedImage> renderPages(byte[] pdfBytes) throws IOException {
    List<ExtractedImage> rendered = new ArrayList<>();

    try (PDDocument document = Loader.loadPDF(pdfBytes)) {
      PDFRenderer renderer = new PDFRenderer(document);
      int numPages = document.getNumberOfPages();

      for (int pageIdx = 0; pageIdx < numPages; pageIdx++) {
        int pageNumber = pageIdx + 1;
        try {
          BufferedImage pageImage = renderer.renderImageWithDPI(pageIdx, RENDER_DPI, ImageType.RGB);
          byte[] imageBytes = convertImageToBytes(pageImage, "PNG");

          if (imageBytes.length > 0) {
            rendered.add(
                new ExtractedImage(
                    imageBytes,
                    "PNG",
                    pageNumber,
                    pageImage.getWidth(),
                    pageImage.getHeight(),
                    0,
                    Source.PAGE_RENDER,
                    null));
            log.debug(
                "Rendered page {} as image: {}x{} ({} bytes)",
                pageNumber,
                pageImage.getWidth(),
                pageImage.getHeight(),
                imageBytes.length);
          }
        } catch (IOException e) {
          log.warn("Failed to render page {}: {}", pageNumber, e.getMessage());
        }
      }
    }

    log.info("Rendered {} pages as images", rendered.size());
    return rendered;
  }

  /**
   * Filters out decorative images (solid fills, gradients on white backgrounds). An image is
   * considered decorative if more than {@link #DECORATIVE_WHITE_THRESHOLD} of its pixels are
   * near-white (R, G, B all &gt; {@link #WHITE_CHANNEL_THRESHOLD}).
   *
   * @param images List of extracted images
   * @return Filtered list containing only informative images
   */
  static List<ExtractedImage> filterDecorative(List<ExtractedImage> images) {
    List<ExtractedImage> result = new ArrayList<>();
    for (ExtractedImage img : images) {
      if (isInformative(img)) {
        result.add(img);
      } else {
        log.debug(
            "Filtered decorative image on page {} ({}x{})",
            img.pageNumber(),
            img.width(),
            img.height());
      }
    }
    return result;
  }

  /**
   * Determines whether an embedded image carries meaningful information content (not a decorative
   * element like a gradient triangle on a white background).
   *
   * <p>Strategy: decode the image, sample pixels uniformly, and compute the fraction of near-white
   * pixels. Decorative shapes (triangles, arrows, gradient fills) on white backgrounds have &gt;
   * 40% white pixels. Photos, charts, and diagrams typically fill the frame with non-white content.
   *
   * <p>This check is used only for Tier 1 (embedded raster images). Tier 2 vector regions rely on
   * the {@link VectorRegionDetector}'s own spatial heuristics instead.
   */
  static boolean isInformative(ExtractedImage img) {
    try {
      BufferedImage bi = ImageIO.read(new java.io.ByteArrayInputStream(img.imageBytes()));
      if (bi == null) {
        return true; // Can't decode -> assume informative
      }

      int w = bi.getWidth();
      int h = bi.getHeight();
      int totalPixels = w * h;
      if (totalPixels == 0) {
        return false;
      }

      // Sample every Nth pixel for efficiency (sample ~2000 pixels max)
      int step = Math.max(1, (int) Math.sqrt(totalPixels / 2000.0));
      int whiteCount = 0;
      int sampleCount = 0;

      for (int y = 0; y < h; y += step) {
        for (int x = 0; x < w; x += step) {
          int rgb = bi.getRGB(x, y);
          int r = (rgb >> 16) & 0xFF;
          int g = (rgb >> 8) & 0xFF;
          int b = rgb & 0xFF;
          if (r > WHITE_CHANNEL_THRESHOLD
              && g > WHITE_CHANNEL_THRESHOLD
              && b > WHITE_CHANNEL_THRESHOLD) {
            whiteCount++;
          }
          sampleCount++;
        }
      }

      double whiteFraction = (double) whiteCount / sampleCount;
      boolean informative = whiteFraction < DECORATIVE_WHITE_THRESHOLD;

      if (!informative) {
        log.debug(
            "Image on page {} is {}{} white — classified as decorative",
            img.pageNumber(),
            String.format("%.1f", whiteFraction * 100),
            "%");
      }

      return informative;
    } catch (IOException e) {
      return true; // Decode error -> assume informative
    }
  }

  /**
   * Content stream walker that collects images as they are drawn. Only images rendered via the
   * page's content stream trigger {@link #drawImage(PDImage)}, so inherited-but-unused resources
   * are ignored.
   */
  private static final class ImageCollector extends PDFGraphicsStreamEngine {

    final List<PDImageXObject> images = new ArrayList<>();
    private final int minWidth;
    private final int minHeight;
    private final Set<COSDictionary> globalSeen;

    ImageCollector(PDPage page, int minWidth, int minHeight, Set<COSDictionary> globalSeen) {
      super(page);
      this.minWidth = minWidth;
      this.minHeight = minHeight;
      this.globalSeen = globalSeen;
    }

    @Override
    public void drawImage(PDImage pdImage) throws IOException {
      if (!(pdImage instanceof PDImageXObject image)) {
        return;
      }
      int width = image.getWidth();
      int height = image.getHeight();
      if (width < minWidth || height < minHeight) {
        log.debug("Skipping small image: {}x{}", width, height);
        return;
      }

      // Deduplicate by COSDictionary identity so the same stream isn't extracted twice
      COSDictionary cos = image.getCOSObject();
      if (!globalSeen.add(cos)) {
        log.debug("Skipping duplicate image: {}x{}", width, height);
        return;
      }

      images.add(image);
    }

    // --- Required no-op overrides (all throw IOException per contract) ---

    @Override
    public void appendRectangle(Point2D p0, Point2D p1, Point2D p2, Point2D p3)
        throws IOException {}

    @Override
    public void clip(int windingRule) throws IOException {}

    @Override
    public void moveTo(float x, float y) throws IOException {}

    @Override
    public void lineTo(float x, float y) throws IOException {}

    @Override
    public void curveTo(float x1, float y1, float x2, float y2, float x3, float y3)
        throws IOException {}

    @Override
    public Point2D getCurrentPoint() throws IOException {
      return new Point2D.Float(0, 0);
    }

    @Override
    public void closePath() throws IOException {}

    @Override
    public void endPath() throws IOException {}

    @Override
    public void strokePath() throws IOException {}

    @Override
    public void fillPath(int windingRule) throws IOException {}

    @Override
    public void fillAndStrokePath(int windingRule) throws IOException {}

    @Override
    public void shadingFill(org.apache.pdfbox.cos.COSName shadingName) throws IOException {}
  }

  /**
   * Extracts a single PDImageXObject into an ExtractedImage record.
   *
   * @return ExtractedImage or null if extraction fails
   */
  private static ExtractedImage extractImage(PDImageXObject image, int pageNumber, int imageIndex) {
    try {
      BufferedImage bufferedImage = image.getImage();
      if (bufferedImage == null) {
        log.warn("Could not convert PDImageXObject to BufferedImage on page {}", pageNumber);
        return null;
      }

      // Ensure RGB color model (CMYK/indexed images can't be encoded directly)
      if (bufferedImage.getType() != BufferedImage.TYPE_INT_RGB
          && bufferedImage.getType() != BufferedImage.TYPE_3BYTE_BGR) {
        BufferedImage rgb =
            new BufferedImage(
                bufferedImage.getWidth(), bufferedImage.getHeight(), BufferedImage.TYPE_INT_RGB);
        rgb.createGraphics().drawImage(bufferedImage, 0, 0, null);
        bufferedImage = rgb;
      }

      // Determine format and convert
      String format = determineImageFormat(image);
      byte[] imageBytes = convertImageToBytes(bufferedImage, format);

      // Fallback to PNG if the chosen format produced no output
      if (imageBytes.length == 0 && !"PNG".equals(format)) {
        format = "PNG";
        imageBytes = convertImageToBytes(bufferedImage, format);
      }

      if (imageBytes.length == 0) {
        log.warn(
            "Could not encode image on page {} ({}x{}), skipping",
            pageNumber,
            image.getWidth(),
            image.getHeight());
        return null;
      }

      log.debug(
          "Extracted image {} from page {}: {}x{} ({})",
          imageIndex + 1,
          pageNumber,
          image.getWidth(),
          image.getHeight(),
          format);

      return new ExtractedImage(
          imageBytes,
          format,
          pageNumber,
          image.getWidth(),
          image.getHeight(),
          imageIndex,
          Source.EMBEDDED,
          null);

    } catch (IOException e) {
      log.error("Error extracting image from page {}: {}", pageNumber, e.getMessage(), e);
      return null;
    }
  }

  /**
   * Determines the image format from PDImageXObject.
   *
   * @param image PDF image object
   * @return Format string ("JPEG" or "PNG")
   */
  private static String determineImageFormat(PDImageXObject image) {
    String suffix = image.getSuffix();
    if (suffix != null) {
      if (suffix.equalsIgnoreCase("jpg") || suffix.equalsIgnoreCase("jpeg")) {
        return "JPEG";
      } else if (suffix.equalsIgnoreCase("png")) {
        return "PNG";
      }
    }

    // Default to PNG for lossless conversion
    return "PNG";
  }

  /**
   * Converts BufferedImage to byte array.
   *
   * @param image Image to convert
   * @param format Image format ("JPEG" or "PNG")
   * @return Image bytes
   * @throws IOException if conversion fails
   */
  static byte[] convertImageToBytes(BufferedImage image, String format) throws IOException {
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    ImageIO.write(image, format, baos);
    return baos.toByteArray();
  }

  /** How the image was obtained from the PDF. */
  public enum Source {
    /** Embedded raster image extracted from a PDF image XObject. */
    EMBEDDED,
    /** Cropped region from a vector-drawn chart/diagram detected via content stream analysis. */
    VECTOR_REGION,
    /** Full-page render via PDFRenderer (captures vector-drawn content). */
    PAGE_RENDER,
    /** Chart/diagram cropped from a page render using GPT-4o Vision bounding-box detection. */
    VISION_CROP,
    /**
     * Chart/figure cropped using ONNX document layout detection (DocLayout-YOLO DocStructBench).
     */
    ONNX_LAYOUT
  }

  /**
   * Represents an extracted image from a PDF.
   *
   * @param imageBytes Raw image bytes
   * @param format Image format (JPEG, PNG)
   * @param pageNumber Page number (1-indexed)
   * @param width Image width in pixels
   * @param height Image height in pixels
   * @param imageIndex Index of image on the page (0-indexed)
   * @param source How the image was obtained (embedded raster, vector region crop, or page render)
   * @param caption Figure caption if found by caption extraction, or null
   */
  public record ExtractedImage(
      byte[] imageBytes,
      String format,
      int pageNumber,
      int width,
      int height,
      int imageIndex,
      Source source,
      String caption) {

    /** Backward-compatible constructor (caption defaults to null). */
    public ExtractedImage(
        byte[] imageBytes,
        String format,
        int pageNumber,
        int width,
        int height,
        int imageIndex,
        Source source) {
      this(imageBytes, format, pageNumber, width, height, imageIndex, source, null);
    }

    public ExtractedImage {
      if (imageBytes == null || imageBytes.length == 0) {
        throw new IllegalArgumentException("Image bytes cannot be null or empty");
      }
      if (format == null || format.isEmpty()) {
        throw new IllegalArgumentException("Format cannot be null or empty");
      }
      if (pageNumber < 1) {
        throw new IllegalArgumentException("Page number must be positive");
      }
      if (width < 1 || height < 1) {
        throw new IllegalArgumentException("Dimensions must be positive");
      }
      if (source == null) {
        throw new IllegalArgumentException("Source cannot be null");
      }
    }
  }
}
