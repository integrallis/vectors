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

import ai.onnxruntime.OrtException;
import com.integrallis.vectors.demo.rag.model.DocumentChunk;
import com.integrallis.vectors.server.client.DocumentPayload;
import com.integrallis.vectors.server.client.VectorsServerClient;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Base64;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.imageio.ImageIO;
import org.apache.pdfbox.Loader;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDDocumentInformation;
import org.apache.pdfbox.rendering.ImageType;
import org.apache.pdfbox.rendering.PDFRenderer;
import org.apache.pdfbox.text.PDFTextStripper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Service for ingesting and processing multimodal PDF documents.
 *
 * <p>Extracts text and images from PDFs and stores them via the vectors-server client:
 *
 * <ul>
 *   <li>Extracts document metadata (title, author, page count) as a searchable chunk
 *   <li>Extracts text and generates summaries for embedding/search
 *   <li>Extracts images via a tiered strategy: embedded rasters, then vector region detection via
 *       content stream analysis, then GPT-4o Vision page-level classification on page renders
 *   <li>Upserts all documents with embeddings, text, metadata, and optional blobs
 * </ul>
 */
public class PDFIngestionService {

  private static final Logger log = LoggerFactory.getLogger(PDFIngestionService.class);

  private final VectorsServerClient client;
  private final String collectionName;
  private final EmbeddingModel embeddingModel;
  private final VisionChartDetector visionDetector;
  private final OnnxLayoutDetector onnxDetector;

  /**
   * Creates a new PDFIngestionService with ONNX layout detection and vision-based chart detection.
   *
   * @param client vectors-server client
   * @param collectionName collection to upsert into
   * @param embeddingModel model for generating embeddings
   * @param visionModel chat model with vision capabilities (e.g., GPT-4o) for detecting charts in
   *     page renders, or null to skip vision detection
   * @param onnxDetector ONNX layout detector for cropping figures/tables from page renders, or null
   *     to skip ONNX detection
   */
  public PDFIngestionService(
      VectorsServerClient client,
      String collectionName,
      EmbeddingModel embeddingModel,
      ChatModel visionModel,
      OnnxLayoutDetector onnxDetector) {
    this.client = client;
    this.collectionName = collectionName;
    this.embeddingModel = embeddingModel;
    this.visionDetector = visionModel != null ? new VisionChartDetector(visionModel) : null;
    this.onnxDetector = onnxDetector;
  }

  /**
   * Creates a new PDFIngestionService with vision detection only (no ONNX).
   *
   * @param client vectors-server client
   * @param collectionName collection to upsert into
   * @param embeddingModel model for generating embeddings
   * @param visionModel chat model with vision capabilities, or null to skip
   */
  public PDFIngestionService(
      VectorsServerClient client,
      String collectionName,
      EmbeddingModel embeddingModel,
      ChatModel visionModel) {
    this(client, collectionName, embeddingModel, visionModel, null);
  }

  /**
   * Creates a new PDFIngestionService without vision or ONNX detection (backward-compatible).
   *
   * @param client vectors-server client
   * @param collectionName collection to upsert into
   * @param embeddingModel model for generating embeddings
   */
  public PDFIngestionService(
      VectorsServerClient client, String collectionName, EmbeddingModel embeddingModel) {
    this(client, collectionName, embeddingModel, null, null);
  }

  /**
   * Ingests a PDF file and indexes its content.
   *
   * @param pdfFile PDF file to ingest
   * @param documentId Unique document identifier
   * @return Number of chunks processed
   * @throws IOException if PDF processing fails
   * @throws IllegalArgumentException if pdfFile is null
   */
  public int ingestPDF(File pdfFile, String documentId) throws IOException {
    if (pdfFile == null) {
      throw new IllegalArgumentException("PDF file cannot be null");
    }
    if (!pdfFile.exists()) {
      throw new IOException("PDF file does not exist: " + pdfFile.getAbsolutePath());
    }

    log.info("Ingesting PDF: {} with ID: {}", pdfFile.getName(), documentId);

    List<DocumentChunk> chunks = extractChunks(pdfFile, documentId);
    int count = indexChunks(chunks);

    // Commit so blobs are immediately retrievable by subsequent queries
    client.commit(collectionName);
    log.info("Committed collection {}", collectionName);

    return count;
  }

  /**
   * Extracts chunks from PDF.
   *
   * <p>When an ONNX layout detector is available, it renders pages directly via PDFRenderer and
   * runs ONNX detection to crop individual figures/tables. This bypasses {@link PDFImageExtractor}
   * entirely, avoiding the problem where its Tier 2 vector-region detector produces garbage crops
   * that prevent the ONNX path from ever being reached.
   *
   * <p>When no ONNX detector is available, falls back to the PDFImageExtractor pipeline with
   * optional vision-based chart detection.
   *
   * @param pdfFile PDF file
   * @param documentId Document ID
   * @return List of document chunks
   * @throws IOException if extraction fails
   */
  private List<DocumentChunk> extractChunks(File pdfFile, String documentId) throws IOException {
    List<DocumentChunk> chunks = new ArrayList<>();

    try (PDDocument document = Loader.loadPDF(pdfFile)) {
      // Document metadata chunk
      String metadataText = extractMetadata(document, pdfFile.getName());
      if (metadataText != null && !metadataText.isBlank()) {
        chunks.add(DocumentChunk.metadata(documentId, metadataText));
      }

      // Text chunks per page
      PDFTextStripper textStripper = new PDFTextStripper();
      int numPages = document.getNumberOfPages();
      log.info("Processing {} pages", numPages);

      for (int pageNum = 0; pageNum < numPages; pageNum++) {
        textStripper.setStartPage(pageNum + 1);
        textStripper.setEndPage(pageNum + 1);
        String pageText = textStripper.getText(document);

        if (pageText != null && !pageText.trim().isEmpty()) {
          chunks.add(DocumentChunk.text(documentId, pageNum + 1, pageText.trim()));
        }
      }

      // Visual content: ONNX is the PRIMARY strategy when available.
      // It renders pages directly via PDFRenderer, bypassing PDFImageExtractor.
      if (onnxDetector != null) {
        List<DocumentChunk> onnxChunks = detectWithOnnx(document, documentId, pdfFile.getName());
        chunks.addAll(onnxChunks);
        log.info(
            "ONNX detection produced {} visual chunks across {} pages",
            onnxChunks.size(),
            numPages);
      } else {
        // No ONNX — fall back to PDFImageExtractor pipeline
        extractVisualContentFallback(pdfFile, documentId, chunks);
      }
    }

    log.info("Extracted {} total chunks from PDF", chunks.size());
    return chunks;
  }

  /**
   * Renders all pages with PDFRenderer and runs ONNX layout detection to find and crop individual
   * figures and tables. Pages with detections produce individual crop chunks; pages without
   * detections produce PAGE_RENDER chunks (visual context for LLM, not user-displayable).
   *
   * @param document loaded PDF document (must remain open)
   * @param documentId document identifier
   * @param fileName file name for summaries
   * @return list of visual content chunks
   */
  private List<DocumentChunk> detectWithOnnx(
      PDDocument document, String documentId, String fileName) {
    PDFRenderer renderer = new PDFRenderer(document);
    int numPages = document.getNumberOfPages();
    List<DocumentChunk> results = new ArrayList<>();
    int globalCropIndex = 0;

    for (int page = 0; page < numPages; page++) {
      BufferedImage pageImage;
      try {
        pageImage = renderer.renderImageWithDPI(page, 150, ImageType.RGB);
      } catch (IOException e) {
        log.warn("Failed to render page {} for ONNX detection: {}", page + 1, e.getMessage());
        continue;
      }

      List<OnnxLayoutDetector.Detection> detections;
      try {
        detections = onnxDetector.detect(pageImage);
      } catch (OrtException e) {
        log.warn("ONNX detection failed on page {}: {}", page + 1, e.getMessage());
        continue;
      }

      if (detections.isEmpty()) {
        // No detections — store full page render as PAGE_RENDER (visual context for LLM)
        byte[] pageBytes = encodeImage(pageImage);
        if (pageBytes != null) {
          String summary =
              String.format(
                  "Page %d of %s (full page, %dx%d PNG)",
                  page + 1, fileName, pageImage.getWidth(), pageImage.getHeight());
          results.add(DocumentChunk.pageRender(documentId, page + 1, summary, pageBytes, "PNG"));
        }
      } else {
        // Crop each detected figure/table
        for (OnnxLayoutDetector.Detection det : detections) {
          int cx = Math.max(0, det.x());
          int cy = Math.max(0, det.y());
          int cw = Math.min(det.width(), pageImage.getWidth() - cx);
          int ch = Math.min(det.height(), pageImage.getHeight() - cy);
          if (cw <= 0 || ch <= 0) continue;

          BufferedImage cropped = pageImage.getSubimage(cx, cy, cw, ch);
          byte[] croppedBytes = encodeImage(cropped);
          if (croppedBytes == null) continue;

          String summary =
              String.format(
                  "%s from page %d of %s (%dx%d PNG)", det.label(), page + 1, fileName, cw, ch);
          results.add(
              DocumentChunk.image(
                  documentId, page + 1, globalCropIndex, summary, croppedBytes, "PNG"));
          globalCropIndex++;
          log.debug(
              "ONNX crop: page {} {} conf={} {}x{}",
              page + 1,
              det.label(),
              det.confidence(),
              cw,
              ch);
        }
      }
    }

    return results;
  }

  /** Encodes a BufferedImage to PNG bytes, returning null on failure. */
  private static byte[] encodeImage(BufferedImage image) {
    try {
      ByteArrayOutputStream baos = new ByteArrayOutputStream();
      ImageIO.write(image, "PNG", baos);
      return baos.toByteArray();
    } catch (IOException e) {
      return null;
    }
  }

  /**
   * Fallback visual content extraction when ONNX is not available. Uses PDFImageExtractor pipeline
   * (embedded rasters → vector regions → page renders) with optional vision-based chart detection.
   */
  private void extractVisualContentFallback(
      File pdfFile, String documentId, List<DocumentChunk> chunks) throws IOException {
    List<PDFImageExtractor.ExtractedImage> extractedImages;
    try (FileInputStream fis = new FileInputStream(pdfFile)) {
      extractedImages = PDFImageExtractor.extractImages(fis);
    }

    log.info("PDFImageExtractor returned {} images", extractedImages.size());

    // If all results are PAGE_RENDER, try vision detection
    if (allPageRenders(extractedImages) && visionDetector != null) {
      log.info("Using GPT-4o Vision to detect individual charts");
      List<DocumentChunk> visionChunks =
          detectChartsWithVision(extractedImages, documentId, pdfFile.getName());
      if (!visionChunks.isEmpty()) {
        chunks.addAll(visionChunks);
        log.info(
            "Vision detection found {} individual chart(s) across {} page(s)",
            visionChunks.size(),
            extractedImages.size());
        return;
      }
      log.info("Vision detected no charts — falling back to page renders");
    }

    // Use extracted images directly (embedded images or page renders)
    addImageChunksFromExtracted(chunks, extractedImages, documentId, pdfFile.getName());
  }

  /** Returns true if every extracted image is a PAGE_RENDER (no informative embedded images). */
  private static boolean allPageRenders(List<PDFImageExtractor.ExtractedImage> images) {
    if (images.isEmpty()) {
      return false;
    }
    for (PDFImageExtractor.ExtractedImage img : images) {
      if (img.source() != PDFImageExtractor.Source.PAGE_RENDER) {
        return false;
      }
    }
    return true;
  }

  /**
   * Uses GPT-4o Vision to classify pages and describe charts found on them. Stores the full page
   * render as a displayable IMAGE chunk with the vision model's chart descriptions as the text
   * summary. Pages with no detected charts are stored as PAGE_RENDER (visual context only).
   */
  private List<DocumentChunk> detectChartsWithVision(
      List<PDFImageExtractor.ExtractedImage> pageRenders, String documentId, String fileName) {
    List<DocumentChunk> results = new ArrayList<>();
    int globalChartIndex = 0;

    for (PDFImageExtractor.ExtractedImage pageRender : pageRenders) {
      List<String> chartLabels =
          visionDetector.classifyCharts(pageRender.imageBytes(), pageRender.pageNumber());

      if (chartLabels.isEmpty()) {
        String summary =
            String.format(
                "Page %d of %s (full page, %dx%d %s)",
                pageRender.pageNumber(),
                fileName,
                pageRender.width(),
                pageRender.height(),
                pageRender.format());
        results.add(
            DocumentChunk.pageRender(
                documentId,
                pageRender.pageNumber(),
                summary,
                pageRender.imageBytes(),
                pageRender.format()));
      } else {
        String chartDescriptions = String.join("; ", chartLabels);
        String summary =
            String.format(
                "%s — page %d of %s (%dx%d %s)",
                chartDescriptions,
                pageRender.pageNumber(),
                fileName,
                pageRender.width(),
                pageRender.height(),
                pageRender.format());
        results.add(
            DocumentChunk.image(
                documentId,
                pageRender.pageNumber(),
                globalChartIndex,
                summary,
                pageRender.imageBytes(),
                pageRender.format()));
        globalChartIndex++;
      }
    }

    return results;
  }

  /** Adds image chunks from extracted images to the chunk list. */
  private static void addImageChunksFromExtracted(
      List<DocumentChunk> chunks,
      List<PDFImageExtractor.ExtractedImage> images,
      String documentId,
      String fileName) {
    for (PDFImageExtractor.ExtractedImage image : images) {
      String imageSummary = buildImageSummary(image, fileName);
      chunks.add(
          DocumentChunk.image(
              documentId,
              image.pageNumber(),
              image.imageIndex(),
              imageSummary,
              image.imageBytes(),
              image.format()));
    }
    log.info("Added {} images from PDF", images.size());
  }

  /**
   * Builds an image summary string, prepending the caption if available.
   *
   * @param image extracted image
   * @param fileName PDF file name
   * @return formatted summary
   */
  private static String buildImageSummary(PDFImageExtractor.ExtractedImage image, String fileName) {
    String baseSummary =
        String.format(
            "Image %d from page %d of %s (%dx%d %s)",
            image.imageIndex() + 1,
            image.pageNumber(),
            fileName,
            image.width(),
            image.height(),
            image.format());

    if (image.caption() != null && !image.caption().isBlank()) {
      return image.caption() + " -- " + baseSummary;
    }
    return baseSummary;
  }

  /**
   * Extracts document-level metadata (title, author, subject, keywords, page count).
   *
   * @param document loaded PDF document
   * @param fileName file name for display
   * @return formatted metadata text, or null if no meaningful metadata found
   */
  private static String extractMetadata(PDDocument document, String fileName) {
    PDDocumentInformation info = document.getDocumentInformation();
    int pageCount = document.getNumberOfPages();

    StringBuilder sb = new StringBuilder();
    sb.append("Document: ").append(fileName).append("\n");
    sb.append("Pages: ").append(pageCount).append("\n");

    if (info != null) {
      String title = info.getTitle();
      if (title != null && !title.isBlank()) {
        sb.append("Title: ").append(title.trim()).append("\n");
      }
      String author = info.getAuthor();
      if (author != null && !author.isBlank()) {
        sb.append("Author: ").append(author.trim()).append("\n");
      }
      String subject = info.getSubject();
      if (subject != null && !subject.isBlank()) {
        sb.append("Subject: ").append(subject.trim()).append("\n");
      }
      String keywords = info.getKeywords();
      if (keywords != null && !keywords.isBlank()) {
        sb.append("Keywords: ").append(keywords.trim()).append("\n");
      }
    }

    return sb.toString().trim();
  }

  /**
   * Indexes document chunks by upserting them to the vectors-server.
   *
   * @param chunks Document chunks
   * @return Number of chunks indexed
   */
  public int indexChunks(List<DocumentChunk> chunks) {
    List<DocumentPayload> payloads = new ArrayList<>();

    for (DocumentChunk chunk : chunks) {
      try {
        // Generate embedding for text summary
        Embedding embedding = embeddingModel.embed(chunk.textSummary()).content();

        String chunkId = chunk.id();

        // Build metadata
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("chunk_id", chunkId);
        metadata.put("document_id", chunk.documentId());
        metadata.put("page", chunk.pageNumber());
        metadata.put("type", chunk.chunkType().name());
        if (chunk.imageFormat() != null) {
          metadata.put("image_format", chunk.imageFormat());
        }

        // Build blob (Base64 image) if present
        String blob = null;
        if (chunk.hasImage()) {
          blob = Base64.getEncoder().encodeToString(chunk.imageData());
          log.info(
              "Chunk {} has blob: {} bytes raw, {} chars base64",
              chunkId,
              chunk.imageData().length,
              blob.length());
        }

        payloads.add(
            new DocumentPayload(chunkId, embedding.vector(), chunk.textSummary(), metadata, blob));
      } catch (Exception e) {
        log.error("Failed to prepare chunk: {}", e.getMessage(), e);
      }
    }

    if (payloads.isEmpty()) {
      return 0;
    }

    // Split into text-only (no blob) and image (with blob) payloads.
    // Sending all documents in one request can exceed the server's content-length
    // limit due to large Base64 blobs, causing the text-index dual-write to fail.
    List<DocumentPayload> textPayloads = new ArrayList<>();
    List<DocumentPayload> imagePayloads = new ArrayList<>();
    for (DocumentPayload p : payloads) {
      if (p.blob() != null) {
        imagePayloads.add(p);
      } else {
        textPayloads.add(p);
      }
    }

    int total = 0;

    // Batch upsert text chunks (small, no blobs)
    if (!textPayloads.isEmpty()) {
      total += client.upsertDocuments(collectionName, textPayloads);
      log.info("Upserted {} text documents", textPayloads.size());
    }

    // Upsert each image chunk individually so blobs are reliably stored
    for (DocumentPayload img : imagePayloads) {
      try {
        client.upsertDocuments(collectionName, List.of(img));
        total++;
        log.info("Upserted image document: {}", img.id());
      } catch (Exception e) {
        log.error("Failed to upsert image {}: {}", img.id(), e.getMessage());
      }
    }

    log.info("Upserted {} total documents to collection {}", total, collectionName);
    return total;
  }
}
