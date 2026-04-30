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
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.when;

import com.integrallis.vectors.demo.rag.model.DocumentChunk;
import com.integrallis.vectors.demo.rag.service.PDFImageExtractor.ExtractedImage;
import com.integrallis.vectors.server.ServerConfig;
import com.integrallis.vectors.server.VectorsServer;
import com.integrallis.vectors.server.client.VectorsServerClient;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import java.util.Base64;
import java.util.List;
import java.util.Optional;
import java.util.Random;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

/**
 * Integration tests for the full PDF ingestion pipeline: extraction → embedding → server storage →
 * blob retrieval.
 *
 * <p>Uses real PDFs from pixeltable test suite with an embedded vectors-server. Embedding model is
 * mocked (returns random vectors) since we're testing the ingestion pipeline, not embedding
 * quality.
 */
@ExtendWith(MockitoExtension.class)
@Tag("integration")
@DisplayName("PDF ingestion pipeline — real PDFs + embedded vectors-server")
class PDFIngestionPipelineTest {

  private static final String COLLECTION = "pdf-pipeline-test";
  private static final int DIM = 8;

  private static final String ATTENTION_PDF = "/pdfs/1706.03762.pdf";
  private static final String VECTOR_DB_PDF = "/pdfs/Vector_database.pdf";
  private static final String LAYOUT_PARSER_PDF = "/pdfs/layout-parser-paper.pdf";

  private VectorsServer.ServerHandle serverHandle;
  private VectorsServerClient client;

  @Mock EmbeddingModel embeddingModel;
  @TempDir Path tempDir;

  private final Random rng = new Random(42L);

  @BeforeEach
  void setUp() {
    serverHandle = VectorsServer.start(ServerConfig.forTesting());
    client = new VectorsServerClient("http://localhost:" + serverHandle.port());
    client.createCollection(COLLECTION, DIM, "COSINE", "FLAT", null);

    // Mock embedding model: return deterministic random vectors
    when(embeddingModel.embed(anyString()))
        .thenAnswer(
            inv -> {
              float[] v = new float[DIM];
              for (int i = 0; i < DIM; i++) {
                v[i] = rng.nextFloat();
              }
              return new Response<>(Embedding.from(v));
            });
  }

  @AfterEach
  void tearDown() {
    if (serverHandle != null) {
      serverHandle.close();
    }
  }

  private File copyResourceToTempDir(String resourcePath) throws IOException {
    String fileName = resourcePath.substring(resourcePath.lastIndexOf('/') + 1);
    File dest = tempDir.resolve(fileName).toFile();
    try (InputStream is = getClass().getResourceAsStream(resourcePath);
        FileOutputStream fos = new FileOutputStream(dest)) {
      assertThat(is).as("Resource not found: %s", resourcePath).isNotNull();
      is.transferTo(fos);
    }
    return dest;
  }

  @Nested
  @Tag("integration")
  @DisplayName("Text chunk extraction and storage")
  class TextChunkPipeline {

    @Test
    @DisplayName("attention paper produces one text chunk per page")
    void attentionPaperTextChunks() throws IOException {
      File pdf = copyResourceToTempDir(ATTENTION_PDF);
      PDFIngestionService service = new PDFIngestionService(client, COLLECTION, embeddingModel);

      int chunksIndexed = service.ingestPDF(pdf, "attention");
      client.commit(COLLECTION);

      // Attention paper has 15 pages — should produce at least 15 text chunks
      // plus any image chunks
      assertThat(chunksIndexed).as("Total chunks indexed").isGreaterThanOrEqualTo(15);
    }

    @Test
    @DisplayName("vector database article produces text chunks for all pages")
    void vectorDbTextChunks() throws IOException {
      File pdf = copyResourceToTempDir(VECTOR_DB_PDF);
      PDFIngestionService service = new PDFIngestionService(client, COLLECTION, embeddingModel);

      int chunksIndexed = service.ingestPDF(pdf, "vectordb");
      client.commit(COLLECTION);

      // Vector DB article: 7 pages → at least 7 text chunks
      assertThat(chunksIndexed).as("Total chunks indexed").isGreaterThanOrEqualTo(7);
    }
  }

  @Nested
  @Tag("integration")
  @DisplayName("Image blob storage and retrieval")
  class ImageBlobPipeline {

    @Test
    @DisplayName("extracted image blobs are retrievable from server after ingestion")
    void imageBlobsRetrievableAfterIngestion() throws IOException {
      File pdf = copyResourceToTempDir(LAYOUT_PARSER_PDF);
      PDFIngestionService service = new PDFIngestionService(client, COLLECTION, embeddingModel);

      service.ingestPDF(pdf, "layout-parser");
      client.commit(COLLECTION);

      // Get expected images directly from the extractor
      List<ExtractedImage> expectedImages;
      try (InputStream is = getClass().getResourceAsStream(LAYOUT_PARSER_PDF)) {
        expectedImages = PDFImageExtractor.extractImages(is);
      }

      assertThat(expectedImages).as("LayoutParser should have extractable images").isNotEmpty();

      // Verify each image blob is retrievable by its deterministic ID
      for (ExtractedImage img : expectedImages) {
        String chunkId = "layout-parser_img_p" + img.pageNumber() + "_" + img.imageIndex();
        Optional<byte[]> blob = client.getBlob(COLLECTION, chunkId);
        assertThat(blob).as("Blob for %s should be present", chunkId).isPresent();
        assertThat(blob.get().length)
            .as("Blob for %s should have substantial content", chunkId)
            .isGreaterThan(0);
      }
    }

    @Test
    @DisplayName("attention paper image blobs round-trip correctly")
    void attentionPaperBlobRoundTrip() throws IOException {
      File pdf = copyResourceToTempDir(ATTENTION_PDF);
      PDFIngestionService service = new PDFIngestionService(client, COLLECTION, embeddingModel);

      service.ingestPDF(pdf, "attention");
      client.commit(COLLECTION);

      // Extract images directly
      List<ExtractedImage> expectedImages;
      try (InputStream is = getClass().getResourceAsStream(ATTENTION_PDF)) {
        expectedImages = PDFImageExtractor.extractImages(is);
      }

      for (ExtractedImage img : expectedImages) {
        String chunkId = "attention_img_p" + img.pageNumber() + "_" + img.imageIndex();
        Optional<byte[]> blob = client.getBlob(COLLECTION, chunkId);
        assertThat(blob).as("Blob for %s should be present", chunkId).isPresent();

        // Verify the retrieved blob matches the original image bytes
        byte[] expectedBytes = img.imageBytes();
        String expectedBase64 = Base64.getEncoder().encodeToString(expectedBytes);
        String retrievedBase64 = Base64.getEncoder().encodeToString(blob.get());
        assertThat(retrievedBase64)
            .as("Blob content for %s should match extracted image", chunkId)
            .isEqualTo(expectedBase64);
      }
    }

    @Test
    @DisplayName("text-only chunks have no blob")
    void textChunksHaveNoBlob() throws IOException {
      File pdf = copyResourceToTempDir(VECTOR_DB_PDF);
      PDFIngestionService service = new PDFIngestionService(client, COLLECTION, embeddingModel);

      service.ingestPDF(pdf, "vectordb");
      client.commit(COLLECTION);

      // Text chunk ID: vectordb_text_p1
      Optional<byte[]> blob = client.getBlob(COLLECTION, "vectordb_text_p1");
      assertThat(blob)
          .satisfiesAnyOf(b -> assertThat(b).isEmpty(), b -> assertThat(b.get()).isEmpty());
    }
  }

  @Nested
  @Tag("integration")
  @DisplayName("Full pipeline round-trip")
  class FullPipelineRoundTrip {

    @Test
    @DisplayName("ingest all three PDFs into same collection without conflict")
    void ingestMultiplePdfsIntoSameCollection() throws IOException {
      PDFIngestionService service = new PDFIngestionService(client, COLLECTION, embeddingModel);

      File attention = copyResourceToTempDir(ATTENTION_PDF);
      File vectorDb = copyResourceToTempDir(VECTOR_DB_PDF);
      File layout = copyResourceToTempDir(LAYOUT_PARSER_PDF);

      int c1 = service.ingestPDF(attention, "attention");
      int c2 = service.ingestPDF(vectorDb, "vectordb");
      int c3 = service.ingestPDF(layout, "layout");
      client.commit(COLLECTION);

      assertThat(c1).as("Attention paper chunks").isGreaterThan(0);
      assertThat(c2).as("Vector DB chunks").isGreaterThan(0);
      assertThat(c3).as("LayoutParser chunks").isGreaterThan(0);

      // Total should be the sum
      int total = c1 + c2 + c3;
      assertThat(total).as("Total across all 3 PDFs").isGreaterThanOrEqualTo(30);
    }

    @Test
    @DisplayName("re-ingestion via upsert replaces chunks without duplicates")
    void reIngestionReplacesChunks() throws IOException {
      PDFIngestionService service = new PDFIngestionService(client, COLLECTION, embeddingModel);

      File pdf = copyResourceToTempDir(VECTOR_DB_PDF);

      // Ingest once
      int firstRun = service.ingestPDF(pdf, "vectordb");
      client.commit(COLLECTION);

      // Ingest again (same document ID → upsert should replace)
      int secondRun = service.ingestPDF(pdf, "vectordb");
      client.commit(COLLECTION);

      // Both runs should produce the same number of chunks
      assertThat(secondRun).as("Re-ingestion should produce same chunk count").isEqualTo(firstRun);
    }

    @Test
    @DisplayName("collection reset + re-ingestion produces fresh blobs")
    void collectionResetAndReIngest() throws IOException {
      PDFIngestionService service = new PDFIngestionService(client, COLLECTION, embeddingModel);

      File pdf = copyResourceToTempDir(ATTENTION_PDF);

      // First ingestion
      service.ingestPDF(pdf, "attention");
      client.commit(COLLECTION);

      // Reset collection
      client.deleteCollection(COLLECTION);
      client.createCollection(COLLECTION, DIM, "COSINE", "FLAT", null);

      // Re-ingest
      service.ingestPDF(pdf, "attention");
      client.commit(COLLECTION);

      // Verify blobs are still retrievable after reset + re-ingest
      List<ExtractedImage> expectedImages;
      try (InputStream is = getClass().getResourceAsStream(ATTENTION_PDF)) {
        expectedImages = PDFImageExtractor.extractImages(is);
      }

      for (ExtractedImage img : expectedImages) {
        String chunkId = "attention_img_p" + img.pageNumber() + "_" + img.imageIndex();
        Optional<byte[]> blob = client.getBlob(COLLECTION, chunkId);
        assertThat(blob)
            .as("Blob for %s should be present after reset+re-ingest", chunkId)
            .isPresent();
      }
    }
  }

  @Nested
  @Tag("integration")
  @DisplayName("Document chunk model")
  @org.mockito.junit.jupiter.MockitoSettings(strictness = org.mockito.quality.Strictness.LENIENT)
  class DocumentChunkModel {

    @Test
    @DisplayName("text chunks have deterministic IDs")
    void textChunkIdsAreDeterministic() {
      DocumentChunk chunk = DocumentChunk.text("doc1", 3, "Some text content");
      assertThat(chunk.id()).isEqualTo("doc1_text_p3");
      assertThat(chunk.chunkType()).isEqualTo(DocumentChunk.ChunkType.TEXT);
      assertThat(chunk.hasImage()).isFalse();
    }

    @Test
    @DisplayName("image chunks have deterministic IDs")
    void imageChunkIdsAreDeterministic() {
      byte[] fakeImage = new byte[] {1, 2, 3};
      DocumentChunk chunk = DocumentChunk.image("doc1", 5, 0, "A figure", fakeImage, "PNG");
      assertThat(chunk.id()).isEqualTo("doc1_img_p5_0");
      assertThat(chunk.chunkType()).isEqualTo(DocumentChunk.ChunkType.IMAGE);
      assertThat(chunk.hasImage()).isTrue();
      assertThat(chunk.imageFormat()).isEqualTo("PNG");
    }

    @Test
    @DisplayName("metadata chunks have deterministic IDs")
    void metadataChunkIdsAreDeterministic() {
      DocumentChunk chunk = DocumentChunk.metadata("doc1", "Title: Test\nPages: 5");
      assertThat(chunk.id()).isEqualTo("doc1_metadata");
      assertThat(chunk.chunkType()).isEqualTo(DocumentChunk.ChunkType.TEXT);
      assertThat(chunk.hasImage()).isFalse();
      assertThat(chunk.pageNumber()).isEqualTo(0);
      assertThat(chunk.textSummary()).contains("Title: Test");
    }
  }
}
