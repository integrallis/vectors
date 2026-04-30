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
package com.integrallis.vectors.demo.rag;

import com.integrallis.vectors.demo.rag.model.DocumentChunk;
import com.integrallis.vectors.demo.rag.service.FigureCaptionExtractor;
import com.integrallis.vectors.demo.rag.service.PDFImageExtractor;
import com.integrallis.vectors.server.ServerConfig;
import com.integrallis.vectors.server.VectorsServer;
import com.integrallis.vectors.server.client.DocumentPayload;
import com.integrallis.vectors.server.client.SearchHit;
import com.integrallis.vectors.server.client.VectorsServerClient;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.ImageContent;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import java.io.File;
import java.io.FileInputStream;
import java.util.ArrayList;
import java.util.Base64;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Standalone CLI demo for multimodal RAG using an embedded vectors-server.
 *
 * <p>This demonstration shows:
 *
 * <ol>
 *   <li>Starting an embedded vectors-server on an ephemeral port
 *   <li>Creating a collection and ingesting a PDF with text + image extraction
 *   <li>Running hybrid search queries (vector + text, RRF fusion)
 *   <li>Multimodal RAG with GPT-4o vision for image analysis
 * </ol>
 */
public class MultimodalRAGStandalone {

  private static final int VECTOR_DIM = 384;
  private static final String COLLECTION_NAME = "research-papers";
  private static final String PDF_PATH = "src/test/resources/test-pdfs/Attention.pdf";

  public static void main(String[] args) throws Exception {
    System.out.println("=".repeat(80));
    System.out.println("Multimodal RAG with java-vectors (embedded server)");
    System.out.println("=".repeat(80));
    System.out.println();

    // Check for OpenAI API key
    String apiKey = System.getenv("OPENAI_API_KEY");
    if (apiKey == null || apiKey.isEmpty()) {
      System.err.println("ERROR: OPENAI_API_KEY environment variable not set");
      System.err.println("Set it with: export OPENAI_API_KEY=sk-...");
      System.exit(1);
    }

    // Step 1: Start embedded vectors-server
    System.out.println("Step 1: Starting embedded vectors-server");
    System.out.println("-".repeat(80));

    ServerConfig serverConfig = ServerConfig.forTesting();
    try (VectorsServer.ServerHandle serverHandle = VectorsServer.start(serverConfig)) {
      int port = serverHandle.port();
      String baseUrl = "http://localhost:" + port;
      System.out.println("Vectors-server started on port " + port);
      System.out.println();

      try (VectorsServerClient client = new VectorsServerClient(baseUrl)) {

        // Step 2: Create collection
        System.out.println("Step 2: Creating collection");
        System.out.println("-".repeat(80));

        if (!client.collectionExists(COLLECTION_NAME)) {
          client.createCollection(COLLECTION_NAME, VECTOR_DIM, "COSINE", "HNSW", null);
          System.out.println("Created collection: " + COLLECTION_NAME);
        } else {
          System.out.println("Collection already exists: " + COLLECTION_NAME);
        }
        System.out.println();

        // Step 3: Initialize embedding model
        System.out.println("Step 3: Initializing components");
        System.out.println("-".repeat(80));

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        System.out.println(
            "Embedding model: sentence-transformers/all-MiniLM-L6-v2 (" + VECTOR_DIM + " dims)");
        System.out.println();

        // Step 4: Ingest PDF
        File pdfFile = new File(PDF_PATH);
        if (pdfFile.exists()) {
          System.out.println("Step 4: Ingesting " + pdfFile.getName());
          System.out.println("-".repeat(80));

          int indexed = ingestPdf(client, embeddingModel, pdfFile, apiKey);
          System.out.println("Indexed " + indexed + " chunks total");
          System.out.println();

          // Commit to ensure all data is searchable
          client.commit(COLLECTION_NAME);

          // Step 5: Query examples
          System.out.println("Step 5: Query Examples");
          System.out.println("=".repeat(80));
          System.out.println();

          ChatModel chatModel =
              OpenAiChatModel.builder()
                  .apiKey(apiKey)
                  .modelName("gpt-4o")
                  .temperature(0.7)
                  .maxTokens(500)
                  .build();

          // 5a: Basic hybrid search
          basicHybridSearch(client, embeddingModel, "attention mechanism in neural networks");

          // 5b: Text RAG query
          textRAGQuery(client, embeddingModel, chatModel, "What is the Transformer architecture?");

          // 5c: Multimodal query
          multimodalQuery(
              client, embeddingModel, chatModel, "Describe the Multi-Head Attention diagram.");

        } else {
          System.out.println("Step 4: Skipping PDF ingestion (file not found: " + PDF_PATH + ")");
          System.out.println(
              "Place a PDF at the above path to enable ingestion, or adjust PDF_PATH.");
          System.out.println();
        }

        System.out.println("=".repeat(80));
        System.out.println("Demo completed successfully!");
        System.out.println("=".repeat(80));
      }
    }
  }

  private static int ingestPdf(
      VectorsServerClient client, EmbeddingModel embeddingModel, File pdfFile, String apiKey)
      throws Exception {
    List<DocumentChunk> chunks = new ArrayList<>();
    String documentId = "attention-paper";

    // Extract text chunks
    try (var document = org.apache.pdfbox.Loader.loadPDF(pdfFile)) {
      var textStripper = new org.apache.pdfbox.text.PDFTextStripper();
      int numPages = document.getNumberOfPages();
      System.out.println("Processing " + numPages + " pages...");

      for (int pageNum = 0; pageNum < numPages; pageNum++) {
        textStripper.setStartPage(pageNum + 1);
        textStripper.setEndPage(pageNum + 1);
        String pageText = textStripper.getText(document);
        if (pageText != null && !pageText.trim().isEmpty()) {
          chunks.add(DocumentChunk.text(documentId, pageNum + 1, pageText.trim()));
        }
      }
    }
    System.out.println("Extracted " + chunks.size() + " text chunks");

    // Extract figure captions
    Map<Integer, List<FigureCaptionExtractor.FigureCaption>> captionsByPage =
        FigureCaptionExtractor.extractCaptions(pdfFile);
    int totalCaptions = captionsByPage.values().stream().mapToInt(List::size).sum();
    System.out.println("Found " + totalCaptions + " figure caption(s)");

    // Extract images with GPT-4o descriptions
    System.out.println("Generating semantic descriptions for images using GPT-4o...");
    ChatModel visionModel =
        OpenAiChatModel.builder()
            .apiKey(apiKey)
            .modelName("gpt-4o")
            .temperature(0.3)
            .maxTokens(150)
            .build();

    try (FileInputStream fis = new FileInputStream(pdfFile)) {
      List<PDFImageExtractor.ExtractedImage> extractedImages =
          PDFImageExtractor.extractImages(fis, 200, 200);
      System.out.println("Extracted " + extractedImages.size() + " embedded images");

      for (PDFImageExtractor.ExtractedImage image : extractedImages) {
        FigureCaptionExtractor.FigureCaption caption =
            FigureCaptionExtractor.findCaptionForImage(
                image.pageNumber(), image.imageIndex(), captionsByPage);

        String base64Image = Base64.getEncoder().encodeToString(image.imageBytes());

        List<dev.langchain4j.data.message.Content> visionContents = new ArrayList<>();
        String prompt;
        if (caption != null) {
          prompt =
              "This is Figure "
                  + caption.figureNumber()
                  + " from a research paper. "
                  + "Describe this diagram or figure in 1-2 sentences. "
                  + "Focus on what it shows, the key components, and its purpose. "
                  + "Begin your description with 'Figure "
                  + caption.figureNumber()
                  + ": '";
        } else {
          prompt =
              "Describe this diagram or figure from a research paper in 1-2 sentences. "
                  + "Focus on what it shows, the key components, and its purpose. "
                  + "Be specific and technical.";
        }

        visionContents.add(TextContent.from(prompt));
        visionContents.add(ImageContent.from(base64Image, "image/png"));

        List<ChatMessage> visionMessages = new ArrayList<>();
        visionMessages.add(UserMessage.from(visionContents));

        ChatResponse descriptionResponse = visionModel.chat(visionMessages);
        String richDescription = descriptionResponse.aiMessage().text();

        if (caption != null
            && !richDescription.toLowerCase().contains("figure " + caption.figureNumber())) {
          richDescription = "Figure " + caption.figureNumber() + ": " + richDescription;
        }

        chunks.add(
            DocumentChunk.image(
                documentId,
                image.pageNumber(),
                image.imageIndex(),
                richDescription,
                image.imageBytes(),
                image.format()));
        System.out.println("  - " + truncate(richDescription, 100));
      }
    }

    // Index all chunks
    System.out.println("Indexing " + chunks.size() + " chunks...");
    List<DocumentPayload> payloads = new ArrayList<>();
    for (DocumentChunk chunk : chunks) {
      Embedding embedding = embeddingModel.embed(chunk.textSummary()).content();
      Map<String, Object> metadata = new HashMap<>();
      metadata.put("chunk_id", chunk.id());
      metadata.put("document_id", chunk.documentId());
      metadata.put("page", chunk.pageNumber());
      metadata.put("type", chunk.chunkType().name());

      String blob = null;
      if (chunk.hasImage()) {
        blob = Base64.getEncoder().encodeToString(chunk.imageData());
      }

      payloads.add(
          new DocumentPayload(chunk.id(), embedding.vector(), chunk.textSummary(), metadata, blob));
    }

    return client.upsertDocuments(COLLECTION_NAME, payloads);
  }

  private static void basicHybridSearch(
      VectorsServerClient client, EmbeddingModel embeddingModel, String queryText) {
    System.out.println("Example 5a: Hybrid Search");
    System.out.println("-".repeat(80));
    System.out.println("Query: '" + queryText + "'");
    System.out.println();

    float[] queryVector = embeddingModel.embed(queryText).content().vector();
    List<SearchHit> results =
        client.hybridSearch(COLLECTION_NAME, queryVector, queryText, 3, "RRF");

    System.out.println("Top " + results.size() + " results:");
    for (int i = 0; i < results.size(); i++) {
      SearchHit hit = results.get(i);
      String type =
          hit.metadata() != null
              ? String.valueOf(hit.metadata().getOrDefault("type", "TEXT"))
              : "TEXT";
      String page =
          hit.metadata() != null ? String.valueOf(hit.metadata().getOrDefault("page", "?")) : "?";

      System.out.println(
          "  "
              + (i + 1)
              + ". ["
              + type
              + "] Page "
              + page
              + " (score: "
              + String.format("%.4f", hit.score())
              + ")");
      System.out.println("     " + truncate(hit.text(), 100));
      System.out.println();
    }
  }

  private static void textRAGQuery(
      VectorsServerClient client,
      EmbeddingModel embeddingModel,
      ChatModel chatModel,
      String question) {
    System.out.println("Example 5b: Text RAG Query");
    System.out.println("-".repeat(80));
    System.out.println("Question: " + question);
    System.out.println();

    float[] queryVector = embeddingModel.embed(question).content().vector();
    List<SearchHit> results =
        client.hybridSearch(COLLECTION_NAME, queryVector, question, 10, "RRF");

    StringBuilder context = new StringBuilder("Context:\n");
    for (SearchHit hit : results) {
      if (hit.text() != null) {
        context.append(hit.text()).append("\n\n");
      }
    }

    List<ChatMessage> messages = new ArrayList<>();
    messages.add(SystemMessage.from("You are a helpful research paper assistant."));
    messages.add(UserMessage.from(context + "\nQuestion: " + question));

    ChatResponse response = chatModel.chat(messages);
    String answer = response.aiMessage().text();

    System.out.println("Retrieved " + results.size() + " relevant chunks");
    System.out.println();
    System.out.println("GPT-4o Response:");
    System.out.println(answer);
    System.out.println();
  }

  private static void multimodalQuery(
      VectorsServerClient client,
      EmbeddingModel embeddingModel,
      ChatModel chatModel,
      String question) {
    System.out.println("Example 5c: Multimodal Query (Text + Images)");
    System.out.println("-".repeat(80));
    System.out.println("Question: " + question);
    System.out.println();

    float[] queryVector = embeddingModel.embed(question).content().vector();
    List<SearchHit> results =
        client.hybridSearch(COLLECTION_NAME, queryVector, question, 10, "RRF");

    List<SearchHit> textHits = new ArrayList<>();
    List<SearchHit> imageHits = new ArrayList<>();

    for (SearchHit hit : results) {
      String type =
          hit.metadata() != null
              ? String.valueOf(hit.metadata().getOrDefault("type", "TEXT"))
              : "TEXT";
      if ("IMAGE".equals(type)) {
        imageHits.add(hit);
      } else {
        textHits.add(hit);
      }
    }

    System.out.println(
        "Retrieved "
            + results.size()
            + " total: "
            + textHits.size()
            + " text, "
            + imageHits.size()
            + " images");

    if (imageHits.isEmpty()) {
      System.out.println("No images found. Falling back to text-only RAG.");
      textRAGQuery(client, embeddingModel, chatModel, question);
      return;
    }

    List<dev.langchain4j.data.message.Content> messageContents = new ArrayList<>();

    if (!textHits.isEmpty()) {
      StringBuilder contextText = new StringBuilder("Text Context:\n");
      for (SearchHit hit : textHits) {
        if (hit.text() != null) {
          contextText.append(hit.text()).append("\n\n");
        }
      }
      contextText.append("Question: ").append(question);
      messageContents.add(TextContent.from(contextText.toString()));
    } else {
      messageContents.add(TextContent.from("Question: " + question));
    }

    System.out.println("Images being sent to GPT-4o:");
    for (SearchHit imgHit : imageHits) {
      System.out.println("  - " + truncate(imgHit.text(), 80));
      Optional<byte[]> blobOpt = client.getBlob(COLLECTION_NAME, imgHit.id());
      if (blobOpt.isPresent()) {
        String base64Image = Base64.getEncoder().encodeToString(blobOpt.get());
        messageContents.add(ImageContent.from(base64Image, "image/png"));
      }
    }

    List<ChatMessage> messages = new ArrayList<>();
    messages.add(
        SystemMessage.from(
            "You are a helpful assistant that analyzes research papers. "
                + "When images are provided, carefully analyze their visual content."));
    messages.add(UserMessage.from(messageContents));

    System.out.println();
    System.out.println("GPT-4o Response (with vision):");
    System.out.println("-".repeat(80));
    ChatResponse response = chatModel.chat(messages);
    System.out.println(response.aiMessage().text());
    System.out.println();
  }

  private static String truncate(String text, int maxLength) {
    if (text == null) return "";
    if (text.length() <= maxLength) return text;
    return text.substring(0, maxLength) + "...";
  }
}
