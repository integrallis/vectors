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
import com.integrallis.vectors.demo.rag.config.AppConfig;
import com.integrallis.vectors.demo.rag.model.LLMConfig;
import com.integrallis.vectors.router.Route;
import com.integrallis.vectors.router.SemanticRouter;
import com.integrallis.vectors.server.client.VectorsServerClient;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

/**
 * Factory for creating and wiring up application services.
 *
 * <p>Handles initialization of vectors-server client, embedding models, and all service
 * dependencies.
 */
public class ServiceFactory {

  private static final String DEFAULT_COLLECTION_NAME = "rag_multimodal_docs";
  private static final int VECTOR_DIM = 384; // All-MiniLM-L6-v2 dimensions

  private static final String DEFAULT_MODEL_NAME = "doclayout_yolo_docstructbench_imgsz1024.onnx";

  private VectorsServerClient client;
  private EmbeddingModel embeddingModel;
  private CostTracker costTracker;
  private String collectionName;
  private OnnxLayoutDetector onnxDetector;

  /**
   * Initializes all services.
   *
   * @param serverUrl vectors-server base URL (e.g. http://localhost:8287)
   * @throws Exception if initialization fails
   */
  public void initialize(String serverUrl) throws Exception {
    initialize(serverUrl, DEFAULT_COLLECTION_NAME);
  }

  /**
   * Initializes all services with a custom collection name.
   *
   * @param serverUrl vectors-server base URL
   * @param collectionName name of the collection to use
   * @throws Exception if initialization fails
   */
  public void initialize(String serverUrl, String collectionName) throws Exception {
    this.collectionName = collectionName;

    // Initialize vectors-server client
    client = new VectorsServerClient(serverUrl);
    System.out.println("Connected to vectors-server at: " + serverUrl);

    // Initialize embedding model (local, no API key needed)
    embeddingModel = new AllMiniLmL6V2EmbeddingModel();
    System.out.println("Embedding model initialized: all-MiniLM-L6-v2 (" + VECTOR_DIM + " dims)");

    // Initialize cost tracker
    costTracker = new JTokKitCostTracker();

    // Ensure collection exists
    if (!client.collectionExists(collectionName)) {
      client.createCollection(collectionName, VECTOR_DIM, "COSINE", "HNSW", null);
      System.out.println("Created collection: " + collectionName);
    } else {
      System.out.println("Collection already exists: " + collectionName);
    }
  }

  /**
   * Creates a RAG service with the specified LLM configuration.
   *
   * @param config LLM configuration
   * @return RAGService instance
   * @throws IllegalStateException if services not initialized
   */
  public RAGService createRAGService(LLMConfig config) {
    if (client == null || embeddingModel == null || costTracker == null) {
      throw new IllegalStateException("Services not initialized. Call initialize() first.");
    }

    // Create chat model based on provider
    ChatModel chatModel = createChatModel(config);

    return new RAGService(client, collectionName, embeddingModel, chatModel, costTracker, config);
  }

  /**
   * Creates a PDF ingestion service with GPT-4o Vision for chart detection.
   *
   * <p>If an OpenAI API key is available via AppConfig, creates a vision-capable ChatModel for
   * detecting charts/diagrams in page renders. Otherwise, falls back to page renders without
   * vision-based cropping.
   *
   * @return PDFIngestionService instance
   * @throws IllegalStateException if services not initialized
   */
  public PDFIngestionService createPDFIngestionService() {
    if (client == null || embeddingModel == null) {
      throw new IllegalStateException("Services not initialized. Call initialize() first.");
    }

    // Try to create ONNX detector and vision model for chart detection
    OnnxLayoutDetector onnx = createOnnxDetector();
    ChatModel visionModel = createVisionModel();
    return new PDFIngestionService(client, collectionName, embeddingModel, visionModel, onnx);
  }

  /**
   * Creates an ONNX layout detector if the model can be loaded. Uses AppConfig.getOnnxModelPath()
   * override if configured, otherwise downloads from HuggingFace.
   *
   * @return OnnxLayoutDetector or null if model cannot be loaded
   */
  private OnnxLayoutDetector createOnnxDetector() {
    try {
      String configPath = AppConfig.getInstance().getOnnxModelPath();
      Path modelPath;
      if (configPath != null) {
        modelPath = Path.of(configPath);
        System.out.println("Using ONNX model from config: " + modelPath);
      } else {
        modelPath = OnnxModelManager.ensureModel(DEFAULT_MODEL_NAME);
      }
      OnnxLayoutDetector detector = new OnnxLayoutDetector(modelPath);
      this.onnxDetector = detector;
      System.out.println("ONNX layout detection enabled (DocLayout-YOLO DocStructBench)");
      return detector;
    } catch (IOException | OrtException e) {
      System.out.println("ONNX layout detection disabled: " + e.getMessage());
      return null;
    }
  }

  /**
   * Creates a ChatModel for vision-based chart detection if an API key is available.
   *
   * @return ChatModel or null if API key is not configured
   */
  private ChatModel createVisionModel() {
    try {
      LLMConfig config = AppConfig.getInstance().getLLMConfig();
      if (config.apiKey() == null || config.apiKey().isEmpty()) {
        System.out.println("No API key configured — vision chart detection disabled");
        return null;
      }
      // Use the same provider/model as the main LLM for vision detection
      ChatModel model = createChatModel(config);
      System.out.println(
          "Vision chart detection enabled using " + config.provider() + "/" + config.model());
      return model;
    } catch (Exception e) {
      System.out.println("Could not create vision model: " + e.getMessage());
      return null;
    }
  }

  /**
   * Creates a semantic router with the given routes.
   *
   * @param routes list of routes to register
   * @return SemanticRouter instance
   * @throws IllegalStateException if services not initialized
   */
  public SemanticRouter createSemanticRouter(List<Route> routes) {
    if (embeddingModel == null) {
      throw new IllegalStateException("Services not initialized. Call initialize() first.");
    }
    return new SemanticRouter(text -> embeddingModel.embed(text).content().vector(), routes);
  }

  /**
   * Creates a chat language model based on configuration.
   *
   * @param config LLM configuration
   * @return ChatModel instance
   */
  private ChatModel createChatModel(LLMConfig config) {
    return switch (config.provider()) {
      case OPENAI ->
          OpenAiChatModel.builder()
              .apiKey(config.apiKey())
              .modelName(config.model())
              .temperature(config.temperature())
              .maxTokens(config.maxTokens())
              .build();

      case ANTHROPIC, AZURE, OLLAMA ->
          throw new UnsupportedOperationException(
              config.provider()
                  + " not supported in this demo. "
                  + "Add the corresponding langchain4j provider dependency to build.gradle.kts.");
    };
  }

  /**
   * Gets the vectors-server client.
   *
   * @return VectorsServerClient instance
   */
  public VectorsServerClient getClient() {
    return client;
  }

  /**
   * Gets the embedding model.
   *
   * @return EmbeddingModel instance
   */
  public EmbeddingModel getEmbeddingModel() {
    return embeddingModel;
  }

  /**
   * Gets the cost tracker.
   *
   * @return CostTracker instance
   */
  public CostTracker getCostTracker() {
    return costTracker;
  }

  /**
   * Gets the collection name.
   *
   * @return collection name
   */
  public String getCollectionName() {
    return collectionName;
  }

  /**
   * Resets the collection by deleting and recreating it. This provides a clean slate when uploading
   * a new PDF — both the vector data and the H2 text index (which stores blobs for images) are
   * wiped and rebuilt during ingestion.
   */
  public void resetCollection() {
    if (client == null) {
      throw new IllegalStateException("Services not initialized. Call initialize() first.");
    }
    if (client.collectionExists(collectionName)) {
      client.deleteCollection(collectionName);
    }
    client.createCollection(collectionName, VECTOR_DIM, "COSINE", "HNSW", null);
  }

  /** Closes all resources. */
  public void close() {
    if (onnxDetector != null) {
      onnxDetector.close();
    }
    if (client != null) {
      client.close();
    }
  }
}
