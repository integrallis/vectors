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
package com.integrallis.vectors.demo.rag.ui;

import com.integrallis.vectors.demo.rag.config.AppConfig;
import com.integrallis.vectors.demo.rag.model.CacheType;
import com.integrallis.vectors.demo.rag.model.ChatMessage;
import com.integrallis.vectors.demo.rag.model.LLMConfig;
import com.integrallis.vectors.demo.rag.model.LogEntry.Category;
import com.integrallis.vectors.demo.rag.service.EventLogger;
import com.integrallis.vectors.demo.rag.service.PDFIngestionService;
import com.integrallis.vectors.demo.rag.service.RAGService;
import com.integrallis.vectors.demo.rag.service.ServiceFactory;
import com.integrallis.vectors.router.Route;
import com.integrallis.vectors.router.RouteMatch;
import com.integrallis.vectors.router.SemanticRouter;
import java.io.File;
import java.time.Instant;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import javafx.application.Platform;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.geometry.Insets;
import javafx.geometry.Orientation;
import javafx.scene.control.Button;
import javafx.scene.control.ListCell;
import javafx.scene.control.ListView;
import javafx.scene.control.SplitPane;
import javafx.scene.control.TextField;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.stage.FileChooser;

/**
 * Main chat interface controller.
 *
 * <p>Manages the chat UI, message flow, and interactions with RAGService.
 */
public class ChatController extends BorderPane {

  private static final int SETTINGS_TAB = 0;
  private static final int ROUTER_TAB = 1;
  private static final int LOG_TAB = 2;

  private final ObservableList<ChatMessage> messages = FXCollections.observableArrayList();
  private final ListView<ChatMessage> messageListView;
  private final TextField inputField;
  private final Button sendButton;
  private final PDFViewerPanel pdfViewer;
  private final SettingsPanel settingsPanel;
  private final LogPanel logPanel;
  private final SemanticRouterPanel routerPanel;
  private final EventLogger eventLogger;

  private RAGService ragService;
  private ServiceFactory serviceFactory;
  private PDFIngestionService pdfIngestionService;
  private SemanticRouter semanticRouter;
  private final ExecutorService executor = Executors.newSingleThreadExecutor();
  private final AppConfig config = AppConfig.getInstance();

  @SuppressWarnings("this-escape")
  public ChatController() {
    // Initialize event logger first
    eventLogger = new EventLogger();
    eventLogger.info(Category.SYSTEM, "Application starting...");

    // PDF viewer panel (left side)
    pdfViewer = new PDFViewerPanel();
    pdfViewer.setPrefWidth(450);
    pdfViewer.setMinWidth(300);

    // Settings panel content
    settingsPanel = new SettingsPanel();
    settingsPanel.getUploadPdfButton().setOnAction(e -> uploadPdf());

    // Log panel content
    logPanel = new LogPanel();
    logPanel.setEventLogger(eventLogger);

    // Semantic router panel content
    routerPanel = new SemanticRouterPanel();
    routerPanel.setOnEnabledChange(this::onRouterEnabledChange);

    // Vertical tab pane (right side)
    VerticalTabPane tabPane = new VerticalTabPane();
    tabPane.addTab("\u2699", "Settings", settingsPanel); // Tab 0 - gear
    tabPane.addTab("\u2443", "Semantic Router", routerPanel); // Tab 1 - fork
    tabPane.addTab("\u2261", "Event Log", logPanel, true); // Tab 2 - triple bar (bottom)

    // Start with settings panel open (for PDF upload)
    tabPane.selectTab(SETTINGS_TAB);

    // Update log badge when count changes
    logPanel.setCountListener(
        count -> tabPane.setBadge(LOG_TAB, count > 0 ? String.valueOf(count) : null));

    // Message list (center)
    messageListView = new ListView<>(messages);
    messageListView.setCellFactory(
        lv ->
            new ListCell<>() {
              @Override
              protected void updateItem(ChatMessage item, boolean empty) {
                super.updateItem(item, empty);
                if (empty || item == null) {
                  setGraphic(null);
                } else {
                  setGraphic(new MessageBubble(item, pdfViewer::goToPage));
                }
              }
            });
    messageListView.setPadding(new Insets(10));
    messageListView.getStyleClass().add("message-list");

    // Use SplitPane for PDF viewer and chat
    SplitPane splitPane = new SplitPane();
    splitPane.setOrientation(Orientation.HORIZONTAL);
    splitPane.getItems().addAll(pdfViewer, messageListView);
    splitPane.setDividerPositions(0.4);
    SplitPane.setResizableWithParent(pdfViewer, false);

    setCenter(splitPane);

    // Bottom input area
    HBox inputSection = new HBox(10);
    inputSection.setPadding(new Insets(10));
    inputSection.getStyleClass().add("input-area");

    inputField = new TextField();
    inputField.setPromptText("Ask a question about the PDF...");
    inputField.getStyleClass().add("input-field");
    HBox.setHgrow(inputField, Priority.ALWAYS);
    inputField.setOnAction(e -> sendMessage());

    sendButton = new Button("Send");
    sendButton.setDefaultButton(true);
    sendButton.getStyleClass().add("send-button");
    sendButton.setOnAction(e -> sendMessage());

    inputSection.getChildren().addAll(inputField, sendButton);
    setBottom(inputSection);

    // Right side: vertical tab pane
    setRight(tabPane);

    // Add welcome message
    addSystemMessage(
        "Welcome to Multimodal RAG Demo!\n\nConfiguration loaded from application.properties:\n\u2022 Provider: "
            + config.getLLMProvider().getDisplayName()
            + "\n\u2022 Model: "
            + config.getLLMConfig().model()
            + "\n\nUpload a PDF to get started.");
  }

  private void sendMessage() {
    String userInput = inputField.getText().trim();
    if (userInput.isEmpty()) {
      return;
    }

    // Clear input
    inputField.clear();

    // Initialize RAG service if needed
    if (ragService == null) {
      initializeRAGService();
      if (ragService == null) {
        addSystemMessage(
            "Failed to initialize RAG service. Check configuration in application.properties");
        return;
      }
    }

    // Add user message
    ChatMessage userMessage = ChatMessage.user(userInput);
    messages.add(userMessage);
    scrollToBottom();

    // Disable input while processing
    setInputEnabled(false);
    settingsPanel.setStatus("Thinking...");

    CacheType cacheType = settingsPanel.getCacheComboBox().getValue();
    double threshold = settingsPanel.getThreshold();
    eventLogger.info(
        Category.LLM, "Query: \"" + truncate(userInput, 50) + "\" (cache: " + cacheType + ")");

    // Process in background
    long startTime = System.currentTimeMillis();
    executor.submit(
        () -> {
          try {
            // Check semantic router BEFORE RAG if enabled
            if (routerPanel.isRouterEnabled() && semanticRouter != null) {
              RouteMatch match = semanticRouter.route(userInput);
              if (match.getName() != null) {
                String routeName = match.getName();
                double distance = match.getDistance() != null ? match.getDistance() : 0.0;
                Platform.runLater(
                    () -> {
                      String blockMessage =
                          String.format(
                              "This query was blocked by the semantic router.\n\n"
                                  + "Matched route: %s\nDistance: %.4f\n\n"
                                  + "The query appears to be off-topic or outside"
                                  + " the scope of the document.",
                              routeName, distance);
                      addSystemMessage(blockMessage);
                      eventLogger.warn(
                          Category.RETRIEVAL,
                          "Query BLOCKED by router: "
                              + routeName
                              + " (distance: "
                              + String.format("%.4f", distance)
                              + ")");
                      setInputEnabled(true);
                      settingsPanel.setStatus("Ready");
                      inputField.requestFocus();
                    });
                return; // Don't proceed to RAG
              }
              eventLogger.debug(Category.RETRIEVAL, "Router check passed - no route match");
            }

            ragService.setSimilarityThreshold(threshold);
            ChatMessage response = ragService.query(userInput, cacheType);
            long elapsed = System.currentTimeMillis() - startTime;
            ChatMessage timedResponse = response.withElapsedMs(elapsed);

            Platform.runLater(
                () -> {
                  messages.add(timedResponse);
                  scrollToBottom();

                  if (timedResponse.fromCache()) {
                    eventLogger.info(Category.CACHE, "Cache HIT - Response in " + elapsed + "ms");
                  } else {
                    eventLogger.info(
                        Category.LLM,
                        "LLM response in "
                            + elapsed
                            + "ms ("
                            + timedResponse.tokenCount()
                            + " tokens)");
                  }

                  if (!timedResponse.references().isEmpty()) {
                    eventLogger.debug(
                        Category.RETRIEVAL,
                        "Retrieved " + timedResponse.references().size() + " source references");
                  }
                });
          } catch (Exception e) {
            Platform.runLater(
                () -> {
                  addSystemMessage("Error: " + e.getMessage());
                  eventLogger.error(Category.LLM, "Query failed: " + e.getMessage());
                });
          } finally {
            Platform.runLater(
                () -> {
                  setInputEnabled(true);
                  settingsPanel.setStatus("Ready");
                  inputField.requestFocus();
                });
          }
        });
  }

  private void initializeRAGService() {
    if (serviceFactory == null) {
      addSystemMessage("Service factory not initialized. Check server connection.");
      eventLogger.error(Category.SYSTEM, "Service factory not initialized");
      return;
    }

    // Validate configuration
    if (!config.validateConfig()) {
      addSystemMessage(
          "Configuration validation failed. Please check application.properties and"
              + " ensure API keys are set.");
      eventLogger.error(Category.SYSTEM, "Configuration validation failed");
      return;
    }

    try {
      LLMConfig llmConfig = config.getLLMConfig();
      eventLogger.info(
          Category.LLM,
          "Initializing "
              + llmConfig.provider().getDisplayName()
              + " ("
              + llmConfig.model()
              + ")...");
      ragService = serviceFactory.createRAGService(llmConfig);
      addSystemMessage("Initialized " + llmConfig.provider().getDisplayName() + " successfully.");
      eventLogger.info(Category.LLM, "LLM initialized successfully");
    } catch (Exception e) {
      addSystemMessage("Failed to initialize LLM: " + e.getMessage());
      eventLogger.error(Category.LLM, "LLM initialization failed: " + e.getMessage());
    }
  }

  private void uploadPdf() {
    if (serviceFactory == null) {
      addSystemMessage("Service factory not initialized. Check server connection.");
      eventLogger.error(Category.PDF, "Upload failed - service factory not initialized");
      return;
    }

    FileChooser fileChooser = new FileChooser();
    fileChooser.setTitle("Select PDF Document");
    fileChooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("PDF Files", "*.pdf"));

    File file = fileChooser.showOpenDialog(getScene().getWindow());
    if (file != null) {
      settingsPanel.setStatus("Processing PDF: " + file.getName());
      eventLogger.info(Category.PDF, "Loading PDF: " + file.getName());

      // Disable input and show thinking bubble while processing
      setInputEnabled(false);
      ChatMessage thinkingMsg =
          ChatMessage.thinking(
              "Processing PDF: "
                  + file.getName()
                  + "\nExtracting text, images, and building search index...");
      messages.add(thinkingMsg);
      scrollToBottom();

      // Load PDF into viewer immediately (on JavaFX thread)
      try {
        pdfViewer.loadPDF(file);
        eventLogger.debug(
            Category.PDF, "PDF viewer loaded: " + pdfViewer.getTotalPages() + " pages");
      } catch (Exception e) {
        addSystemMessage("Warning: Could not display PDF preview: " + e.getMessage());
        eventLogger.warn(Category.PDF, "PDF preview failed: " + e.getMessage());
      }

      // Index PDF in background
      long startTime = System.currentTimeMillis();
      executor.submit(
          () -> {
            try {
              // Reset collection to ensure a clean slate (old data + H2 text index wiped)
              serviceFactory.resetCollection();

              // Clear semantic cache since document context is changing
              if (ragService != null) {
                ragService.clearCache();
              }

              // Initialize PDF service if needed
              if (pdfIngestionService == null) {
                pdfIngestionService = serviceFactory.createPDFIngestionService();
              }

              // Deterministic document ID from filename so re-uploading replaces via upsert
              String documentId = file.getName().replaceAll("[^a-zA-Z0-9._-]", "_").toLowerCase();
              int chunks = pdfIngestionService.ingestPDF(file, documentId);

              long elapsed = System.currentTimeMillis() - startTime;

              Platform.runLater(
                  () -> {
                    messages.remove(thinkingMsg); // Remove thinking bubble
                    addSystemMessage(
                        String.format(
                            "PDF processed successfully. Indexed %d chunks."
                                + " You can now ask questions about the"
                                + " document.",
                            chunks));
                    settingsPanel.setLoadedPdf(file.getName() + " (" + chunks + " chunks)");
                    settingsPanel.setStatus("Ready");
                    setInputEnabled(true);
                    inputField.requestFocus();
                    eventLogger.info(
                        Category.PDF, "Indexed " + chunks + " chunks in " + elapsed + "ms");
                  });
            } catch (Exception e) {
              Platform.runLater(
                  () -> {
                    messages.remove(thinkingMsg); // Remove thinking bubble
                    addSystemMessage("Error processing PDF: " + e.getMessage());
                    settingsPanel.setStatus("Ready");
                    setInputEnabled(true);
                    eventLogger.error(Category.PDF, "Indexing failed: " + e.getMessage());
                  });
            }
          });
    }
  }

  private void addSystemMessage(String content) {
    ChatMessage systemMessage =
        new ChatMessage(
            UUID.randomUUID().toString(),
            ChatMessage.Role.ASSISTANT,
            content,
            Instant.now(),
            0,
            0.0,
            "System",
            false,
            List.of(),
            0,
            null);
    messages.add(systemMessage);
    scrollToBottom();
  }

  private void setInputEnabled(boolean enabled) {
    inputField.setDisable(!enabled);
    sendButton.setDisable(!enabled);
  }

  private void scrollToBottom() {
    Platform.runLater(
        () -> {
          if (!messages.isEmpty()) {
            messageListView.scrollTo(messages.size() - 1);
          }
        });
  }

  /**
   * Sets the service factory.
   *
   * @param serviceFactory Service factory instance
   */
  public void setServiceFactory(ServiceFactory serviceFactory) {
    this.serviceFactory = serviceFactory;
    settingsPanel.setStatus("Connected to vectors-server. Upload a PDF to start.");
    eventLogger.info(Category.SYSTEM, "Connected to vectors-server");
  }

  /** Shutdown executor and close resources. */
  public void shutdown() {
    executor.shutdown();
    if (pdfViewer != null) {
      pdfViewer.close();
    }
  }

  /**
   * Gets the PDF viewer panel for external navigation.
   *
   * @return PDFViewerPanel instance
   */
  public PDFViewerPanel getPdfViewer() {
    return pdfViewer;
  }

  /**
   * Gets the semantic router panel for external configuration.
   *
   * @return SemanticRouterPanel instance
   */
  public SemanticRouterPanel getRouterPanel() {
    return routerPanel;
  }

  /**
   * Handles router enabled/disabled toggle changes.
   *
   * @param enabled Whether routing is now enabled
   */
  private void onRouterEnabledChange(boolean enabled) {
    if (enabled) {
      // Initialize the semantic router with sample routes
      if (semanticRouter == null && serviceFactory != null) {
        eventLogger.info(Category.RETRIEVAL, "Initializing semantic router...");
        routerPanel.setStatus("Initializing router...");

        executor.submit(
            () -> {
              try {
                // Convert sample routes to Route objects
                List<Route> routes =
                    routerPanel.getSampleRoutes().stream()
                        .map(
                            sample ->
                                Route.builder()
                                    .name(sample.name())
                                    .references(sample.references())
                                    .distanceThreshold(sample.threshold())
                                    .build())
                        .toList();

                semanticRouter = serviceFactory.createSemanticRouter(routes);
                routerPanel.setRouter(semanticRouter);

                Platform.runLater(
                    () -> {
                      eventLogger.info(
                          Category.RETRIEVAL,
                          "Semantic router initialized with " + routes.size() + " routes");
                      routerPanel.setStatus("Router ready - " + routes.size() + " routes");
                      addSystemMessage(
                          "Semantic router enabled with "
                              + routes.size()
                              + " routes:\n\u2022 "
                              + String.join(
                                  "\n\u2022 ", routes.stream().map(Route::getName).toList()));
                    });
              } catch (Exception e) {
                Platform.runLater(
                    () -> {
                      eventLogger.error(
                          Category.RETRIEVAL, "Router initialization failed: " + e.getMessage());
                      routerPanel.setStatus("Router initialization failed");
                      routerPanel.setRouterEnabled(false);
                      addSystemMessage("Failed to initialize semantic router: " + e.getMessage());
                    });
              }
            });
      } else if (semanticRouter != null) {
        eventLogger.info(Category.RETRIEVAL, "Semantic router enabled");
        addSystemMessage("Semantic router enabled. Off-topic queries will be blocked.");
      } else {
        eventLogger.warn(Category.RETRIEVAL, "Cannot enable router - services not initialized");
        routerPanel.setStatus("Connect to server first");
        routerPanel.setRouterEnabled(false);
      }
    } else {
      eventLogger.info(Category.RETRIEVAL, "Semantic router disabled");
      addSystemMessage("Semantic router disabled. All queries will be processed.");
    }
  }

  private static String truncate(String text, int maxLength) {
    if (text == null || text.length() <= maxLength) {
      return text;
    }
    return text.substring(0, maxLength) + "...";
  }
}
