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

import com.integrallis.vectors.demo.rag.config.AppConfig;
import com.integrallis.vectors.demo.rag.service.ServiceFactory;
import com.integrallis.vectors.demo.rag.ui.ChatController;
import java.awt.Taskbar;
import java.io.InputStream;
import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.control.Alert;
import javafx.scene.control.Alert.AlertType;
import javafx.stage.Stage;

/**
 * Multimodal RAG demo application using java-vectors and LangChain4J.
 *
 * <p>Features:
 *
 * <ul>
 *   <li>Chat interface with message bubbles
 *   <li>Cost tracking per message
 *   <li>Local semantic caching
 *   <li>PDF ingestion with multimodal support
 *   <li>Multiple LLM provider support
 *   <li>Semantic routing for query filtering
 * </ul>
 */
public class MultimodalRAGApp extends Application {

  private ChatController chatController;
  private ServiceFactory serviceFactory;

  @Override
  public void start(Stage primaryStage) throws Exception {
    // Get vectors-server URL from config or environment
    String serverUrl =
        System.getenv()
            .getOrDefault("VECTORS_SERVER_URL", AppConfig.getInstance().getVectorsServerUrl());

    // Initialize services
    serviceFactory = new ServiceFactory();
    try {
      serviceFactory.initialize(serverUrl);
      System.out.println("Connected to vectors-server at " + serverUrl);
    } catch (Exception e) {
      showErrorDialog(
          "Connection Failed",
          "Could not connect to vectors-server at "
              + serverUrl
              + "\n\n"
              + "Error: "
              + e.getMessage()
              + "\n\n"
              + "Please ensure vectors-server is running and try again.\n"
              + "Start it with: docker compose -f demos/rag-multimodal/docker-compose.yml up -d");
      return;
    }

    // Create UI
    chatController = new ChatController();
    chatController.setServiceFactory(serviceFactory);

    Scene scene = new Scene(chatController, 1200, 800);
    scene.getStylesheets().add(getClass().getResource("/styles/app.css").toExternalForm());

    primaryStage.setTitle("java-vectors Multimodal RAG Demo");
    primaryStage
        .getIcons()
        .add(
            new javafx.scene.image.Image(
                getClass().getResourceAsStream("/icons/java-vectors.png")));
    primaryStage.setScene(scene);
    primaryStage.setOnCloseRequest(
        e -> {
          chatController.shutdown();
          if (serviceFactory != null) {
            serviceFactory.close();
          }
        });
    primaryStage.show();
  }

  private static void setDockIcon() {
    if (Taskbar.isTaskbarSupported()) {
      Taskbar taskbar = Taskbar.getTaskbar();
      if (taskbar.isSupported(Taskbar.Feature.ICON_IMAGE)) {
        try (InputStream is =
            MultimodalRAGApp.class.getResourceAsStream("/icons/java-vectors.png")) {
          if (is != null) {
            taskbar.setIconImage(javax.imageio.ImageIO.read(is));
          }
        } catch (Exception e) {
          System.err.println("Could not set Dock icon: " + e.getMessage());
        }
      }
    }
  }

  private void showErrorDialog(String title, String content) {
    Alert alert = new Alert(AlertType.ERROR);
    alert.setTitle(title);
    alert.setHeaderText(null);
    alert.setContentText(content);
    alert.showAndWait();
  }

  public static void main(String[] args) {
    System.out.println("java-vectors Multimodal RAG Demo");
    System.out.println("================================");
    System.out.println();
    System.out.println("Environment Variables:");
    System.out.println(
        "  VECTORS_SERVER_URL - vectors-server URL (default: http://localhost:8287)");
    System.out.println();
    System.out.println("Starting application...");

    // Set macOS Dock name and icon before AWT/JavaFX initializes
    System.setProperty("apple.awt.application.name", "java-vectors RAG");
    setDockIcon();

    launch(args);
  }
}
