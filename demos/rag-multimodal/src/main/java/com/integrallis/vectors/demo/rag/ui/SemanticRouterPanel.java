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

import com.integrallis.vectors.router.Route;
import com.integrallis.vectors.router.RouteMatch;
import com.integrallis.vectors.router.SemanticRouter;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;
import javafx.application.Platform;
import javafx.beans.property.BooleanProperty;
import javafx.beans.property.SimpleBooleanProperty;
import javafx.geometry.Insets;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.Label;
import javafx.scene.control.ListView;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.Separator;
import javafx.scene.control.TextField;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;
import javafx.stage.FileChooser;

/**
 * Semantic Router panel for testing query routing.
 *
 * <p>Displays configured routes and allows testing queries against the router. Supports loading
 * routes from YAML/JSON configuration files.
 */
public class SemanticRouterPanel extends ScrollPane {

  /** Sample route for off-topic queries. */
  public static final SampleRoute OFF_TOPIC =
      new SampleRoute(
          "off-topic",
          List.of(
              "what does the paper say about aliens?",
              "are aliens real?",
              "tell me a joke",
              "what's the weather like?"),
          0.3f);

  /** Sample route for political queries. */
  public static final SampleRoute POLITICS =
      new SampleRoute(
          "politics",
          List.of(
              "who did you vote for in this past election?",
              "what are your political beliefs?",
              "why did you vote for that candidate for president?"),
          0.3f);

  /** Sample route for PII requests. */
  public static final SampleRoute PII =
      new SampleRoute(
          "pii",
          List.of(
              "tell me your phone number",
              "tell me your social security number",
              "tell me your address and date of birth"),
          0.3f);

  /** All sample routes. */
  public static final List<SampleRoute> SAMPLE_ROUTES = List.of(OFF_TOPIC, POLITICS, PII);

  private final ListView<String> routeListView;
  private final TextField testQueryField;
  private final Button testButton;
  private final Label resultLabel;
  private final Label statusLabel;
  private final Label configFileLabel;
  private final VBox routeDetailsBox;
  private final BooleanProperty routerEnabled = new SimpleBooleanProperty(false);

  private SemanticRouter router;
  private List<SampleRoute> displayedRoutes = new ArrayList<>();
  private Consumer<String> onRouteTest;
  private Consumer<Boolean> onEnabledChange;
  private String loadedConfigPath;

  /** Creates the semantic router panel with sample routes preloaded. */
  @SuppressWarnings("this-escape")
  public SemanticRouterPanel() {
    VBox content = new VBox(12);
    content.setPadding(new Insets(16));
    content.getStyleClass().add("settings-panel");

    // === Enable Toggle Section ===
    Label enableTitle = new Label("Semantic Router");
    enableTitle.getStyleClass().add("section-title");

    CheckBox enabledCheckBox = new CheckBox("Enable routing");
    enabledCheckBox.getStyleClass().add("router-checkbox");
    enabledCheckBox.selectedProperty().bindBidirectional(routerEnabled);
    enabledCheckBox
        .selectedProperty()
        .addListener(
            (obs, oldVal, newVal) -> {
              updateUIState();
              if (onEnabledChange != null) {
                onEnabledChange.accept(newVal);
              }
            });

    Label enableHint = new Label("When enabled, queries are checked against routes before RAG");
    enableHint.getStyleClass().add("hint-label");
    enableHint.setWrapText(true);

    Separator enableSep = new Separator();

    // === Configuration Section ===
    Label configTitle = new Label("Route Configuration");
    configTitle.getStyleClass().add("section-title");

    configFileLabel = new Label("Using sample routes (3 routes)");
    configFileLabel.getStyleClass().add("info-label");
    configFileLabel.setWrapText(true);

    Button loadConfigButton = new Button("Load YAML/JSON Config");
    loadConfigButton.getStyleClass().add("action-button");
    loadConfigButton.setMaxWidth(Double.MAX_VALUE);
    loadConfigButton.setOnAction(e -> loadConfigFile());

    Separator configSep = new Separator();

    // === Routes Section ===
    Label routesTitle = new Label("Configured Routes");
    routesTitle.getStyleClass().add("section-title");

    routeListView = new ListView<>();
    routeListView.setPrefHeight(120);
    routeListView.getStyleClass().add("route-list");
    routeListView.setPlaceholder(new Label("No routes configured"));

    // Route selection shows details
    routeListView
        .getSelectionModel()
        .selectedItemProperty()
        .addListener(
            (obs, oldVal, newVal) -> {
              if (newVal != null) {
                showRouteDetails(newVal);
              }
            });

    // Route details box
    routeDetailsBox = new VBox(4);
    routeDetailsBox.getStyleClass().add("route-details");
    routeDetailsBox.setPadding(new Insets(8));

    Separator routesSep = new Separator();

    // === Test Section ===
    Label testTitle = new Label("Test Routing");
    testTitle.getStyleClass().add("section-title");

    Label testLabel = new Label("Enter a query to test:");
    testLabel.getStyleClass().add("sub-label");

    testQueryField = new TextField();
    testQueryField.setPromptText("Type a test query...");
    testQueryField.getStyleClass().add("input-field");
    testQueryField.setOnAction(e -> testRoute());

    testButton = new Button("Test Route");
    testButton.getStyleClass().add("action-button");
    testButton.setMaxWidth(Double.MAX_VALUE);
    testButton.setOnAction(e -> testRoute());
    testButton.setDisable(true);

    HBox testButtonBox = new HBox(testButton);
    HBox.setHgrow(testButton, Priority.ALWAYS);

    Separator testSep = new Separator();

    // === Result Section ===
    Label resultTitle = new Label("Routing Result");
    resultTitle.getStyleClass().add("section-title");

    resultLabel = new Label("No test results yet");
    resultLabel.getStyleClass().add("info-label");
    resultLabel.setWrapText(true);

    Separator resultSep = new Separator();

    // === Status Section ===
    Label statusTitle = new Label("Status");
    statusTitle.getStyleClass().add("section-title");

    statusLabel = new Label("Router disabled");
    statusLabel.getStyleClass().add("status-label");
    statusLabel.setWrapText(true);

    content
        .getChildren()
        .addAll(
            enableTitle,
            enabledCheckBox,
            enableHint,
            enableSep,
            configTitle,
            configFileLabel,
            loadConfigButton,
            configSep,
            routesTitle,
            routeListView,
            routeDetailsBox,
            routesSep,
            testTitle,
            testLabel,
            testQueryField,
            testButtonBox,
            testSep,
            resultTitle,
            resultLabel,
            resultSep,
            statusTitle,
            statusLabel);

    setContent(content);
    setFitToWidth(true);
    setHbarPolicy(ScrollBarPolicy.NEVER);
    getStyleClass().add("settings-scroll");

    // Load sample routes by default
    loadSampleRoutes();

    // Initial state
    updateUIState();
  }

  /** Loads sample routes into the display (without requiring a SemanticRouter). */
  private void loadSampleRoutes() {
    displayedRoutes = new ArrayList<>(SAMPLE_ROUTES);
    routeListView.getItems().clear();
    for (SampleRoute route : displayedRoutes) {
      routeListView.getItems().add(route.name());
    }
    if (!displayedRoutes.isEmpty()) {
      routeListView.getSelectionModel().select(0);
    }
  }

  /** Updates UI state based on enabled status. */
  private void updateUIState() {
    boolean enabled = routerEnabled.get();
    boolean hasRouter = router != null;

    testButton.setDisable(!enabled || !hasRouter);
    testQueryField.setDisable(!enabled);
    routeListView.setDisable(!enabled);

    if (!enabled) {
      statusLabel.setText("Router disabled - " + displayedRoutes.size() + " route(s) configured");
    } else if (!hasRouter) {
      statusLabel.setText("Enable routing to test - " + displayedRoutes.size() + " route(s)");
    } else {
      statusLabel.setText("Ready - " + router.getRouteNames().size() + " route(s)");
    }
  }

  /** Opens file chooser to load YAML or JSON config. */
  private void loadConfigFile() {
    FileChooser fileChooser = new FileChooser();
    fileChooser.setTitle("Load Router Configuration");
    fileChooser
        .getExtensionFilters()
        .addAll(
            new FileChooser.ExtensionFilter("Config Files", "*.yaml", "*.yml", "*.json"),
            new FileChooser.ExtensionFilter("YAML Files", "*.yaml", "*.yml"),
            new FileChooser.ExtensionFilter("JSON Files", "*.json"),
            new FileChooser.ExtensionFilter("All Files", "*.*"));

    File file = fileChooser.showOpenDialog(getScene().getWindow());
    if (file != null) {
      loadedConfigPath = file.getAbsolutePath();
      configFileLabel.setText("Config: " + file.getName());
      statusLabel.setText("Config file selected. Initialize via application.");
    }
  }

  /**
   * Gets the path to the loaded config file.
   *
   * @return Path to config file, or null if none loaded
   */
  public String getLoadedConfigPath() {
    return loadedConfigPath;
  }

  /**
   * Gets the sample routes for external initialization.
   *
   * @return List of sample routes
   */
  public List<SampleRoute> getSampleRoutes() {
    return displayedRoutes;
  }

  /**
   * Sets the semantic router instance.
   *
   * @param router SemanticRouter to use for testing
   */
  public void setRouter(SemanticRouter router) {
    this.router = router;
    Platform.runLater(this::refreshRoutes);
  }

  /** Refreshes the route list from the current router. */
  public void refreshRoutes() {
    routeDetailsBox.getChildren().clear();

    if (router == null) {
      // Keep sample routes displayed
      updateUIState();
      return;
    }

    // Update from real router
    routeListView.getItems().clear();
    displayedRoutes.clear();

    List<String> routeNames = router.getRouteNames();
    if (routeNames.isEmpty()) {
      loadSampleRoutes(); // Fallback to sample routes
    } else {
      routeListView.getItems().addAll(routeNames);
      // Select first route
      if (!routeNames.isEmpty()) {
        routeListView.getSelectionModel().select(0);
      }
    }

    updateUIState();
  }

  /**
   * Shows details for the selected route.
   *
   * @param routeName Name of the route to show details for
   */
  private void showRouteDetails(String routeName) {
    routeDetailsBox.getChildren().clear();

    // Try real router first
    if (router != null) {
      Route route = router.get(routeName);
      if (route != null) {
        showRouteDetailsFromRoute(
            route.getName(), route.getReferences(), (float) route.getDistanceThreshold());
        return;
      }
    }

    // Fall back to sample routes
    for (SampleRoute sample : displayedRoutes) {
      if (sample.name().equals(routeName)) {
        showRouteDetailsFromRoute(sample.name(), sample.references(), sample.threshold());
        return;
      }
    }
  }

  private void showRouteDetailsFromRoute(String name, List<String> references, float threshold) {
    Label nameLabel = new Label("Name: " + name);
    nameLabel.getStyleClass().add("detail-label");

    Label thresholdLabel = new Label(String.format("Threshold: %.2f", threshold));
    thresholdLabel.getStyleClass().add("detail-label");

    Label refsLabel = new Label("References: " + references.size());
    refsLabel.getStyleClass().add("detail-label");

    // Show first few references
    VBox refsBox = new VBox(2);
    refsBox.setPadding(new Insets(4, 0, 0, 8));
    int maxRefs = Math.min(3, references.size());
    for (int i = 0; i < maxRefs; i++) {
      String ref = references.get(i);
      String truncated = ref.length() > 40 ? ref.substring(0, 40) + "..." : ref;
      Label refLabel = new Label("\u2022 " + truncated);
      refLabel.getStyleClass().add("reference-label");
      refsBox.getChildren().add(refLabel);
    }
    if (references.size() > maxRefs) {
      Label moreLabel = new Label("  +" + (references.size() - maxRefs) + " more...");
      moreLabel.getStyleClass().add("reference-label");
      refsBox.getChildren().add(moreLabel);
    }

    routeDetailsBox.getChildren().addAll(nameLabel, thresholdLabel, refsLabel, refsBox);
  }

  /** Tests the current query against the router. */
  private void testRoute() {
    String query = testQueryField.getText().trim();
    if (query.isEmpty()) {
      resultLabel.setText("Please enter a test query");
      return;
    }

    if (router == null) {
      resultLabel.setText("Router not initialized");
      return;
    }

    if (!routerEnabled.get()) {
      resultLabel.setText("Router is disabled");
      return;
    }

    testButton.setDisable(true);
    statusLabel.setText("Testing...");

    // Run in background thread
    Thread testThread =
        new Thread(
            () -> {
              try {
                RouteMatch match = router.route(query);
                Platform.runLater(
                    () -> {
                      if (match.getName() != null) {
                        resultLabel.setText(
                            String.format(
                                "BLOCKED: %s%nDistance: %.4f",
                                match.getName(),
                                match.getDistance() != null ? match.getDistance() : 0.0));
                        resultLabel.setStyle("-fx-text-fill: #c62828;"); // Red for blocked
                      } else {
                        resultLabel.setText("ALLOWED (no route match)");
                        resultLabel.setStyle("-fx-text-fill: #2e7d32;"); // Green for allowed
                      }

                      if (onRouteTest != null) {
                        onRouteTest.accept(match.getName());
                      }
                    });
              } catch (Exception e) {
                Platform.runLater(
                    () -> {
                      resultLabel.setText("Error: " + e.getMessage());
                      resultLabel.setStyle("-fx-text-fill: #c62828;"); // Red for error
                    });
              } finally {
                Platform.runLater(
                    () -> {
                      testButton.setDisable(false);
                      statusLabel.setText("Ready - " + router.getRouteNames().size() + " route(s)");
                    });
              }
            });
    testThread.setDaemon(true);
    testThread.start();
  }

  /**
   * Returns whether the router is enabled.
   *
   * @return true if enabled
   */
  public boolean isRouterEnabled() {
    return routerEnabled.get();
  }

  /**
   * Sets whether the router is enabled.
   *
   * @param enabled true to enable
   */
  public void setRouterEnabled(boolean enabled) {
    routerEnabled.set(enabled);
  }

  /**
   * Gets the router enabled property.
   *
   * @return BooleanProperty for binding
   */
  public BooleanProperty routerEnabledProperty() {
    return routerEnabled;
  }

  /**
   * Sets the status message.
   *
   * @param status Status text
   */
  public void setStatus(String status) {
    statusLabel.setText(status);
  }

  /**
   * Sets a callback for route test completion.
   *
   * @param callback Callback receiving the matched route name (or null)
   */
  public void setOnRouteTest(Consumer<String> callback) {
    this.onRouteTest = callback;
  }

  /**
   * Sets a callback for enabled state changes.
   *
   * @param callback Callback receiving the new enabled state
   */
  public void setOnEnabledChange(Consumer<Boolean> callback) {
    this.onEnabledChange = callback;
  }

  /**
   * Gets the test query field for external access.
   *
   * @return TextField for test queries
   */
  public TextField getTestQueryField() {
    return testQueryField;
  }

  /**
   * Sets the config file label.
   *
   * @param text Text to display
   */
  public void setConfigFileLabel(String text) {
    configFileLabel.setText(text);
  }

  /** Record representing a sample route for display. */
  public record SampleRoute(String name, List<String> references, float threshold) {}
}
