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
import com.integrallis.vectors.demo.rag.model.LLMConfig;
import com.integrallis.vectors.demo.rag.model.ModelRegistry;
import java.util.List;
import java.util.function.Consumer;
import javafx.geometry.Insets;
import javafx.scene.control.Button;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.ListCell;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.Separator;
import javafx.scene.control.Slider;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.Region;
import javafx.scene.layout.VBox;

/**
 * Settings panel content for the vertical tab pane.
 *
 * <p>Contains model selector, parameters, cache toggle, upload button, and status display.
 */
public class SettingsPanel extends ScrollPane {

  private final ComboBox<LLMConfig.Provider> providerComboBox;
  private final ComboBox<String> modelComboBox;
  private final ComboBox<CacheType> cacheComboBox;
  private final Slider chunkSizeSlider;
  private final Slider chunkOverlapSlider;
  private final Slider thresholdSlider;
  private final Button uploadPdfButton;
  private final Label statusLabel;
  private final Label loadedPdfLabel;
  private final Label chunkSizeValue;
  private final Label chunkOverlapValue;
  private final Label thresholdValue;

  private Consumer<LLMConfig.Provider> onProviderChange;
  private Consumer<String> onModelChange;
  private Consumer<Integer> onChunkSizeChange;
  private Consumer<Integer> onChunkOverlapChange;
  private Consumer<Double> onThresholdChange;

  /** Creates the settings panel. */
  @SuppressWarnings("this-escape")
  public SettingsPanel() {
    AppConfig config = AppConfig.getInstance();

    VBox content = new VBox(12);
    content.setPadding(new Insets(16));
    content.getStyleClass().add("settings-panel");

    // === Model Selection Section ===
    Label modelTitle = new Label("LLM Provider");
    modelTitle.getStyleClass().add("section-title");

    // Provider dropdown
    providerComboBox = new ComboBox<>();
    providerComboBox.getStyleClass().add("cache-combobox");
    providerComboBox.setMaxWidth(Double.MAX_VALUE);
    providerComboBox.setCellFactory(lv -> new ProviderCell());
    providerComboBox.setButtonCell(new ProviderCell());

    // Populate with available providers
    List<LLMConfig.Provider> availableProviders = ModelRegistry.getAvailableProviders();
    providerComboBox.getItems().addAll(availableProviders);

    // Set initial provider
    LLMConfig.Provider initialProvider = config.getLLMProvider();
    if (availableProviders.contains(initialProvider)) {
      providerComboBox.setValue(initialProvider);
    } else if (!availableProviders.isEmpty()) {
      providerComboBox.setValue(availableProviders.getFirst());
    }

    // Model dropdown
    Label modelLabel = new Label("Model");
    modelLabel.getStyleClass().add("sub-label");
    modelLabel.setPadding(new Insets(8, 0, 0, 0));

    modelComboBox = new ComboBox<>();
    modelComboBox.getStyleClass().add("cache-combobox");
    modelComboBox.setMaxWidth(Double.MAX_VALUE);

    // Populate models for initial provider
    updateModelsForProvider(providerComboBox.getValue());

    // Provider change listener
    providerComboBox
        .valueProperty()
        .addListener(
            (obs, oldVal, newVal) -> {
              if (newVal != null) {
                updateModelsForProvider(newVal);
                if (onProviderChange != null) {
                  onProviderChange.accept(newVal);
                }
              }
            });

    // Model change listener
    modelComboBox
        .valueProperty()
        .addListener(
            (obs, oldVal, newVal) -> {
              if (newVal != null && onModelChange != null) {
                onModelChange.accept(newVal);
              }
            });

    // API key info
    Label apiKeyInfo = new Label(getApiKeyStatus());
    apiKeyInfo.getStyleClass().add("info-text");
    apiKeyInfo.setWrapText(true);

    Separator modelSep = new Separator();

    // === Parameters Section ===
    Label paramsTitle = new Label("RAG Parameters");
    paramsTitle.getStyleClass().add("section-title");

    // Chunk size slider
    Label chunkSizeLabel = new Label("Chunk Size");
    chunkSizeLabel.getStyleClass().add("sub-label");

    chunkSizeSlider = new Slider(200, 2000, 500);
    chunkSizeSlider.setBlockIncrement(100);
    chunkSizeSlider.setMajorTickUnit(400);
    chunkSizeSlider.setMinorTickCount(3);
    chunkSizeSlider.setShowTickMarks(true);
    chunkSizeSlider.getStyleClass().add("param-slider");

    chunkSizeValue = new Label("500");
    chunkSizeValue.getStyleClass().add("slider-value");
    chunkSizeValue.setMinWidth(40);

    HBox chunkSizeBox = new HBox(8, chunkSizeSlider, chunkSizeValue);
    HBox.setHgrow(chunkSizeSlider, Priority.ALWAYS);

    chunkSizeSlider
        .valueProperty()
        .addListener(
            (obs, oldVal, newVal) -> {
              int value = newVal.intValue();
              chunkSizeValue.setText(String.valueOf(value));
              if (onChunkSizeChange != null) {
                onChunkSizeChange.accept(value);
              }
            });

    // Chunk overlap slider
    Label chunkOverlapLabel = new Label("Chunk Overlap");
    chunkOverlapLabel.getStyleClass().add("sub-label");

    chunkOverlapSlider = new Slider(0, 500, 50);
    chunkOverlapSlider.setBlockIncrement(50);
    chunkOverlapSlider.setMajorTickUnit(100);
    chunkOverlapSlider.setMinorTickCount(1);
    chunkOverlapSlider.setShowTickMarks(true);
    chunkOverlapSlider.getStyleClass().add("param-slider");

    chunkOverlapValue = new Label("50");
    chunkOverlapValue.getStyleClass().add("slider-value");
    chunkOverlapValue.setMinWidth(40);

    HBox chunkOverlapBox = new HBox(8, chunkOverlapSlider, chunkOverlapValue);
    HBox.setHgrow(chunkOverlapSlider, Priority.ALWAYS);

    chunkOverlapSlider
        .valueProperty()
        .addListener(
            (obs, oldVal, newVal) -> {
              int value = newVal.intValue();
              chunkOverlapValue.setText(String.valueOf(value));
              if (onChunkOverlapChange != null) {
                onChunkOverlapChange.accept(value);
              }
            });

    Separator paramsSep = new Separator();

    // === Cache Section ===
    Label cacheTitle = new Label("Semantic Cache");
    cacheTitle.getStyleClass().add("section-title");

    cacheComboBox = new ComboBox<>();
    cacheComboBox.getItems().addAll(CacheType.values());
    cacheComboBox.setValue(CacheType.LOCAL);
    cacheComboBox.getStyleClass().add("cache-combobox");
    cacheComboBox.setMaxWidth(Double.MAX_VALUE);

    // Similarity threshold slider (for cache hit detection)
    Label thresholdLabel = new Label("Similarity Threshold");
    thresholdLabel.getStyleClass().add("sub-label");
    thresholdLabel.setPadding(new Insets(8, 0, 0, 0));

    thresholdSlider = new Slider(0.0, 1.0, 0.75);
    thresholdSlider.setBlockIncrement(0.05);
    thresholdSlider.setMajorTickUnit(0.2);
    thresholdSlider.setMinorTickCount(1);
    thresholdSlider.setShowTickMarks(true);
    thresholdSlider.getStyleClass().add("param-slider");

    thresholdValue = new Label("0.75");
    thresholdValue.getStyleClass().add("slider-value");
    thresholdValue.setMinWidth(40);

    HBox thresholdBox = new HBox(8, thresholdSlider, thresholdValue);
    HBox.setHgrow(thresholdSlider, Priority.ALWAYS);

    thresholdSlider
        .valueProperty()
        .addListener(
            (obs, oldVal, newVal) -> {
              double value = newVal.doubleValue();
              thresholdValue.setText(String.format("%.2f", value));
              if (onThresholdChange != null) {
                onThresholdChange.accept(value);
              }
            });

    Separator cacheSep = new Separator();

    // === Document Section ===
    Label docTitle = new Label("Document");
    docTitle.getStyleClass().add("section-title");

    loadedPdfLabel = new Label("No PDF loaded");
    loadedPdfLabel.getStyleClass().add("info-label");
    loadedPdfLabel.setWrapText(true);

    uploadPdfButton = new Button("Upload PDF");
    uploadPdfButton.getStyleClass().add("action-button");
    uploadPdfButton.setMaxWidth(Double.MAX_VALUE);

    Separator docSep = new Separator();

    // === Status Section ===
    Label statusTitle = new Label("Status");
    statusTitle.getStyleClass().add("section-title");

    statusLabel = new Label("Ready");
    statusLabel.getStyleClass().add("status-label");
    statusLabel.setWrapText(true);

    // Spacer
    Region spacer = new Region();
    VBox.setVgrow(spacer, Priority.ALWAYS);

    content
        .getChildren()
        .addAll(
            // Document section FIRST for easy PDF upload visibility
            docTitle,
            loadedPdfLabel,
            uploadPdfButton,
            docSep,
            modelTitle,
            providerComboBox,
            modelLabel,
            modelComboBox,
            apiKeyInfo,
            modelSep,
            paramsTitle,
            chunkSizeLabel,
            chunkSizeBox,
            chunkOverlapLabel,
            chunkOverlapBox,
            paramsSep,
            cacheTitle,
            cacheComboBox,
            thresholdLabel,
            thresholdBox,
            cacheSep,
            spacer,
            statusTitle,
            statusLabel);

    setContent(content);
    setFitToWidth(true);
    setHbarPolicy(ScrollBarPolicy.NEVER);
    getStyleClass().add("settings-scroll");
  }

  private void updateModelsForProvider(LLMConfig.Provider provider) {
    if (provider == null) return;
    modelComboBox.getItems().clear();
    List<String> models = ModelRegistry.getModelsForProvider(provider);
    modelComboBox.getItems().addAll(models);
    if (!models.isEmpty()) {
      modelComboBox.setValue(models.getFirst());
    }
  }

  private String getApiKeyStatus() {
    StringBuilder sb = new StringBuilder();
    sb.append("API Keys:\n");
    for (LLMConfig.Provider p : LLMConfig.Provider.values()) {
      String status = ModelRegistry.isProviderAvailable(p) ? "\u2713" : "\u2717";
      sb.append("  ").append(status).append(" ").append(p.getDisplayName()).append("\n");
    }
    return sb.toString().trim();
  }

  /** Cell renderer for provider dropdown. */
  private static class ProviderCell extends ListCell<LLMConfig.Provider> {
    @Override
    protected void updateItem(LLMConfig.Provider item, boolean empty) {
      super.updateItem(item, empty);
      if (empty || item == null) {
        setText(null);
      } else {
        setText(item.getDisplayName());
      }
    }
  }

  // === Getters ===

  public ComboBox<LLMConfig.Provider> getProviderComboBox() {
    return providerComboBox;
  }

  public ComboBox<String> getModelComboBox() {
    return modelComboBox;
  }

  public ComboBox<CacheType> getCacheComboBox() {
    return cacheComboBox;
  }

  public Button getUploadPdfButton() {
    return uploadPdfButton;
  }

  public Slider getChunkSizeSlider() {
    return chunkSizeSlider;
  }

  public Slider getChunkOverlapSlider() {
    return chunkOverlapSlider;
  }

  public Slider getThresholdSlider() {
    return thresholdSlider;
  }

  public int getChunkSize() {
    return (int) chunkSizeSlider.getValue();
  }

  public int getChunkOverlap() {
    return (int) chunkOverlapSlider.getValue();
  }

  public double getThreshold() {
    return thresholdSlider.getValue();
  }

  public LLMConfig.Provider getSelectedProvider() {
    return providerComboBox.getValue();
  }

  public String getSelectedModel() {
    return modelComboBox.getValue();
  }

  // === Setters ===

  public void setStatus(String status) {
    statusLabel.setText(status);
  }

  public void setLoadedPdf(String pdfName) {
    loadedPdfLabel.setText(pdfName != null ? "PDF: " + pdfName : "No PDF loaded");
  }

  // === Callbacks ===

  public void setOnProviderChange(Consumer<LLMConfig.Provider> callback) {
    this.onProviderChange = callback;
  }

  public void setOnModelChange(Consumer<String> callback) {
    this.onModelChange = callback;
  }

  public void setOnChunkSizeChange(Consumer<Integer> callback) {
    this.onChunkSizeChange = callback;
  }

  public void setOnChunkOverlapChange(Consumer<Integer> callback) {
    this.onChunkOverlapChange = callback;
  }

  public void setOnThresholdChange(Consumer<Double> callback) {
    this.onThresholdChange = callback;
  }
}
