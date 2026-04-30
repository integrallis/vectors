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

import com.integrallis.vectors.demo.rag.model.LogEntry;
import com.integrallis.vectors.demo.rag.service.EventLogger;
import java.util.function.Consumer;
import javafx.application.Platform;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.ListCell;
import javafx.scene.control.ListView;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.Region;

/**
 * Log panel content for the vertical tab pane.
 *
 * <p>Shows event log with color-coded entries by level.
 */
public class LogPanel extends BorderPane {

  private final ObservableList<LogEntry> entries = FXCollections.observableArrayList();
  private final ListView<LogEntry> logListView;
  private Consumer<Integer> countListener;

  private EventLogger eventLogger;

  /** Creates the log panel. */
  @SuppressWarnings("this-escape")
  public LogPanel() {
    getStyleClass().add("log-panel");

    // Header
    HBox header = new HBox(10);
    header.setAlignment(Pos.CENTER_LEFT);
    header.setPadding(new Insets(12, 16, 12, 16));
    header.getStyleClass().add("log-header");

    Label titleLabel = new Label("Event Log");
    titleLabel.getStyleClass().add("log-title");

    Region spacer = new Region();
    HBox.setHgrow(spacer, Priority.ALWAYS);

    Button clearButton = new Button("Clear");
    clearButton.getStyleClass().add("log-clear-button");
    clearButton.setOnAction(e -> clear());

    header.getChildren().addAll(titleLabel, spacer, clearButton);
    setTop(header);

    // Log list
    logListView = new ListView<>(entries);
    logListView.getStyleClass().add("log-list");
    logListView.setCellFactory(lv -> new LogEntryCell());
    logListView.setFocusTraversable(false);
    setCenter(logListView);
  }

  /**
   * Connects the panel to an EventLogger.
   *
   * @param logger EventLogger to receive events from
   */
  public void setEventLogger(EventLogger logger) {
    this.eventLogger = logger;

    // Add existing entries
    entries.addAll(logger.getEntries());
    notifyCountListener();

    // Listen for new entries
    logger.addListener(this::onNewEntry);
  }

  /**
   * Sets a listener to be notified when the entry count changes.
   *
   * @param listener Consumer that receives the new count
   */
  public void setCountListener(Consumer<Integer> listener) {
    this.countListener = listener;
    notifyCountListener();
  }

  private void onNewEntry(LogEntry entry) {
    Platform.runLater(
        () -> {
          entries.add(entry);
          notifyCountListener();
          logListView.scrollTo(entries.size() - 1);
        });
  }

  /** Scrolls to the bottom of the log. */
  public void scrollToBottom() {
    if (!entries.isEmpty()) {
      logListView.scrollTo(entries.size() - 1);
    }
  }

  private void clear() {
    entries.clear();
    if (eventLogger != null) {
      eventLogger.clear();
    }
    notifyCountListener();
  }

  private void notifyCountListener() {
    if (countListener != null) {
      countListener.accept(entries.size());
    }
  }

  /** Returns the current entry count. */
  public int getEntryCount() {
    return entries.size();
  }

  /** Custom cell for rendering log entries with color coding. */
  private static class LogEntryCell extends ListCell<LogEntry> {
    @Override
    protected void updateItem(LogEntry item, boolean empty) {
      super.updateItem(item, empty);

      if (empty || item == null) {
        setText(null);
        setGraphic(null);
        getStyleClass().removeAll("log-info", "log-debug", "log-warn", "log-error");
      } else {
        setText(item.formatted());

        // Apply style based on level
        getStyleClass().removeAll("log-info", "log-debug", "log-warn", "log-error");
        switch (item.level()) {
          case INFO -> getStyleClass().add("log-info");
          case DEBUG -> getStyleClass().add("log-debug");
          case WARN -> getStyleClass().add("log-warn");
          case ERROR -> getStyleClass().add("log-error");
        }
      }
    }
  }
}
