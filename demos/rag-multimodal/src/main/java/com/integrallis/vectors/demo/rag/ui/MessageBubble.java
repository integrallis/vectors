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

import com.integrallis.vectors.demo.rag.model.ChatMessage;
import com.integrallis.vectors.demo.rag.model.Reference;
import com.integrallis.vectors.demo.rag.service.JTokKitCostTracker;
import java.io.ByteArrayInputStream;
import java.util.function.Consumer;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.control.Hyperlink;
import javafx.scene.control.Label;
import javafx.scene.control.ProgressIndicator;
import javafx.scene.control.Tooltip;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.FlowPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.scene.text.Text;
import javafx.scene.text.TextFlow;

/**
 * Custom JavaFX control for rendering chat message bubbles.
 *
 * <p>Displays message content with cost, timing, and token information for assistant messages. Can
 * also render inline images.
 */
public class MessageBubble extends HBox {

  private final ChatMessage message;
  private final Consumer<Integer> pageNavigator;

  /**
   * Creates a new message bubble.
   *
   * @param message Chat message to display
   */
  @SuppressWarnings("this-escape")
  public MessageBubble(ChatMessage message) {
    this(message, null);
  }

  /**
   * Creates a new message bubble with page navigation support.
   *
   * @param message Chat message to display
   * @param pageNavigator Callback for navigating to a page (receives 0-based page number)
   */
  @SuppressWarnings("this-escape")
  public MessageBubble(ChatMessage message, Consumer<Integer> pageNavigator) {
    this.message = message;
    this.pageNavigator = pageNavigator;
    build();
  }

  private void build() {
    setSpacing(10);
    setPadding(new Insets(5, 10, 5, 10));

    // Create message content container
    VBox contentBox = new VBox(5);
    contentBox.setMaxWidth(600);

    // Message text
    TextFlow textFlow = new TextFlow();
    Text messageText = new Text(message.content());
    messageText.setWrappingWidth(580);
    textFlow.getChildren().add(messageText);

    contentBox.getChildren().add(textFlow);

    // Add spinner for thinking/processing messages
    if (message.isThinking()) {
      HBox thinkingBox = new HBox(8);
      thinkingBox.setAlignment(Pos.CENTER_LEFT);
      ProgressIndicator spinner = new ProgressIndicator();
      spinner.setPrefSize(20, 20);
      spinner.setMaxSize(20, 20);
      Label thinkingLabel = new Label("Processing...");
      thinkingLabel.getStyleClass().add("thinking-label");
      thinkingBox.getChildren().addAll(spinner, thinkingLabel);
      contentBox.getChildren().add(thinkingBox);
      contentBox.getStyleClass().add("thinking");
    }

    // Add inline image if present
    if (message.imageBytes() != null && message.imageBytes().length > 0) {
      try {
        Image image = new Image(new ByteArrayInputStream(message.imageBytes()));
        ImageView imageView = new ImageView(image);
        imageView.setPreserveRatio(true);
        imageView.setFitWidth(500);
        imageView.setSmooth(true);
        contentBox.getChildren().add(imageView);
      } catch (Exception e) {
        Label errorLabel = new Label("Failed to render image: " + e.getMessage());
        errorLabel.getStyleClass().add("cost-label");
        contentBox.getChildren().add(errorLabel);
      }
    }

    // Add info line for assistant messages (timing, tokens, cost)
    if (message.role() == ChatMessage.Role.ASSISTANT) {
      StringBuilder infoText = new StringBuilder();

      // Always show elapsed time if available
      if (message.elapsedMs() > 0) {
        infoText.append(formatDuration(message.elapsedMs()));
      }

      // Show tokens and cost for LLM responses
      if (message.tokenCount() > 0) {
        if (infoText.length() > 0) {
          infoText.append(" \u2022 ");
        }
        infoText.append(JTokKitCostTracker.formatTokens(message.tokenCount()));
        infoText.append(" \u2022 ");
        infoText.append(JTokKitCostTracker.formatCost(message.costUsd()));
        infoText.append(" \u2022 ");
        infoText.append(message.model());
      }

      if (message.fromCache()) {
        if (infoText.length() > 0) {
          infoText.append(" \u2022 ");
        }
        infoText.append("From Cache");
      }

      if (infoText.length() > 0) {
        Label costLabel = new Label(infoText.toString());
        costLabel.getStyleClass().add("cost-label");
        if (message.fromCache()) {
          costLabel.getStyleClass().add("cached");
        }
        contentBox.getChildren().add(costLabel);
      }
    }

    // Add references section if available
    if (message.references() != null && !message.references().isEmpty()) {
      FlowPane refsPane = new FlowPane();
      refsPane.setHgap(5);
      refsPane.setVgap(3);
      refsPane.getStyleClass().add("references-pane");

      Label refsLabel = new Label("Sources:");
      refsLabel.getStyleClass().add("refs-label");
      refsPane.getChildren().add(refsLabel);

      for (Reference ref : message.references()) {
        Hyperlink pageLink = new Hyperlink("p." + ref.page());
        pageLink.getStyleClass().add("page-link");

        // Set tooltip with preview
        String tooltipText = ref.type() + ": " + ref.preview();
        Tooltip tooltip = new Tooltip(tooltipText);
        tooltip.setWrapText(true);
        tooltip.setMaxWidth(300);
        pageLink.setTooltip(tooltip);

        // Navigate to page on click (convert 1-indexed to 0-indexed)
        if (pageNavigator != null) {
          pageLink.setOnAction(e -> pageNavigator.accept(ref.page() - 1));
        } else {
          pageLink.setDisable(true);
        }

        refsPane.getChildren().add(pageLink);
      }

      contentBox.getChildren().add(refsPane);
    }

    // Style the bubble based on role
    contentBox.getStyleClass().add("message-bubble");
    if (message.role() == ChatMessage.Role.USER) {
      contentBox.getStyleClass().add("user");
      setAlignment(Pos.CENTER_RIGHT);
    } else {
      contentBox.getStyleClass().add("assistant");
      setAlignment(Pos.CENTER_LEFT);
      if (message.fromCache()) {
        contentBox.getStyleClass().add("cached");
      }
    }

    getChildren().add(contentBox);
  }

  /** Formats a duration in milliseconds as a human-readable string. */
  private static String formatDuration(long ms) {
    if (ms < 1000) {
      return ms + "ms";
    } else {
      return String.format("%.1fs", ms / 1000.0);
    }
  }

  /**
   * Gets the chat message.
   *
   * @return Chat message
   */
  public ChatMessage getMessage() {
    return message;
  }
}
