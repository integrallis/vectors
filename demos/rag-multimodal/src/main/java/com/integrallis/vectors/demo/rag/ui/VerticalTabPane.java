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

import java.util.ArrayList;
import java.util.List;
import javafx.animation.KeyFrame;
import javafx.animation.KeyValue;
import javafx.animation.Timeline;
import javafx.beans.property.IntegerProperty;
import javafx.beans.property.SimpleIntegerProperty;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Node;
import javafx.scene.control.Label;
import javafx.scene.control.ToggleButton;
import javafx.scene.control.ToggleGroup;
import javafx.scene.control.Tooltip;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.StackPane;
import javafx.scene.layout.VBox;
import javafx.util.Duration;

/**
 * Material UI-style vertical tab pane with tabs on the right edge.
 *
 * <p>Features:
 *
 * <ul>
 *   <li>Tab bar fixed to right edge (44px)
 *   <li>Tabs can be added to top or bottom of the tab bar
 *   <li>Only one tab active at a time (ToggleGroup)
 *   <li>Click active tab to collapse panel entirely
 *   <li>Active tab indicator (left border)
 *   <li>Smooth expand/collapse animation
 * </ul>
 */
public class VerticalTabPane extends HBox {

  private static final double TAB_BAR_WIDTH = 44;
  private static final double CONTENT_WIDTH = 280;
  private static final Duration ANIMATION_DURATION = Duration.millis(250);

  private final StackPane contentArea;
  private final VBox topTabs;
  private final VBox bottomTabs;
  private final ToggleGroup toggleGroup;
  private final List<TabEntry> tabs = new ArrayList<>();
  private final IntegerProperty selectedIndex = new SimpleIntegerProperty(-1);

  private boolean collapsed = true; // Start collapsed (matches constructor state)
  private Timeline animation;

  /** Creates a new vertical tab pane. */
  @SuppressWarnings("this-escape")
  public VerticalTabPane() {
    getStyleClass().add("vertical-tab-pane");
    setAlignment(Pos.TOP_RIGHT);

    // Content area (left side, shows selected tab's content)
    contentArea = new StackPane();
    contentArea.getStyleClass().add("vtab-content-area");
    contentArea.setPrefWidth(CONTENT_WIDTH);
    contentArea.setMinWidth(0);
    HBox.setHgrow(contentArea, Priority.NEVER);

    // Top tabs container
    topTabs = new VBox(4);
    topTabs.setAlignment(Pos.TOP_CENTER);

    // Bottom tabs container
    bottomTabs = new VBox(4);
    bottomTabs.setAlignment(Pos.BOTTOM_CENTER);

    // Tab bar using BorderPane for top/bottom layout
    BorderPane tabBarContainer = new BorderPane();
    tabBarContainer.getStyleClass().add("vtab-bar");
    tabBarContainer.setPadding(new Insets(8, 0, 8, 0));
    tabBarContainer.setMinWidth(TAB_BAR_WIDTH);
    tabBarContainer.setMaxWidth(TAB_BAR_WIDTH);
    tabBarContainer.setPrefWidth(TAB_BAR_WIDTH);
    tabBarContainer.setTop(topTabs);
    tabBarContainer.setBottom(bottomTabs);

    toggleGroup = new ToggleGroup();

    getChildren().addAll(contentArea, tabBarContainer);

    // Start collapsed
    contentArea.setVisible(false);
    contentArea.setManaged(false);
    contentArea.setPrefWidth(0);
  }

  /**
   * Adds a tab with an icon and content at the top.
   *
   * @param icon Icon text (emoji or symbol)
   * @param tooltip Tooltip text
   * @param content Content node for this tab
   * @return Index of the added tab
   */
  public int addTab(String icon, String tooltip, Node content) {
    return addTab(icon, tooltip, content, false);
  }

  /**
   * Adds a tab with an icon and content.
   *
   * @param icon Icon text (emoji or symbol)
   * @param tooltip Tooltip text
   * @param content Content node for this tab
   * @param atBottom If true, add to bottom of tab bar
   * @return Index of the added tab
   */
  public int addTab(String icon, String tooltip, Node content, boolean atBottom) {
    int index = tabs.size();

    ToggleButton tabButton = new ToggleButton(icon);
    tabButton.getStyleClass().add("vtab-button");
    tabButton.setToggleGroup(toggleGroup);
    tabButton.setTooltip(new Tooltip(tooltip));

    // Badge label (optional, hidden by default)
    Label badge = new Label();
    badge.getStyleClass().add("vtab-badge");
    badge.setVisible(false);
    badge.setManaged(false);

    // Wrap button and badge in a container
    StackPane tabContainer = new StackPane(tabButton, badge);
    tabContainer.setAlignment(Pos.TOP_RIGHT);
    StackPane.setAlignment(badge, Pos.TOP_RIGHT);
    StackPane.setMargin(badge, new Insets(-4, -4, 0, 0));

    TabEntry entry = new TabEntry(index, tabButton, badge, content, atBottom);
    tabs.add(entry);

    // Handle tab selection
    final int tabIndex = index;
    tabButton.setOnAction(
        e -> {
          if (tabButton.isSelected()) {
            selectTab(tabIndex);
          } else {
            // Clicking active tab collapses the panel
            collapse();
          }
        });

    // Add to appropriate container
    if (atBottom) {
      bottomTabs.getChildren().add(tabContainer);
    } else {
      topTabs.getChildren().add(tabContainer);
    }

    contentArea.getChildren().add(content);
    content.setVisible(false);
    content.setManaged(false);

    return index;
  }

  /**
   * Selects a tab by index.
   *
   * @param index Tab index
   */
  public void selectTab(int index) {
    if (index < 0 || index >= tabs.size()) {
      return;
    }

    // Update selection
    selectedIndex.set(index);

    // Update toggle state
    tabs.get(index).button().setSelected(true);

    // Show selected content, hide others
    for (int i = 0; i < tabs.size(); i++) {
      Node content = tabs.get(i).content();
      boolean selected = (i == index);
      content.setVisible(selected);
      content.setManaged(selected);
    }

    // Expand if collapsed
    if (collapsed) {
      expand();
    }
  }

  /** Collapses the content panel (hides content, shows only tab bar). */
  public void collapse() {
    if (collapsed) {
      return;
    }
    collapsed = true;

    // Deselect all tabs
    toggleGroup.selectToggle(null);
    selectedIndex.set(-1);

    // Animate collapse
    animateWidth(0, false);
  }

  /** Expands the content panel. */
  public void expand() {
    if (!collapsed) {
      return;
    }
    collapsed = false;

    // Animate expand
    animateWidth(CONTENT_WIDTH, true);
  }

  private void animateWidth(double targetWidth, boolean showAfter) {
    if (animation != null && animation.getStatus() == Timeline.Status.RUNNING) {
      animation.stop();
    }

    if (showAfter) {
      contentArea.setVisible(true);
      contentArea.setManaged(true);
    }

    animation =
        new Timeline(
            new KeyFrame(
                ANIMATION_DURATION, new KeyValue(contentArea.prefWidthProperty(), targetWidth)));

    animation.setOnFinished(
        e -> {
          if (!showAfter) {
            contentArea.setVisible(false);
            contentArea.setManaged(false);
          }
        });

    animation.play();
  }

  /**
   * Sets a badge on a tab.
   *
   * @param index Tab index
   * @param text Badge text (null or empty to hide)
   */
  public void setBadge(int index, String text) {
    if (index < 0 || index >= tabs.size()) {
      return;
    }
    Label badge = tabs.get(index).badge();
    if (text == null || text.isEmpty()) {
      badge.setVisible(false);
      badge.setManaged(false);
    } else {
      badge.setText(text);
      badge.setVisible(true);
      badge.setManaged(true);
    }
  }

  /** Returns whether the panel is collapsed. */
  public boolean isCollapsed() {
    return collapsed;
  }

  /** Returns the currently selected tab index (-1 if none). */
  public int getSelectedIndex() {
    return selectedIndex.get();
  }

  /** Returns the selected index property. */
  public IntegerProperty selectedIndexProperty() {
    return selectedIndex;
  }

  /** Internal record for tab data. */
  private record TabEntry(
      int index, ToggleButton button, Label badge, Node content, boolean atBottom) {}
}
