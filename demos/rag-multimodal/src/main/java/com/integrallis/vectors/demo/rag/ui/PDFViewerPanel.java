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

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javafx.embed.swing.SwingFXUtils;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.ScrollPane;
import javafx.scene.image.ImageView;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.StackPane;
import javafx.scene.layout.VBox;
import org.apache.pdfbox.Loader;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.rendering.PDFRenderer;

/**
 * Panel for displaying PDF documents with page navigation.
 *
 * <p>Uses PDFBox to render PDF pages as images for display in JavaFX.
 */
public class PDFViewerPanel extends BorderPane {

  private static final float RENDER_DPI = 100f;

  private PDDocument document;
  private PDFRenderer renderer;
  private int currentPage = 0;
  private int totalPages = 0;

  private final ImageView pageImageView;
  private final Label pageLabel;
  private final Label titleLabel;
  private final Button prevButton;
  private final Button nextButton;
  private final StackPane placeholder;
  private final ScrollPane scrollPane;

  @SuppressWarnings("this-escape")
  public PDFViewerPanel() {
    getStyleClass().add("pdf-viewer-panel");

    // Title bar
    titleLabel = new Label("No PDF loaded");
    titleLabel.getStyleClass().add("pdf-title");

    // Page image view
    pageImageView = new ImageView();
    pageImageView.setPreserveRatio(true);
    pageImageView.setSmooth(true);

    // Wrap in ScrollPane for large pages
    scrollPane = new ScrollPane(pageImageView);
    scrollPane.setFitToWidth(true);
    scrollPane.setPannable(true);
    scrollPane.getStyleClass().add("pdf-scroll-pane");

    // Placeholder when no PDF loaded
    placeholder = new StackPane();
    placeholder.getStyleClass().add("pdf-placeholder");
    Label placeholderLabel = new Label("Upload a PDF to view it here");
    placeholderLabel.getStyleClass().add("placeholder-text");
    placeholder.getChildren().add(placeholderLabel);

    // Navigation controls
    prevButton = new Button("\u25C0");
    prevButton.getStyleClass().add("nav-button");
    prevButton.setOnAction(e -> previousPage());
    prevButton.setDisable(true);

    nextButton = new Button("\u25B6");
    nextButton.getStyleClass().add("nav-button");
    nextButton.setOnAction(e -> nextPage());
    nextButton.setDisable(true);

    pageLabel = new Label("0 / 0");
    pageLabel.getStyleClass().add("page-label");

    HBox navBox = new HBox(10, prevButton, pageLabel, nextButton);
    navBox.setAlignment(Pos.CENTER);
    navBox.setPadding(new Insets(10));
    navBox.getStyleClass().add("nav-box");

    // Header with title
    VBox header = new VBox(5, titleLabel);
    header.setAlignment(Pos.CENTER);
    header.setPadding(new Insets(10));
    header.getStyleClass().add("pdf-header");

    setTop(header);
    setCenter(placeholder);
    setBottom(navBox);
  }

  /**
   * Loads a PDF file for display.
   *
   * @param file PDF file to load
   * @throws IOException if loading fails
   */
  public void loadPDF(File file) throws IOException {
    // Close previous document if open
    if (document != null) {
      document.close();
    }

    document = Loader.loadPDF(file);
    renderer = new PDFRenderer(document);
    totalPages = document.getNumberOfPages();
    currentPage = 0;

    titleLabel.setText(file.getName());

    // Switch from placeholder to scroll pane
    setCenter(scrollPane);

    updateNavigation();
    renderCurrentPage();
  }

  /** Navigates to the previous page. */
  public void previousPage() {
    if (currentPage > 0) {
      currentPage--;
      updateNavigation();
      renderCurrentPage();
    }
  }

  /** Navigates to the next page. */
  public void nextPage() {
    if (currentPage < totalPages - 1) {
      currentPage++;
      updateNavigation();
      renderCurrentPage();
    }
  }

  /**
   * Navigates to a specific page.
   *
   * @param pageNumber Zero-based page number
   */
  public void goToPage(int pageNumber) {
    if (pageNumber >= 0 && pageNumber < totalPages) {
      currentPage = pageNumber;
      updateNavigation();
      renderCurrentPage();
    }
  }

  /**
   * Gets the current page number.
   *
   * @return Zero-based current page number
   */
  public int getCurrentPage() {
    return currentPage;
  }

  /**
   * Gets the total number of pages.
   *
   * @return Total page count
   */
  public int getTotalPages() {
    return totalPages;
  }

  private void updateNavigation() {
    prevButton.setDisable(currentPage <= 0);
    nextButton.setDisable(currentPage >= totalPages - 1);
    pageLabel.setText((currentPage + 1) + " / " + totalPages);
  }

  private void renderCurrentPage() {
    if (renderer == null) return;

    try {
      BufferedImage bufferedImage = renderer.renderImageWithDPI(currentPage, RENDER_DPI);
      pageImageView.setImage(SwingFXUtils.toFXImage(bufferedImage, null));

      // Fit to width of scroll pane
      pageImageView.fitWidthProperty().bind(scrollPane.widthProperty().subtract(20));
    } catch (IOException e) {
      System.err.println("Failed to render page: " + e.getMessage());
    }
  }

  /** Closes the PDF document and releases resources. */
  public void close() {
    if (document != null) {
      try {
        document.close();
      } catch (IOException e) {
        // Ignore
      }
      document = null;
      renderer = null;
    }
  }
}
