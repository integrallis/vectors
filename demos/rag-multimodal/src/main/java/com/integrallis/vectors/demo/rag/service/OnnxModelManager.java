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

import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Manages ONNX model file downloads and caching.
 *
 * <p>Downloads the DocLayout-YOLO DocStructBench model (75 MB) from HuggingFace. Models are cached
 * in {@code ~/.java-vectors/models/} so they only need to be downloaded once. If the file already
 * exists and is non-empty, it is reused without re-downloading.
 */
public final class OnnxModelManager {

  private static final Logger log = LoggerFactory.getLogger(OnnxModelManager.class);

  private static final String MODEL_URL =
      "https://huggingface.co/wybxc/"
          + "DocLayout-YOLO-DocStructBench-onnx/resolve/main/"
          + "doclayout_yolo_docstructbench_imgsz1024.onnx";

  private static final Path CACHE_DIR =
      Path.of(System.getProperty("user.home"), ".java-vectors", "models");

  private OnnxModelManager() {}

  /**
   * Returns the path to a cached model file, downloading it from HuggingFace if not present.
   *
   * @param modelName file name for the cached model (e.g.
   *     "doclayout_yolo_docstructbench_imgsz1024.onnx")
   * @return path to the cached model file
   * @throws IOException if the download fails or the cache directory cannot be created
   */
  public static Path ensureModel(String modelName) throws IOException {
    Path modelPath = CACHE_DIR.resolve(modelName);

    if (Files.exists(modelPath) && Files.size(modelPath) > 0) {
      log.info("ONNX model already cached: {}", modelPath);
      return modelPath;
    }

    Files.createDirectories(CACHE_DIR);
    log.info("Downloading ONNX model to {} ...", modelPath);

    Path tmpFile = modelPath.resolveSibling(modelName + ".tmp");
    try {
      downloadFile(MODEL_URL, tmpFile);
      Files.move(tmpFile, modelPath, StandardCopyOption.REPLACE_EXISTING);
      log.info("ONNX model downloaded: {} ({} MB)", modelPath, Files.size(modelPath) / 1_000_000);
    } catch (IOException e) {
      // Clean up partial download
      Files.deleteIfExists(tmpFile);
      throw new IOException("Failed to download ONNX model from " + MODEL_URL + ": " + e, e);
    }

    return modelPath;
  }

  private static void downloadFile(String url, Path destination) throws IOException {
    try (HttpClient client =
        HttpClient.newBuilder().followRedirects(HttpClient.Redirect.NORMAL).build()) {
      HttpRequest request = HttpRequest.newBuilder().uri(URI.create(url)).GET().build();
      HttpResponse<InputStream> response =
          client.send(request, HttpResponse.BodyHandlers.ofInputStream());

      if (response.statusCode() != 200) {
        throw new IOException("HTTP " + response.statusCode() + " downloading " + url);
      }

      long contentLength = response.headers().firstValueAsLong("Content-Length").orElse(-1);
      log.info(
          "Download started ({})",
          contentLength > 0 ? (contentLength / 1_000_000) + " MB" : "unknown size");

      try (InputStream body = response.body()) {
        Files.copy(body, destination, StandardCopyOption.REPLACE_EXISTING);
      }
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new IOException("Download interrupted", e);
    }
  }
}
