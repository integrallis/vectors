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
package com.integrallis.vectors.studio.sidecart.sources;

import com.integrallis.vectors.studio.sidecart.SidecartRecord;
import com.integrallis.vectors.studio.sidecart.SidecartSource;
import com.integrallis.vectors.studio.sidecart.SidecartSourceException;
import java.io.IOException;
import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Generic HTTP {@link SidecartSource}. Builds a request URL by substituting {@code {id}} into a
 * configured template — covering presigned S3/GCS URLs, Cloudflare R2, internal blob services, and
 * any plain HTTP file server. Native AWS S3 SDK auth is intentionally <i>not</i> taken as a
 * dependency by this module; deployments that need IAM-signed requests can either pre-sign URLs
 * upstream or implement {@code SidecartSource} directly with the AWS SDK.
 */
public final class HttpSidecartSource implements SidecartSource {

  private final HttpClient http;
  private final String urlTemplate;
  private final boolean textMode;
  private final String defaultMime;
  private final Map<String, String> headers;

  /**
   * @param urlTemplate URL template, must contain a single {@code {id}} placeholder
   * @param textMode true → return body as UTF-8 text; false → return raw bytes as a blob
   * @param defaultMime MIME type to attach when the response carries no {@code Content-Type}
   * @param headers extra request headers (e.g. {@code Authorization}); may be empty / null
   */
  public HttpSidecartSource(
      String urlTemplate, boolean textMode, String defaultMime, Map<String, String> headers) {
    this.urlTemplate = Objects.requireNonNull(urlTemplate, "urlTemplate");
    if (!urlTemplate.contains("{id}")) {
      throw new IllegalArgumentException("urlTemplate must contain '{id}': " + urlTemplate);
    }
    this.textMode = textMode;
    this.defaultMime = defaultMime;
    this.headers = headers == null ? Map.of() : Map.copyOf(headers);
    this.http = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build();
  }

  @Override
  public Optional<SidecartRecord> get(String id) {
    if (id == null || id.isEmpty()) return Optional.empty();
    String url = urlTemplate.replace("{id}", URLEncoder.encode(id, StandardCharsets.UTF_8));
    HttpRequest.Builder rb = HttpRequest.newBuilder(URI.create(url)).GET();
    headers.forEach(rb::header);
    HttpRequest req = rb.build();
    try {
      HttpResponse<byte[]> res = http.send(req, HttpResponse.BodyHandlers.ofByteArray());
      int code = res.statusCode();
      if (code == 404) return Optional.empty();
      if (code / 100 != 2) {
        throw new SidecartSourceException("HTTP " + code + " from sidecart URL " + url);
      }
      String mime = res.headers().firstValue("Content-Type").orElse(defaultMime);
      byte[] bytes = res.body();
      if (textMode) {
        return Optional.of(SidecartRecord.ofText(new String(bytes, StandardCharsets.UTF_8), mime));
      }
      return Optional.of(SidecartRecord.ofBlob(bytes, mime));
    } catch (IOException e) {
      throw new SidecartSourceException("HTTP fetch failed for id=" + id, e);
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new SidecartSourceException("HTTP fetch interrupted for id=" + id, e);
    }
  }
}
