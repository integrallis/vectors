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
package com.integrallis.vectors.studio.distributed;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.ivf.TierPolicy;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Parses {@code r2://<bucket>/<prefix>?wal=...&dim=...&metric=...&name=...&endpoint=...&region=...}
 * connection strings into a {@link DistributedConnectionConfig}. Credentials (access key / secret)
 * are <i>not</i> embedded in the URL — they are passed in separately, typically sourced from
 * environment variables or a {@code .env} file by the caller.
 */
public final class DistributedConnectionParser {

  private DistributedConnectionParser() {}

  /**
   * Parses the URL portion of an {@code r2://} connection string. Required query parameters: {@code
   * wal} (WAL directory), {@code dim} (vector dimension), {@code endpoint} (R2 S3 endpoint).
   * Optional: {@code metric} (default {@code COSINE}), {@code region} (default {@code auto}),
   * {@code name} (default: bucket/prefix), {@code t1} / {@code t2} (TierPolicy thresholds).
   */
  public static DistributedConnectionConfig parse(String url, String accessKey, String secretKey) {
    if (url == null || !url.startsWith("r2://")) {
      throw new IllegalArgumentException("expected r2:// url, got: " + url);
    }
    String rest = url.substring("r2://".length());
    int q = rest.indexOf('?');
    String pathPart = q < 0 ? rest : rest.substring(0, q);
    String queryPart = q < 0 ? "" : rest.substring(q + 1);
    int slash = pathPart.indexOf('/');
    String bucket;
    String prefix;
    if (slash < 0) {
      bucket = pathPart;
      prefix = "";
    } else {
      bucket = pathPart.substring(0, slash);
      prefix = pathPart.substring(slash + 1);
    }
    if (bucket.isBlank()) {
      throw new IllegalArgumentException("r2:// url is missing a bucket: " + url);
    }
    Map<String, String> params = parseQuery(queryPart);
    String walRaw = required(params, "wal", url);
    String dimRaw = required(params, "dim", url);
    String endpoint = required(params, "endpoint", url);
    String region = params.getOrDefault("region", "auto");
    String name = params.getOrDefault("name", deriveName(bucket, prefix));
    SimilarityFunction metric = SimilarityFunction.valueOf(params.getOrDefault("metric", "COSINE"));
    int dim;
    try {
      dim = Integer.parseInt(dimRaw);
    } catch (NumberFormatException e) {
      throw new IllegalArgumentException("dim must be an integer: " + dimRaw, e);
    }
    if (dim <= 0) throw new IllegalArgumentException("dim must be positive: " + dim);
    int t1 = parseInt(params, "t1", 5);
    int t2 = parseInt(params, "t2", 2);
    return new DistributedConnectionConfig(
        name,
        endpoint,
        region,
        bucket,
        prefix,
        accessKey,
        secretKey,
        Path.of(walRaw),
        dim,
        metric,
        new TierPolicy(t1, t2));
  }

  private static Map<String, String> parseQuery(String query) {
    Map<String, String> out = new LinkedHashMap<>();
    if (query.isEmpty()) return out;
    for (String pair : query.split("&")) {
      if (pair.isEmpty()) continue;
      int eq = pair.indexOf('=');
      String k = eq < 0 ? pair : pair.substring(0, eq);
      String v = eq < 0 ? "" : pair.substring(eq + 1);
      out.put(
          URLDecoder.decode(k, StandardCharsets.UTF_8),
          URLDecoder.decode(v, StandardCharsets.UTF_8));
    }
    return out;
  }

  private static String required(Map<String, String> params, String key, String url) {
    String v = params.get(key);
    if (v == null || v.isBlank()) {
      throw new IllegalArgumentException("r2:// url is missing required '" + key + "': " + url);
    }
    return v;
  }

  private static int parseInt(Map<String, String> params, String key, int fallback) {
    String v = params.get(key);
    if (v == null || v.isBlank()) return fallback;
    try {
      return Integer.parseInt(v);
    } catch (NumberFormatException e) {
      throw new IllegalArgumentException(key + " must be an integer: " + v, e);
    }
  }

  private static String deriveName(String bucket, String prefix) {
    if (prefix == null || prefix.isBlank()) return bucket;
    String trimmed = prefix.endsWith("/") ? prefix.substring(0, prefix.length() - 1) : prefix;
    return bucket + "-" + trimmed.replace('/', '-');
  }
}
