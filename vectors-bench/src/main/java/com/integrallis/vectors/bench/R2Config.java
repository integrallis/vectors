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
package com.integrallis.vectors.bench;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

/**
 * Cloudflare R2 (S3-compatible) configuration loaded from the repo-root {@code .env} (P1.8).
 * Mirrors the proven loader used by the R2 integration tests so the benchmark reads the same
 * credentials.
 *
 * <p>Required keys (all must be non-blank for {@link #fromEnv()} to return non-null): {@code
 * VECTORS_R2_BUCKET}, {@code VECTORS_R2_ACCESS_KEY}, {@code VECTORS_R2_SECRET_KEY}, and {@code
 * VECTORS_R2_ACCOUNT_ID} (used to derive the endpoint when {@code VECTORS_R2_ENDPOINT} is unset).
 * Optional: {@code VECTORS_R2_ENDPOINT}, {@code VECTORS_R2_REGION} (default {@code auto}).
 *
 * <p>Lookup precedence per key: process environment &gt; JVM system property &gt; {@code .env}.
 */
final class R2Config {

  final String accountId;
  final String bucket;
  final String accessKey;
  final String secretKey;
  final String endpoint;
  final String region;

  private R2Config(
      String accountId,
      String bucket,
      String accessKey,
      String secretKey,
      String endpoint,
      String region) {
    this.accountId = accountId;
    this.bucket = bucket;
    this.accessKey = accessKey;
    this.secretKey = secretKey;
    this.endpoint = endpoint;
    this.region = region;
  }

  /** Returns a populated config, or {@code null} when any required R2 key is missing/blank. */
  static R2Config fromEnv() {
    Map<String, String> dotenv = loadDotenv(findRepoRoot());
    String accountId = first(dotenv, "VECTORS_R2_ACCOUNT_ID");
    String bucket = first(dotenv, "VECTORS_R2_BUCKET");
    String accessKey = first(dotenv, "VECTORS_R2_ACCESS_KEY");
    String secretKey = first(dotenv, "VECTORS_R2_SECRET_KEY");
    if (isBlank(bucket) || isBlank(accessKey) || isBlank(secretKey)) {
      return null;
    }
    String endpoint = first(dotenv, "VECTORS_R2_ENDPOINT");
    if (isBlank(endpoint)) {
      if (isBlank(accountId)) {
        return null;
      }
      endpoint = "https://" + accountId + ".r2.cloudflarestorage.com";
    }
    String region = first(dotenv, "VECTORS_R2_REGION");
    if (isBlank(region)) {
      region = "auto";
    }
    return new R2Config(accountId, bucket, accessKey, secretKey, endpoint, region);
  }

  private static String first(Map<String, String> dotenv, String key) {
    String env = System.getenv(key);
    if (!isBlank(env)) {
      return env.trim();
    }
    String prop = System.getProperty(key);
    if (!isBlank(prop)) {
      return prop.trim();
    }
    String v = dotenv.get(key);
    return isBlank(v) ? null : v.trim();
  }

  /**
   * Minimal {@code .env} reader (KEY=value, {@code #} comments, optional surrounding quotes). Kept
   * dependency-free so the bench main classpath does not pull dotenv-java into vectors-optimizer /
   * vectors-studio-web (which depend on vectors-bench).
   */
  private static Map<String, String> loadDotenv(Path root) {
    Map<String, String> m = new HashMap<>();
    Path env = root.resolve(".env");
    if (!Files.isReadable(env)) {
      return m;
    }
    try {
      for (String line : Files.readAllLines(env)) {
        String s = line.strip();
        if (s.isEmpty() || s.startsWith("#")) {
          continue;
        }
        int eq = s.indexOf('=');
        if (eq <= 0) {
          continue;
        }
        String key = s.substring(0, eq).trim();
        String val = s.substring(eq + 1).trim();
        if (val.length() >= 2
            && ((val.startsWith("\"") && val.endsWith("\""))
                || (val.startsWith("'") && val.endsWith("'")))) {
          val = val.substring(1, val.length() - 1);
        }
        m.put(key, val);
      }
    } catch (IOException ignored) {
      // treat an unreadable .env as "not configured"
    }
    return m;
  }

  private static boolean isBlank(String s) {
    return s == null || s.isBlank();
  }

  /**
   * Walks up from CWD (max 8 levels) until it finds a directory containing {@code .env.example}.
   */
  private static Path findRepoRoot() {
    Path cwd = Paths.get("").toAbsolutePath();
    Path p = cwd;
    for (int i = 0; i < 8 && p != null; i++) {
      if (Files.exists(p.resolve(".env.example"))) {
        return p;
      }
      p = p.getParent();
    }
    return cwd;
  }
}
