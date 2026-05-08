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
package com.integrallis.vectors.ivf;

import io.github.cdimascio.dotenv.Dotenv;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Cloudflare R2 configuration loaded from a {@code .env} file at the repo root.
 *
 * <p>Keys (all required for the IT to be considered configured):
 *
 * <ul>
 *   <li>{@code VECTORS_R2_ACCOUNT_ID} — used to derive the default S3 endpoint.
 *   <li>{@code VECTORS_R2_BUCKET} — the test bucket.
 *   <li>{@code VECTORS_R2_ACCESS_KEY} — R2 access key id.
 *   <li>{@code VECTORS_R2_SECRET_KEY} — R2 secret access key.
 * </ul>
 *
 * <p>Optional overrides:
 *
 * <ul>
 *   <li>{@code VECTORS_R2_ENDPOINT} — full endpoint URL; defaults to {@code
 *       https://<account-id>.r2.cloudflarestorage.com}.
 *   <li>{@code VECTORS_R2_REGION} — defaults to {@code auto}.
 * </ul>
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

  /**
   * Returns a populated {@code R2Config} or {@code null} if any of the four required keys is
   * missing or blank. Walks upward from the current working directory looking for a {@code .env}
   * file so the IT works regardless of which subproject Gradle is invoked from.
   */
  static R2Config fromEnv() {
    Path repoRoot = findRepoRoot();
    Dotenv dotenv =
        Dotenv.configure()
            .directory(repoRoot.toAbsolutePath().toString())
            .ignoreIfMissing()
            .ignoreIfMalformed()
            .load();
    String accountId = first(dotenv, "VECTORS_R2_ACCOUNT_ID");
    String bucket = first(dotenv, "VECTORS_R2_BUCKET");
    String accessKey = first(dotenv, "VECTORS_R2_ACCESS_KEY");
    String secretKey = first(dotenv, "VECTORS_R2_SECRET_KEY");
    if (isBlank(bucket) || isBlank(accessKey) || isBlank(secretKey)) {
      return null;
    }
    String endpoint = first(dotenv, "VECTORS_R2_ENDPOINT");
    if (isBlank(endpoint)) {
      if (isBlank(accountId)) return null;
      endpoint = "https://" + accountId + ".r2.cloudflarestorage.com";
    }
    String region = first(dotenv, "VECTORS_R2_REGION");
    if (isBlank(region)) region = "auto";
    return new R2Config(accountId, bucket, accessKey, secretKey, endpoint, region);
  }

  /** Process env > dotenv lookup. Lets a CI runner override .env without touching the file. */
  private static String first(Dotenv dotenv, String key) {
    String env = System.getenv(key);
    if (!isBlank(env)) return env.trim();
    String prop = System.getProperty(key);
    if (!isBlank(prop)) return prop.trim();
    return dotenv.get(key);
  }

  private static boolean isBlank(String s) {
    return s == null || s.isBlank();
  }

  /** Walks up from CWD until it finds a directory containing {@code .env.example}. */
  private static Path findRepoRoot() {
    Path cwd = Paths.get("").toAbsolutePath();
    Path p = cwd;
    for (int i = 0; i < 8 && p != null; i++) {
      if (Files.exists(p.resolve(".env.example"))) return p;
      p = p.getParent();
    }
    return cwd;
  }
}
