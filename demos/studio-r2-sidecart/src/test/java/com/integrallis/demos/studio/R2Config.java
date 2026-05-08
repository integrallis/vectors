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
package com.integrallis.demos.studio;

import io.github.cdimascio.dotenv.Dotenv;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Cloudflare R2 configuration loaded from {@code demos/studio-r2-sidecart/.env}. Returns {@code
 * null} from {@link #fromEnv()} when any required key is blank, so the IT self-skips silently when
 * credentials are absent.
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

  /** Returns a populated {@code R2Config} or {@code null} when any required key is missing. */
  static R2Config fromEnv() {
    Path dotenvDir = findDotenvDir();
    Dotenv dotenv =
        Dotenv.configure()
            .directory(dotenvDir.toAbsolutePath().toString())
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

  /**
   * Walks up from the current working directory looking for the first ancestor that contains a
   * {@code .env.example}. Returns CWD as a last resort. Because this demo ships its own {@code
   * .env.example} alongside its sources, the walk-up resolves to the demo's project directory when
   * the test is invoked via {@code :demos:studio-r2-sidecart:test}.
   */
  private static Path findDotenvDir() {
    Path cwd = Paths.get("").toAbsolutePath();
    Path p = cwd;
    for (int i = 0; i < 8 && p != null; i++) {
      if (Files.exists(p.resolve(".env.example"))) return p;
      p = p.getParent();
    }
    return cwd;
  }
}
