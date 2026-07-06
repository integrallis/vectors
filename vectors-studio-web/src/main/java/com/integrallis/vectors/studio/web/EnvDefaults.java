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
package com.integrallis.vectors.studio.web;

import io.github.cdimascio.dotenv.Dotenv;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Resolves R2 and D1 credentials for {@link StudioServer} from (in order) the JVM process
 * environment, JVM system properties, and a {@code .env} file at the repository root. Mirrors the
 * lookup order used by the {@code R2CorpusSeeder} so a single gitignored {@code .env} drives both
 * the seeder and the server.
 */
public final class EnvDefaults {

  private final Dotenv dotenv;

  private EnvDefaults(Dotenv dotenv) {
    this.dotenv = dotenv;
  }

  public static EnvDefaults fromRepoRoot() {
    Path dir = resolveDotenvDirectory();
    Dotenv dotenv =
        Dotenv.configure()
            .directory(dir.toAbsolutePath().toString())
            .ignoreIfMissing()
            .ignoreIfMalformed()
            .load();
    return new EnvDefaults(dotenv);
  }

  public String get(String key) {
    String env = System.getenv(key);
    if (!isBlank(env)) return env.trim();
    String prop = System.getProperty(key);
    if (!isBlank(prop)) return prop.trim();
    String dot = dotenv.get(key);
    return isBlank(dot) ? null : dot.trim();
  }

  /**
   * Returns the Cloudflare account id used for D1, falling back to {@code VECTORS_R2_ACCOUNT_ID} so
   * users with a single Cloudflare account need only configure it once.
   */
  String cfAccountId() {
    String d1 = get("VECTORS_CF_ACCOUNT_ID");
    return d1 != null ? d1 : get("VECTORS_R2_ACCOUNT_ID");
  }

  private static boolean isBlank(String s) {
    return s == null || s.isBlank();
  }

  private static Path resolveDotenvDirectory() {
    String prop = System.getProperty("dotenv.directory");
    if (prop != null && !prop.isBlank()) return Paths.get(prop);
    Path cwd = Paths.get("").toAbsolutePath();
    Path p = cwd;
    for (int i = 0; i < 8 && p != null; i++) {
      if (Files.exists(p.resolve(".env.example"))) return p;
      p = p.getParent();
    }
    return cwd;
  }
}
