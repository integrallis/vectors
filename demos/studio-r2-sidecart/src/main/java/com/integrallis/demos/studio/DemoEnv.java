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
 * Loads environment values from (in order) the JVM process environment, JVM system properties, and
 * a {@code .env} file in this demo's project directory ({@code demos/studio-r2-sidecart/.env}).
 * Used by the seeder CLI and by the live integration test so users keep R2 + D1 credentials in a
 * single gitignored file scoped to the demo.
 *
 * <p>Recognised keys (none are required at construction time — callers ask for whichever ones they
 * need and decide how to handle absence):
 *
 * <ul>
 *   <li>{@code VECTORS_R2_ACCOUNT_ID} — Cloudflare account id; also used to derive the default S3
 *       endpoint when {@code VECTORS_R2_ENDPOINT} is unset.
 *   <li>{@code VECTORS_R2_BUCKET} — R2 bucket name.
 *   <li>{@code VECTORS_R2_ACCESS_KEY} / {@code VECTORS_R2_SECRET_KEY} — R2 S3 credentials.
 *   <li>{@code VECTORS_R2_ENDPOINT} (optional) — full S3 endpoint URL; defaults to {@code
 *       https://<account-id>.r2.cloudflarestorage.com}.
 *   <li>{@code VECTORS_R2_REGION} (optional) — defaults to {@code auto}.
 *   <li>{@code VECTORS_CF_ACCOUNT_ID} — Cloudflare account id for D1 (falls back to {@code
 *       VECTORS_R2_ACCOUNT_ID} when unset, since both live in the same account in practice).
 *   <li>{@code VECTORS_CF_API_TOKEN} — Cloudflare API token with D1 read/write permission.
 *   <li>{@code VECTORS_D1_DATABASE_ID} — D1 database id from {@code wrangler d1 create}.
 * </ul>
 */
public final class DemoEnv {

  private final Dotenv dotenv;

  private DemoEnv(Dotenv dotenv) {
    this.dotenv = dotenv;
  }

  /**
   * Builds a {@code DemoEnv} that reads {@code .env} from the configured directory. Lookup order:
   * {@code dotenv.directory} system property (set automatically by the {@code runSeed} Gradle task
   * to this demo's project directory) &rarr; first ancestor of CWD containing {@code .env.example}
   * (the demo dir wins because it ships its own {@code .env.example}) &rarr; CWD itself.
   */
  public static DemoEnv load() {
    Path dir = resolveDotenvDirectory();
    Dotenv dotenv =
        Dotenv.configure()
            .directory(dir.toAbsolutePath().toString())
            .ignoreIfMissing()
            .ignoreIfMalformed()
            .load();
    return new DemoEnv(dotenv);
  }

  /**
   * Returns the value for {@code key}, preferring (in order) the process environment, JVM system
   * property, and {@code .env} entry. Returns {@code null} when all three are blank.
   */
  public String get(String key) {
    String env = System.getenv(key);
    if (!isBlank(env)) return env.trim();
    String prop = System.getProperty(key);
    if (!isBlank(prop)) return prop.trim();
    String dot = dotenv.get(key);
    return isBlank(dot) ? null : dot.trim();
  }

  /** Returns {@code get(key)} when present, otherwise {@code fallback}. */
  public String getOrDefault(String key, String fallback) {
    String v = get(key);
    return v == null ? fallback : v;
  }

  /**
   * Returns the configured R2 endpoint, deriving {@code https://<account-id>.r2.cloudflarestorage
   * .com} from {@code VECTORS_R2_ACCOUNT_ID} when {@code VECTORS_R2_ENDPOINT} is unset. Returns
   * {@code null} when neither is provided.
   */
  public String r2Endpoint() {
    String explicit = get("VECTORS_R2_ENDPOINT");
    if (!isBlank(explicit)) return explicit;
    String account = get("VECTORS_R2_ACCOUNT_ID");
    if (isBlank(account)) return null;
    return "https://" + account + ".r2.cloudflarestorage.com";
  }

  /** Returns the configured R2 region, defaulting to {@code auto} (Cloudflare's recommendation). */
  public String r2Region() {
    return getOrDefault("VECTORS_R2_REGION", "auto");
  }

  /**
   * Returns the Cloudflare account id used by D1, falling back to {@code VECTORS_R2_ACCOUNT_ID} so
   * users with one account need only configure it once.
   */
  public String cfAccountId() {
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
