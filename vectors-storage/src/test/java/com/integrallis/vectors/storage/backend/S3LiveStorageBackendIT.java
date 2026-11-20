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
package com.integrallis.vectors.storage.backend;

import com.integrallis.vectors.storage.testkit.StorageBackendContract;
import java.io.IOException;
import java.net.URI;
import java.util.List;
import java.util.UUID;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Tag;

/**
 * Runs the full {@link StorageBackendContract} — put/get, etag reads, conditional-put (CAS), range
 * reads, list, delete — against a <em>real, live</em> S3-compatible object store: AWS S3,
 * Cloudflare R2, or a self-hosted MinIO. This is the on-cloud counterpart to {@link
 * S3StorageBackendIT} (which runs the same contract against LocalStack in a container).
 *
 * <p>Opt-in and CI-safe: it self-skips unless {@code VECTORS_LIVE_S3_BUCKET} is set, so it never
 * runs (or touches the network) in a normal build. To run it:
 *
 * <pre>{@code
 * # AWS S3 (credentials from the standard provider chain / env):
 * VECTORS_LIVE_S3_BUCKET=my-bucket AWS_REGION=us-east-1 ./gradlew :vectors-storage:test \
 *     --tests '*S3LiveStorageBackendIT*'
 *
 * # Cloudflare R2 or MinIO (custom endpoint + explicit keys):
 * VECTORS_LIVE_S3_BUCKET=my-bucket \
 * VECTORS_LIVE_S3_ENDPOINT=https://<acct>.r2.cloudflarestorage.com \
 * AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_REGION=auto \
 *     ./gradlew :vectors-storage:test --tests '*S3LiveStorageBackendIT*'
 * }</pre>
 *
 * <p><strong>Bucket hygiene:</strong> Every key the contract writes is prefixed with {@code
 * storage-contract-it/<run-uuid>/} (override with {@code VECTORS_LIVE_S3_PREFIX}). After all tests
 * finish, an {@code @AfterAll} lists every key under that prefix and deletes them. This still
 * requires a <em>dedicated test bucket</em> — do not run against a bucket that holds production
 * data.
 *
 * <p>The target bucket must already exist. The contract does not delete the bucket itself.
 */
@Tag("live-object-store")
class S3LiveStorageBackendIT implements StorageBackendContract {

  private static final String RUN_UUID = UUID.randomUUID().toString();
  private static final String RUN_PREFIX;

  static {
    String configured = System.getenv("VECTORS_LIVE_S3_PREFIX");
    if (configured != null && !configured.isBlank()) {
      RUN_PREFIX = configured.endsWith("/") ? configured : configured + "/";
    } else {
      RUN_PREFIX = "storage-contract-it/" + RUN_UUID + "/";
    }
  }

  private static volatile StorageBackend sharedBackend;

  @Override
  public String key(String suffix) {
    return RUN_PREFIX + suffix + "-" + UUID.randomUUID();
  }

  @AfterAll
  static void deleteRunPrefix() throws IOException {
    StorageBackend b = sharedBackend;
    if (b == null) {
      return;
    }
    List<String> keys = b.list(RUN_PREFIX);
    for (String key : keys) {
      b.delete(key);
    }
  }

  @Override
  public StorageBackend backend() {
    String bucket = System.getenv("VECTORS_LIVE_S3_BUCKET");
    Assumptions.assumeTrue(
        bucket != null && !bucket.isBlank(),
        "set VECTORS_LIVE_S3_BUCKET (and AWS creds) to run live object-store tests");

    StorageBackend existing = sharedBackend;
    if (existing != null) {
      return existing;
    }
    synchronized (S3LiveStorageBackendIT.class) {
      if (sharedBackend == null) {
        sharedBackend = create(bucket);
      }
      return sharedBackend;
    }
  }

  private static StorageBackend create(String bucket) {
    String region = envOr("AWS_REGION", "us-east-1");
    String endpoint = System.getenv("VECTORS_LIVE_S3_ENDPOINT");
    if (endpoint != null && !endpoint.isBlank()) {
      // R2 / MinIO / any S3-compatible endpoint: explicit endpoint + static credentials.
      return S3StorageBackend.create(
          URI.create(endpoint),
          bucket,
          region,
          System.getenv("AWS_ACCESS_KEY_ID"),
          System.getenv("AWS_SECRET_ACCESS_KEY"));
    }
    // Real AWS S3: credentials from the standard provider chain (env vars, profile, IAM role).
    return S3StorageBackend.create(bucket, region);
  }

  private static String envOr(String name, String fallback) {
    String v = System.getenv(name);
    return (v == null || v.isBlank()) ? fallback : v;
  }
}
