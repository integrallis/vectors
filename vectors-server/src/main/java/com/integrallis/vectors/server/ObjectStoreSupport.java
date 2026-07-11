/*
 * Copyright 2025-2026 Integrallis Software, LLC
 *
 * Licensed under the Functional Source License, Version 1.1, Apache 2.0 Future License
 * (the "License"); you may not use this file except in compliance with the License.
 *
 *     https://fsl.software/FSL-1.1-ALv2.txt
 *
 * Unless required by applicable law or agreed to in writing, software distributed under
 * the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
 * ANY KIND, either express or implied. See the License for the specific language
 * governing permissions and limitations under the License.
 *
 * Change Date: April 25, 2028
 * Change License: Apache License, Version 2.0
 */
package com.integrallis.vectors.server;

import com.integrallis.vectors.storage.backend.S3StorageBackend;
import com.integrallis.vectors.storage.backend.StorageBackend;
import java.net.URI;

/**
 * Builds the object-storage {@link StorageBackend} for the server's collections from a {@link
 * ServerConfig.ObjectStore}, and computes each collection's key prefix.
 */
public final class ObjectStoreSupport {

  private ObjectStoreSupport() {}

  /**
   * Opens a {@link StorageBackend} for the configured object store. A non-null endpoint targets an
   * S3-compatible service (MinIO, Cloudflare R2, …) with static credentials and path-style
   * addressing; a null endpoint targets AWS S3 with the default credential chain.
   */
  public static StorageBackend open(ServerConfig.ObjectStore cfg) {
    if (cfg.endpoint() != null && !cfg.endpoint().isBlank()) {
      return S3StorageBackend.create(
          URI.create(cfg.endpoint()), cfg.bucket(), cfg.region(), cfg.accessKey(), cfg.secretKey());
    }
    return S3StorageBackend.create(cfg.bucket(), cfg.region());
  }

  /** The object-storage key prefix under which collection {@code name}'s generations live. */
  public static String keyPrefixFor(ServerConfig.ObjectStore cfg, String name) {
    String base = cfg.prefix();
    if (base.isEmpty()) {
      return name + "/";
    }
    return (base.endsWith("/") ? base : base + "/") + name + "/";
  }
}
