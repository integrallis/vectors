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

import java.nio.file.Path;
import java.util.Objects;

/**
 * Runtime configuration for a {@link VectorsServer} instance.
 *
 * @param port HTTP listen port (default {@value #DEFAULT_PORT}, a prime unassigned by IANA; reads
 *     as {@code VCTR} on a phone keypad). Use {@code 0} for an OS-assigned ephemeral port — the
 *     actual port is available via {@link VectorsServer.ServerHandle#port()} after start.
 * @param dataDir base directory for persistent collections; may be {@code null} for in-memory
 *     collections (tests only). When {@link #objectStore} is set this directory is the local cache
 *     in front of the object-storage durable floor.
 * @param maxConnections soft cap on concurrent TCP connections; mapped to Helidon's {@code
 *     maxTcpConnections}
 * @param shutdownTimeoutSeconds grace period for inflight requests on SIGTERM; mapped to Helidon's
 *     {@code shutdownGracePeriod}
 * @param apiKey bearer token required for protected API routes; {@code null} disables
 *     authentication
 * @param objectStore object-storage durable floor (S3 / R2 / GCS / MinIO) for collections; {@code
 *     null} keeps persistence on the local {@link #dataDir} only
 */
public record ServerConfig(
    int port,
    Path dataDir,
    int maxConnections,
    int shutdownTimeoutSeconds,
    String apiKey,
    ObjectStore objectStore) {

  /** Default HTTP port. {@code 8287 == VCTR} on a phone keypad; prime; IANA-unassigned. */
  public static final int DEFAULT_PORT = 8287;

  /**
   * Object-storage durable-floor configuration for the server's collections. An {@code endpoint} of
   * {@code null} targets AWS S3 with the default credential chain; a non-null endpoint (with {@code
   * accessKey}/{@code secretKey}) targets an S3-compatible service — Cloudflare R2, MinIO, or any
   * other. Each collection lives under {@code prefix + name + "/"}.
   *
   * @param endpoint S3-compatible endpoint URI, or {@code null} for AWS S3
   * @param bucket bucket name
   * @param region region (e.g. {@code us-east-1}; {@code auto} for R2)
   * @param accessKey access key, or {@code null} to use the AWS default credential chain
   * @param secretKey secret key, or {@code null} to use the AWS default credential chain
   * @param prefix key prefix shared by all collections (may be empty)
   */
  public record ObjectStore(
      String endpoint,
      String bucket,
      String region,
      String accessKey,
      String secretKey,
      String prefix) {
    public ObjectStore {
      Objects.requireNonNull(bucket, "bucket must not be null");
      Objects.requireNonNull(region, "region must not be null");
      prefix = prefix == null ? "" : prefix;
    }
  }

  public ServerConfig {
    if (port < 0 || port > 65535) {
      throw new IllegalArgumentException("port out of range: " + port);
    }
    if (maxConnections <= 0) {
      throw new IllegalArgumentException("maxConnections must be positive: " + maxConnections);
    }
    if (shutdownTimeoutSeconds < 0) {
      throw new IllegalArgumentException(
          "shutdownTimeoutSeconds must be non-negative: " + shutdownTimeoutSeconds);
    }
    if (apiKey != null) {
      apiKey = apiKey.trim();
      if (apiKey.isEmpty()) {
        throw new IllegalArgumentException("apiKey must not be blank");
      }
    }
    if (objectStore != null && dataDir == null) {
      throw new IllegalArgumentException(
          "dataDir (the local cache) is required when objectStore is configured");
    }
  }

  /** Canonical config without an object store (local-only persistence). */
  public ServerConfig(
      int port, Path dataDir, int maxConnections, int shutdownTimeoutSeconds, String apiKey) {
    this(port, dataDir, maxConnections, shutdownTimeoutSeconds, apiKey, null);
  }

  public ServerConfig(int port, Path dataDir, int maxConnections, int shutdownTimeoutSeconds) {
    this(port, dataDir, maxConnections, shutdownTimeoutSeconds, null, null);
  }

  /**
   * @return a server config suitable for tests (ephemeral port, in-memory, fast shutdown)
   */
  public static ServerConfig forTesting() {
    return new ServerConfig(0, null, 64, 5);
  }

  /**
   * @return a server config with the supplied port and defaults elsewhere
   */
  public static ServerConfig withPort(int port) {
    return new ServerConfig(port, null, 512, 30);
  }

  /**
   * @return a production-ish default: port 8287, no data dir (callers set one), 512 max conns, 30s
   *     shutdown
   */
  public static ServerConfig defaults() {
    return new ServerConfig(DEFAULT_PORT, null, 512, 30);
  }

  /**
   * @return a copy of this config with {@code dataDir} replaced by the supplied value
   */
  public ServerConfig withDataDir(Path dataDir) {
    return new ServerConfig(
        port, dataDir, maxConnections, shutdownTimeoutSeconds, apiKey, objectStore);
  }

  /**
   * @return a copy of this config with {@code port} replaced by the supplied value
   */
  public ServerConfig withPortOverride(int newPort) {
    return new ServerConfig(
        newPort, dataDir, maxConnections, shutdownTimeoutSeconds, apiKey, objectStore);
  }

  /**
   * @return a copy of this config with bearer-token authentication enabled
   */
  public ServerConfig withApiKey(String apiKey) {
    return new ServerConfig(
        port, dataDir, maxConnections, shutdownTimeoutSeconds, apiKey, objectStore);
  }

  /**
   * @return a copy of this config with the supplied object-storage durable floor
   */
  public ServerConfig withObjectStore(ObjectStore objectStore) {
    return new ServerConfig(
        port, dataDir, maxConnections, shutdownTimeoutSeconds, apiKey, objectStore);
  }

  /**
   * @return {@code true} when protected routes require a bearer token
   */
  public boolean authEnabled() {
    return apiKey != null;
  }

  /**
   * @return {@code true} if this configuration points at on-disk persistence
   */
  public boolean isPersistent() {
    return dataDir != null;
  }

  /**
   * @return {@code true} when collections are backed by an object-storage durable floor
   */
  public boolean isObjectStoreBacked() {
    return objectStore != null;
  }

  /**
   * @return the supplied config's {@link #dataDir} or throws if missing
   */
  public Path requireDataDir() {
    return Objects.requireNonNull(dataDir, "dataDir must be configured");
  }
}
