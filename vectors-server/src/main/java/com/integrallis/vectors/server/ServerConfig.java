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
 *     collections (tests only)
 * @param maxConnections soft cap on concurrent TCP connections; mapped to Helidon's {@code
 *     maxTcpConnections}
 * @param shutdownTimeoutSeconds grace period for inflight requests on SIGTERM; mapped to Helidon's
 *     {@code shutdownGracePeriod}
 */
public record ServerConfig(int port, Path dataDir, int maxConnections, int shutdownTimeoutSeconds) {

  /** Default HTTP port. {@code 8287 == VCTR} on a phone keypad; prime; IANA-unassigned. */
  public static final int DEFAULT_PORT = 8287;

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
    return new ServerConfig(port, dataDir, maxConnections, shutdownTimeoutSeconds);
  }

  /**
   * @return a copy of this config with {@code port} replaced by the supplied value
   */
  public ServerConfig withPortOverride(int newPort) {
    return new ServerConfig(newPort, dataDir, maxConnections, shutdownTimeoutSeconds);
  }

  /**
   * @return {@code true} if this configuration points at on-disk persistence
   */
  public boolean isPersistent() {
    return dataDir != null;
  }

  /**
   * @return the supplied config's {@link #dataDir} or throws if missing
   */
  public Path requireDataDir() {
    return Objects.requireNonNull(dataDir, "dataDir must be configured");
  }
}
