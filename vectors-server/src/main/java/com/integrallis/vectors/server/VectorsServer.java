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

import com.integrallis.vectors.server.routing.ApiRouting;
import io.helidon.webserver.WebServer;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Objects;
import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

/**
 * Main entry point for the java-vectors HTTP server.
 *
 * <p>Typical use:
 *
 * <pre>
 *   java --add-modules jdk.incubator.vector -jar vectors-server.jar \
 *        --port 8287 --data-dir /var/lib/java-vectors
 * </pre>
 *
 * <p>Programmatic use ({@link #start(ServerConfig)}) returns a running {@link WebServer} which the
 * caller is responsible for {@link WebServer#stop() stopping}. This is the form integration tests
 * prefer.
 *
 * <p>The default port {@code 8287} maps to {@code VCTR} on a phone keypad (V=8, C=2, T=8, R=7) and
 * is a prime unassigned by IANA — chosen in the style of Redis' {@code 6379 = MERZ}.
 */
@Command(
    name = "vectors-server",
    description = "java-vectors HTTP server (Helidon SE 4 / Nima)",
    mixinStandardHelpOptions = true)
public final class VectorsServer implements Callable<Integer> {

  private static final Logger LOG = LoggerFactory.getLogger(VectorsServer.class);

  @Option(names = "--port", description = "HTTP listen port (default 8287 / VCTR)")
  private int port = ServerConfig.DEFAULT_PORT;

  @Option(names = "--data-dir", description = "Base directory for persistent collections")
  private Path dataDir;

  @Option(names = "--max-connections", description = "Soft cap on concurrent connections")
  private int maxConnections = 512;

  @Option(
      names = "--shutdown-timeout-seconds",
      description = "Grace period for inflight requests on shutdown")
  private int shutdownTimeoutSeconds = 30;

  public static void main(String[] args) {
    int exit = new CommandLine(new VectorsServer()).execute(args);
    System.exit(exit);
  }

  @Override
  public Integer call() throws InterruptedException {
    ServerConfig config = new ServerConfig(port, dataDir, maxConnections, shutdownTimeoutSeconds);
    ServerHandle handle = start(config);
    CountDownLatch shutdown = new CountDownLatch(1);
    Runtime.getRuntime()
        .addShutdownHook(
            new Thread(
                () -> {
                  handle.stop();
                  shutdown.countDown();
                },
                "vectors-server-shutdown"));
    LOG.info("vectors-server listening on port {}", handle.port());
    shutdown.await();
    return 0;
  }

  /**
   * Starts the HTTP server with the supplied configuration. Caller closes the returned handle.
   *
   * @param config server configuration
   * @return a handle exposing the running {@link WebServer} and its owned {@link
   *     CollectionRegistry}
   */
  public static ServerHandle start(ServerConfig config) {
    Objects.requireNonNull(config, "config");
    CollectionRegistry registry = new CollectionRegistry();
    registry.setDataDir(config.dataDir());
    try {
      if (config.isPersistent()) {
        int reopened = CollectionDiscovery.discoverAndOpen(registry, config.dataDir());
        if (reopened > 0) {
          LOG.info("reopened {} collection(s) from {}", reopened, config.dataDir());
        }
      }
      WebServer server =
          WebServer.builder()
              .port(config.port())
              .maxTcpConnections(config.maxConnections())
              .shutdownGracePeriod(Duration.ofSeconds(config.shutdownTimeoutSeconds()))
              .routing(builder -> new ApiRouting(registry, config).apply(builder))
              .build()
              .start();
      return new ServerHandle(server, registry);
    } catch (RuntimeException e) {
      registry.close();
      throw e;
    }
  }

  /** Handle returned by {@link #start(ServerConfig)}; call {@link #close()} to stop the server. */
  public static final class ServerHandle implements AutoCloseable {
    private final WebServer server;
    private final CollectionRegistry registry;

    ServerHandle(WebServer server, CollectionRegistry registry) {
      this.server = server;
      this.registry = registry;
    }

    /**
     * @return the running web server
     */
    public WebServer webServer() {
      return server;
    }

    /**
     * @return the registry owned by this server
     */
    public CollectionRegistry registry() {
      return registry;
    }

    /**
     * @return the listen port, resolved after {@link WebServer#start() start} (useful for
     *     ephemeral-port test setups)
     */
    public int port() {
      return server.port();
    }

    /** Stops the server and releases all collection resources. */
    public void stop() {
      try {
        server.stop();
      } finally {
        registry.close();
      }
    }

    @Override
    public void close() {
      stop();
    }
  }
}
