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

import com.integrallis.vectors.studio.core.StudioSession;
import com.integrallis.vectors.studio.core.connection.ConnectionConfig;
import com.integrallis.vectors.studio.core.connection.StudioBackend;
import com.integrallis.vectors.studio.core.connection.StudioBackendFactory;
import com.integrallis.vectors.studio.web.projection.ProjectionJobManager;
import com.integrallis.vectors.studio.web.routing.StudioRouting;
import io.helidon.webserver.WebServer;
import java.net.URI;
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

/** PicoCLI entry point for the Vectors Studio web frontend. */
@Command(
    name = "vectors-studio",
    description = "Vectors Studio web frontend (Helidon SE 4)",
    mixinStandardHelpOptions = true)
public final class StudioServer implements Callable<Integer> {

  private static final Logger LOG = LoggerFactory.getLogger(StudioServer.class);

  @Option(names = "--port", description = "HTTP listen port (default 8288)")
  private int port = StudioConfig.DEFAULT_PORT;

  @Option(
      names = "--connection",
      description = "embedded:/path/to/data  OR  http://host:port",
      required = true)
  private String connection;

  @Option(names = "--token", description = "Bearer token for remote backend (optional)")
  private String token;

  public static void main(String[] args) {
    System.exit(new CommandLine(new StudioServer()).execute(args));
  }

  @Override
  public Integer call() throws InterruptedException {
    StudioBackend backend = StudioBackendFactory.open(parseConnection(connection, token));
    StudioSession session = new StudioSession(backend);
    StudioServerHandle handle = start(new StudioConfig(port, session));
    LOG.info("vectors-studio listening on port {}", handle.port());
    CountDownLatch latch = new CountDownLatch(1);
    Runtime.getRuntime()
        .addShutdownHook(
            new Thread(
                () -> {
                  handle.stop();
                  latch.countDown();
                },
                "studio-shutdown"));
    latch.await();
    return 0;
  }

  /** Programmatic entry: starts a server and returns a handle. Caller closes the handle. */
  public static StudioServerHandle start(StudioConfig config) {
    Objects.requireNonNull(config, "config");
    ProjectionJobManager jobs = new ProjectionJobManager();
    StudioRouting routing = new StudioRouting(config.session(), jobs, config.sidecart());
    WebServer server =
        WebServer.builder().port(config.port()).routing(routing::apply).build().start();
    return new StudioServerHandle(server, config.session(), jobs);
  }

  /** Parses an {@code embedded:/path} or {@code http://host:port} connection string. */
  static ConnectionConfig parseConnection(String spec, String token) {
    Objects.requireNonNull(spec, "connection");
    if (spec.startsWith("embedded:")) {
      return new ConnectionConfig.Embedded(Path.of(spec.substring("embedded:".length())));
    }
    if (spec.startsWith("http://") || spec.startsWith("https://")) {
      return new ConnectionConfig.Remote(URI.create(spec), token, Duration.ofSeconds(10));
    }
    throw new IllegalArgumentException("connection must start with 'embedded:' or 'http(s)://'");
  }
}
