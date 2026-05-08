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
import com.integrallis.vectors.studio.sidecart.SidecartRegistry;
import com.integrallis.vectors.studio.sidecart.SidecartSource;
import com.integrallis.vectors.studio.sidecart.sources.D1SidecartSource;
import com.integrallis.vectors.studio.sidecart.sources.H2SidecartSource;
import com.integrallis.vectors.studio.web.projection.ProjectionJobManager;
import com.integrallis.vectors.studio.web.routing.StudioRouting;
import io.helidon.webserver.WebServer;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.net.URI;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Map;
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
      description =
          "embedded:/path/to/data  OR  http(s)://host:port  OR  r2://bucket/prefix?wal=...&dim=...&endpoint=...",
      required = true)
  private String connection;

  @Option(names = "--token", description = "Bearer token for remote backend (optional)")
  private String token;

  // ─── R2 credentials (only used when --connection is r2://) ───────────────

  @Option(names = "--r2-access-key", description = "R2 access key (or env VECTORS_R2_ACCESS_KEY)")
  private String r2AccessKey;

  @Option(names = "--r2-secret-key", description = "R2 secret key (or env VECTORS_R2_SECRET_KEY)")
  private String r2SecretKey;

  // ─── sidecart flags ──────────────────────────────────────────────────────

  enum SidecartKind {
    NONE,
    H2,
    D1
  }

  @Option(names = "--sidecart", description = "Sidecart kind: NONE (default), H2, or D1")
  private SidecartKind sidecart = SidecartKind.NONE;

  @Option(names = "--sidecart-table", description = "Sidecart table (default: docs)")
  private String sidecartTable = "docs";

  @Option(names = "--sidecart-id-column", description = "ID column (default: doc_id)")
  private String sidecartIdColumn = "doc_id";

  @Option(names = "--sidecart-text-column", description = "Text column (default: content)")
  private String sidecartTextColumn = "content";

  @Option(names = "--sidecart-blob-column", description = "Blob column (optional)")
  private String sidecartBlobColumn;

  @Option(names = "--sidecart-mime-column", description = "Mime column (optional)")
  private String sidecartMimeColumn;

  @Option(names = "--sidecart-h2-url", description = "H2 JDBC URL")
  private String sidecartH2Url;

  @Option(names = "--sidecart-h2-user", description = "H2 user")
  private String sidecartH2User = "sa";

  @Option(names = "--sidecart-h2-password", description = "H2 password")
  private String sidecartH2Password = "";

  @Option(
      names = "--sidecart-d1-account",
      description = "Cloudflare account id (or VECTORS_CF_ACCOUNT_ID)")
  private String sidecartD1Account;

  @Option(
      names = "--sidecart-d1-database",
      description = "D1 database id (or VECTORS_D1_DATABASE_ID)")
  private String sidecartD1Database;

  @Option(
      names = "--sidecart-d1-token",
      description = "Cloudflare API token (or VECTORS_CF_API_TOKEN)")
  private String sidecartD1Token;

  public static void main(String[] args) {
    System.exit(new CommandLine(new StudioServer()).execute(args));
  }

  @Override
  public Integer call() throws InterruptedException {
    EnvDefaults env = EnvDefaults.fromRepoRoot();
    ConnectionConfig cfg = parseConnection(connection, token, env);
    StudioBackend backend = StudioBackendFactory.open(cfg);
    SidecartSource sidecartSource = buildSidecart(env);
    if (sidecartSource != null) {
      attachSidecartIfSupported(backend, sidecartSource);
    }
    String collectionName = collectionNameFor(cfg);
    SidecartRegistry registry =
        sidecartSource == null
            ? SidecartRegistry.empty()
            : SidecartRegistry.of(Map.of(collectionName, sidecartSource));
    StudioSession session = new StudioSession(backend);
    StudioServerHandle handle = start(new StudioConfig(port, session, registry));
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

  /**
   * Parses an {@code embedded:/path}, {@code http(s)://host:port}, or {@code
   * r2://bucket/prefix?...} connection string. The {@code r2://} branch defers to {@code
   * DistributedConnectionParser} on the runtime classpath via reflection so this module retains a
   * clean compile-time boundary.
   */
  ConnectionConfig parseConnection(String spec, String token, EnvDefaults env) {
    Objects.requireNonNull(spec, "connection");
    if (spec.startsWith("embedded:")) {
      return new ConnectionConfig.Embedded(Path.of(spec.substring("embedded:".length())));
    }
    if (spec.startsWith("http://") || spec.startsWith("https://")) {
      return new ConnectionConfig.Remote(URI.create(spec), token, Duration.ofSeconds(10));
    }
    if (spec.startsWith("r2://")) {
      return parseR2Connection(spec, env);
    }
    throw new IllegalArgumentException(
        "connection must start with 'embedded:', 'http(s)://', or 'r2://'");
  }

  private ConnectionConfig parseR2Connection(String spec, EnvDefaults env) {
    String accessKey = firstNonBlank(r2AccessKey, env.get("VECTORS_R2_ACCESS_KEY"));
    String secretKey = firstNonBlank(r2SecretKey, env.get("VECTORS_R2_SECRET_KEY"));
    if (accessKey == null || secretKey == null) {
      throw new IllegalArgumentException(
          "r2:// connection requires --r2-access-key/--r2-secret-key or VECTORS_R2_ACCESS_KEY / VECTORS_R2_SECRET_KEY (env or .env)");
    }
    try {
      Class<?> parser =
          Class.forName("com.integrallis.vectors.studio.distributed.DistributedConnectionParser");
      Method m = parser.getMethod("parse", String.class, String.class, String.class);
      return (ConnectionConfig) m.invoke(null, spec, accessKey, secretKey);
    } catch (ClassNotFoundException e) {
      throw new IllegalStateException(
          "r2:// support requires :vectors-studio-distributed on the runtime classpath", e);
    } catch (InvocationTargetException e) {
      Throwable cause = e.getCause();
      if (cause instanceof RuntimeException re) throw re;
      throw new IllegalArgumentException("invalid r2:// connection: " + spec, cause);
    } catch (ReflectiveOperationException e) {
      throw new IllegalStateException("failed to invoke DistributedConnectionParser", e);
    }
  }

  private SidecartSource buildSidecart(EnvDefaults env) {
    return switch (sidecart) {
      case NONE -> null;
      case H2 -> {
        if (sidecartH2Url == null || sidecartH2Url.isBlank()) {
          throw new IllegalArgumentException("--sidecart=H2 requires --sidecart-h2-url");
        }
        yield new H2SidecartSource(
            sidecartH2Url,
            sidecartH2User,
            sidecartH2Password,
            sidecartTable,
            sidecartIdColumn,
            sidecartTextColumn,
            sidecartBlobColumn,
            sidecartMimeColumn);
      }
      case D1 -> {
        String account = firstNonBlank(sidecartD1Account, env.cfAccountId());
        String database = firstNonBlank(sidecartD1Database, env.get("VECTORS_D1_DATABASE_ID"));
        String tok = firstNonBlank(sidecartD1Token, env.get("VECTORS_CF_API_TOKEN"));
        if (account == null || database == null || tok == null) {
          throw new IllegalArgumentException(
              "--sidecart=D1 requires account/database/token via flags or env/.env (VECTORS_CF_ACCOUNT_ID, VECTORS_D1_DATABASE_ID, VECTORS_CF_API_TOKEN)");
        }
        yield new D1SidecartSource(
            account,
            database,
            tok,
            sidecartTable,
            sidecartIdColumn,
            sidecartTextColumn,
            sidecartBlobColumn,
            sidecartMimeColumn);
      }
    };
  }

  /**
   * If the backend is the distributed one, calls its {@code attachSidecart} method via reflection
   * so {@code vectors-studio-web} keeps a clean compile-time dep boundary.
   */
  private static void attachSidecartIfSupported(StudioBackend backend, SidecartSource source) {
    try {
      Method m = backend.getClass().getMethod("attachSidecart", SidecartSource.class);
      m.invoke(backend, source);
    } catch (NoSuchMethodException ignored) {
      // Backend doesn't support attachSidecart — registry-only wiring is sufficient.
    } catch (ReflectiveOperationException e) {
      throw new IllegalStateException("failed to attach sidecart to backend", e);
    }
  }

  /** Best-effort collection name for registry binding. Must match what the backend reports. */
  private static String collectionNameFor(ConnectionConfig cfg) {
    try {
      Method m = cfg.getClass().getMethod("collectionName");
      Object v = m.invoke(cfg);
      if (v instanceof String s && !s.isBlank()) return s;
    } catch (NoSuchMethodException ignored) {
      // Fall through to default.
    } catch (ReflectiveOperationException e) {
      throw new IllegalStateException("failed to read collectionName from " + cfg, e);
    }
    return "docs";
  }

  private static String firstNonBlank(String a, String b) {
    if (a != null && !a.isBlank()) return a;
    if (b != null && !b.isBlank()) return b;
    return null;
  }
}
