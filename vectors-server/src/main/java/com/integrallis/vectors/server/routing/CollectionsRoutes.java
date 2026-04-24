package com.integrallis.vectors.server.routing;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.integrallis.vectors.db.VectorCollection;
import com.integrallis.vectors.server.CollectionRegistry;
import com.integrallis.vectors.server.ObjectMapperHolder;
import com.integrallis.vectors.server.ProblemDetails;
import com.integrallis.vectors.server.ServerConfig;
import com.integrallis.vectors.server.dto.CollectionInfo;
import com.integrallis.vectors.server.dto.CreateCollectionRequest;
import com.integrallis.vectors.server.dto.ListCollectionsResponse;
import io.helidon.http.HeaderNames;
import io.helidon.http.Status;
import io.helidon.webserver.http.HttpRules;
import io.helidon.webserver.http.HttpService;
import io.helidon.webserver.http.ServerRequest;
import io.helidon.webserver.http.ServerResponse;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.stream.Stream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * HTTP routes for collection lifecycle: create, list, describe, drop.
 *
 * <p>Phase 2 wires an in-memory / mmap-backed {@link VectorCollection} per collection; Phase 3
 * reopens persisted collections under {@link ServerConfig#dataDir()} on startup.
 *
 * <p>All error responses use RFC 7807 {@code application/problem+json} bodies.
 */
public final class CollectionsRoutes implements HttpService {

  private static final Logger LOG = LoggerFactory.getLogger(CollectionsRoutes.class);
  private static final String JSON = "application/json";
  private static final String PROBLEM = "application/problem+json";

  private final CollectionRegistry registry;
  private final ServerConfig config;
  private final ObjectMapper mapper;

  /**
   * @param registry shared registry of open collections
   * @param config server configuration (data directory for persistent collections)
   */
  public CollectionsRoutes(CollectionRegistry registry, ServerConfig config) {
    this.registry = Objects.requireNonNull(registry, "registry");
    this.config = Objects.requireNonNull(config, "config");
    this.mapper = ObjectMapperHolder.shared();
  }

  @Override
  public void routing(HttpRules rules) {
    rules
        .post("/v1/collections", this::create)
        .get("/v1/collections", this::list)
        .get("/v1/collections/{name}", this::describe)
        .delete("/v1/collections/{name}", this::drop);
  }

  private void create(ServerRequest req, ServerResponse res) {
    CreateCollectionRequest body;
    try {
      body = mapper.readValue(req.content().inputStream(), CreateCollectionRequest.class);
    } catch (Exception e) {
      sendProblem(res, Status.BAD_REQUEST_400, "malformed request body", e.getMessage(), req);
      return;
    }
    String validationError = body.validate();
    if (validationError != null) {
      sendProblem(res, Status.BAD_REQUEST_400, "invalid create request", validationError, req);
      return;
    }
    if (registry.get(body.name()).isPresent()) {
      sendProblem(res, Status.CONFLICT_409, "collection already exists", body.name(), req);
      return;
    }
    try {
      Path storageRoot = storageRootFor(body.name());
      VectorCollection created =
          registry.create(body.name(), name -> body.toCollection(storageRoot));
      Instant ts =
          registry
              .createdAt(body.name())
              .orElseThrow(() -> new IllegalStateException("missing createdAt"));
      sendJson(res, Status.CREATED_201, CollectionInfo.of(body.name(), created, ts));
    } catch (IllegalStateException dup) {
      sendProblem(res, Status.CONFLICT_409, "collection already exists", body.name(), req);
    } catch (IllegalArgumentException | UnsupportedOperationException e) {
      LOG.debug("rejected create for '{}': {}", body.name(), e.getMessage());
      sendProblem(res, Status.BAD_REQUEST_400, "invalid collection spec", e.getMessage(), req);
    }
  }

  private void list(ServerRequest req, ServerResponse res) {
    List<CollectionInfo> infos = new ArrayList<>(registry.size());
    for (String name : registry.names()) {
      Optional<VectorCollection> col = registry.get(name);
      Optional<Instant> ts = registry.createdAt(name);
      if (col.isPresent() && ts.isPresent()) {
        infos.add(CollectionInfo.of(name, col.get(), ts.get()));
      }
    }
    sendJson(res, Status.OK_200, new ListCollectionsResponse(List.copyOf(infos)));
  }

  private void describe(ServerRequest req, ServerResponse res) {
    String name = req.path().pathParameters().get("name");
    Optional<VectorCollection> col = registry.get(name);
    Optional<Instant> ts = registry.createdAt(name);
    if (col.isEmpty() || ts.isEmpty()) {
      sendProblem(res, Status.NOT_FOUND_404, "collection not found", name, req);
      return;
    }
    sendJson(res, Status.OK_200, CollectionInfo.of(name, col.get(), ts.get()));
  }

  private void drop(ServerRequest req, ServerResponse res) {
    String name = req.path().pathParameters().get("name");
    boolean removed = registry.drop(name);
    if (!removed) {
      sendProblem(res, Status.NOT_FOUND_404, "collection not found", name, req);
      return;
    }
    if (config.isPersistent()) {
      Path dir = config.dataDir().resolve(name);
      try {
        deleteRecursively(dir);
      } catch (IOException e) {
        LOG.warn("failed to delete storage dir for '{}': {}", name, e.getMessage());
      }
    }
    res.status(Status.NO_CONTENT_204).send();
  }

  private Path storageRootFor(String name) {
    if (!config.isPersistent()) {
      return null;
    }
    Path root = config.dataDir().resolve(name).toAbsolutePath();
    try {
      Files.createDirectories(root);
    } catch (IOException e) {
      throw new IllegalStateException("failed to create storage dir for '" + name + "'", e);
    }
    return root;
  }

  private static void deleteRecursively(Path dir) throws IOException {
    if (!Files.exists(dir)) {
      return;
    }
    try (Stream<Path> walk = Files.walk(dir)) {
      walk.sorted(Comparator.reverseOrder())
          .forEach(
              p -> {
                try {
                  Files.deleteIfExists(p);
                } catch (IOException e) {
                  throw new RuntimeException(e);
                }
              });
    } catch (RuntimeException wrapped) {
      if (wrapped.getCause() instanceof IOException ioe) {
        throw ioe;
      }
      throw wrapped;
    }
  }

  private void sendJson(ServerResponse res, Status status, Object body) {
    try {
      String json = mapper.writeValueAsString(body);
      res.headers().set(HeaderNames.CONTENT_TYPE, JSON);
      res.status(status).send(json);
    } catch (Exception e) {
      LOG.error("failed to serialize response body", e);
      res.status(Status.INTERNAL_SERVER_ERROR_500).send();
    }
  }

  private void sendProblem(
      ServerResponse res, Status status, String title, String detail, ServerRequest req) {
    ProblemDetails pd = ProblemDetails.of(status.code(), title, detail, req.path().path());
    try {
      res.headers().set(HeaderNames.CONTENT_TYPE, PROBLEM);
      res.status(status).send(mapper.writeValueAsString(pd));
    } catch (Exception e) {
      LOG.error("failed to serialize problem body", e);
      res.status(Status.INTERNAL_SERVER_ERROR_500).send();
    }
  }
}
