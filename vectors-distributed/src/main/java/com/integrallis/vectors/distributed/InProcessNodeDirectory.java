package com.integrallis.vectors.distributed;

import com.integrallis.vectors.db.VectorCollection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * In-process {@link NodeDirectory} backed by real {@link VectorCollection} instances.
 *
 * <p>Used in unit and integration tests to simulate a multi-node cluster within a single JVM.
 * Thread-safe after construction (the internal map is read-only).
 */
public final class InProcessNodeDirectory implements NodeDirectory {

  private final Map<NodeId, NodeSearchClient> clients;

  private InProcessNodeDirectory(Map<NodeId, NodeSearchClient> clients) {
    this.clients = Collections.unmodifiableMap(new HashMap<>(clients));
  }

  @Override
  public NodeSearchClient clientFor(NodeId nodeId) {
    NodeSearchClient client = clients.get(nodeId);
    if (client == null) {
      throw new IllegalArgumentException("No node registered for id: " + nodeId);
    }
    return client;
  }

  @Override
  public Set<NodeId> allNodes() {
    return clients.keySet();
  }

  /** Fluent builder. */
  public static Builder builder() {
    return new Builder();
  }

  /** Builder that accepts {@link VectorCollection} instances and wraps each in a client. */
  public static final class Builder {

    private final Map<NodeId, NodeSearchClient> clients = new HashMap<>();

    private Builder() {}

    /**
     * Registers a node backed by the given collection.
     *
     * @param nodeId the node identifier
     * @param collection the backing local collection
     * @return this builder
     */
    public Builder register(NodeId nodeId, VectorCollection collection) {
      Objects.requireNonNull(nodeId, "nodeId must not be null");
      Objects.requireNonNull(collection, "collection must not be null");
      clients.put(nodeId, new InProcessNodeSearchClient(collection));
      return this;
    }

    /**
     * Registers a node backed by a custom {@link NodeSearchClient}.
     *
     * @param nodeId the node identifier
     * @param client the custom client (useful for injecting failures in tests)
     * @return this builder
     */
    public Builder registerClient(NodeId nodeId, NodeSearchClient client) {
      Objects.requireNonNull(nodeId, "nodeId must not be null");
      Objects.requireNonNull(client, "client must not be null");
      clients.put(nodeId, client);
      return this;
    }

    public InProcessNodeDirectory build() {
      if (clients.isEmpty()) {
        throw new IllegalStateException("at least one node must be registered");
      }
      return new InProcessNodeDirectory(clients);
    }
  }
}
