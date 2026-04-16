package com.integrallis.vectors.distributed;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.Consumer;

/**
 * Static in-process {@link ClusterMembership} for tests and single-process deployments.
 *
 * <p>The live-node set is fixed at construction time. Change listeners can be registered and fired
 * manually via {@link #fireEvent(ClusterMembership.MembershipEvent)}.
 */
public final class StaticClusterMembership implements ClusterMembership {

  private final Set<NodeId> nodes;
  private final List<Consumer<MembershipEvent>> listeners = new CopyOnWriteArrayList<>();

  /**
   * @param nodes the fixed set of live nodes (must not be null or empty)
   */
  public StaticClusterMembership(Collection<NodeId> nodes) {
    Objects.requireNonNull(nodes, "nodes must not be null");
    if (nodes.isEmpty()) {
      throw new IllegalArgumentException("nodes must not be empty");
    }
    this.nodes = Set.copyOf(nodes);
  }

  @Override
  public Set<NodeId> liveNodes() {
    return nodes;
  }

  @Override
  public void registerChangeListener(Consumer<MembershipEvent> listener) {
    listeners.add(Objects.requireNonNull(listener, "listener must not be null"));
  }

  /**
   * Fires an event to all registered listeners. Useful in tests to simulate node joins, leaves, or
   * BuoyIndex updates.
   *
   * @param event the event to broadcast
   */
  public void fireEvent(MembershipEvent event) {
    Objects.requireNonNull(event, "event must not be null");
    List<RuntimeException> errors = new ArrayList<>();
    for (Consumer<MembershipEvent> listener : listeners) {
      try {
        listener.accept(event);
      } catch (RuntimeException e) {
        errors.add(e);
      }
    }
    if (!errors.isEmpty()) {
      RuntimeException first = errors.get(0);
      errors.subList(1, errors.size()).forEach(first::addSuppressed);
      throw first;
    }
  }
}
