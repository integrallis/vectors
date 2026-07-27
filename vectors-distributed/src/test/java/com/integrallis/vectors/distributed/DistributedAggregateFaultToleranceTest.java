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
package com.integrallis.vectors.distributed;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.SearchResult;
import com.integrallis.vectors.db.VectorCollection;
import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.IntSupplier;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Regression for audit distributed #2: {@code size()}/{@code physicalSize()} used to aggregate node
 * counts in a sequential loop with no timeout and no error handling, so one down node threw the
 * whole call and one slow node hung it. They now fan out in parallel under a timeout and fail fast
 * with a {@link PartialResultException} (carrying the partial sum + unreachable nodes) instead.
 */
@Tag("unit")
class DistributedAggregateFaultToleranceTest {

  private static final Duration TIMEOUT = Duration.ofMillis(500);

  /** NodeSearchClient whose size()/physicalSize() are driven by injectable suppliers. */
  private static final class StubClient implements NodeSearchClient {
    private final IntSupplier size;

    StubClient(IntSupplier size) {
      this.size = size;
    }

    @Override
    public SearchResult search(LocalSearchRequest request) {
      throw new UnsupportedOperationException("not used in aggregate tests");
    }

    @Override
    public int size() {
      return size.getAsInt();
    }

    @Override
    public int physicalSize() {
      return size.getAsInt();
    }
  }

  private static final class StubDirectory implements NodeDirectory {
    private final Map<NodeId, NodeSearchClient> clients;

    StubDirectory(Map<NodeId, NodeSearchClient> clients) {
      this.clients = clients;
    }

    @Override
    public NodeSearchClient clientFor(NodeId nodeId) {
      NodeSearchClient c = clients.get(nodeId);
      if (c == null) throw new IllegalArgumentException("unknown node: " + nodeId);
      return c;
    }

    @Override
    public Set<NodeId> allNodes() {
      return clients.keySet();
    }
  }

  private static DistributedVectorCollection collectionOver(
      Map<NodeId, NodeSearchClient> clients, NodeId local) {
    VectorCollection localCol =
        VectorCollection.builder()
            .dimension(4)
            .metric(SimilarityFunction.DOT_PRODUCT)
            .indexType(IndexType.FLAT)
            .build();
    return DistributedVectorCollection.builder()
        .localCollection(localCol)
        .localNodeId(local)
        .directory(new StubDirectory(clients))
        .allNodes(List.copyOf(clients.keySet()))
        .timeout(TIMEOUT)
        .build();
  }

  @Test
  void allNodesHealthy_returnsExactSum() {
    NodeId a = new NodeId("a");
    NodeId b = new NodeId("b");
    NodeId c = new NodeId("c");
    Map<NodeId, NodeSearchClient> clients = new LinkedHashMap<>();
    clients.put(a, new StubClient(() -> 10));
    clients.put(b, new StubClient(() -> 20));
    clients.put(c, new StubClient(() -> 30));

    DistributedVectorCollection dvc = collectionOver(clients, a);

    assertThat(dvc.size()).isEqualTo(60);
    assertThat(dvc.physicalSize()).isEqualTo(60);
  }

  @Test
  void oneNodeThrows_failsFastWithPartialSignal() {
    NodeId a = new NodeId("a");
    NodeId b = new NodeId("b");
    NodeId down = new NodeId("down");
    Map<NodeId, NodeSearchClient> clients = new LinkedHashMap<>();
    clients.put(a, new StubClient(() -> 10));
    clients.put(b, new StubClient(() -> 20));
    clients.put(
        down,
        new StubClient(
            () -> {
              throw new RuntimeException("node down");
            }));

    DistributedVectorCollection dvc = collectionOver(clients, a);

    assertThatThrownBy(dvc::size)
        .isInstanceOf(PartialResultException.class)
        .satisfies(
            e -> {
              PartialResultException p = (PartialResultException) e;
              assertThat(p.partialTotal()).as("sum over the two healthy nodes").isEqualTo(30);
              assertThat(p.unreachableNodes()).containsExactly(down);
            });
  }

  @Test
  void slowNode_isBoundedByTimeout() {
    NodeId a = new NodeId("a");
    NodeId slow = new NodeId("slow");
    Map<NodeId, NodeSearchClient> clients = new LinkedHashMap<>();
    clients.put(a, new StubClient(() -> 10));
    clients.put(
        slow,
        new StubClient(
            () -> {
              try {
                Thread.sleep(30_000); // far beyond the 500ms timeout
              } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException(e);
              }
              return 999;
            }));

    DistributedVectorCollection dvc = collectionOver(clients, a);

    long start = System.nanoTime();
    assertThatThrownBy(dvc::size)
        .isInstanceOf(PartialResultException.class)
        .satisfies(e -> assertThat(((PartialResultException) e).unreachableNodes()).contains(slow));
    long elapsedMs = (System.nanoTime() - start) / 1_000_000L;

    // The call must return promptly after the timeout, never hang on the 30s-sleeping node.
    assertThat(elapsedMs).as("bounded by the timeout, not the slow node").isLessThan(5_000L);
  }
}
