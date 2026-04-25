# vectors-distributed

Scatter-gather distributed search layer. Broadcasts queries across multiple nodes and merges results with pluggable cluster membership and shard routing.

## Responsibility

- `DistributedVectorCollection` wraps a local `VectorCollection` and routes queries to all cluster nodes
- Scatter-gather execution via virtual threads (one per node per query)
- Pluggable cluster membership (static list, gossip protocol)
- Consistent-hash shard ownership for document-to-shard assignment
- Top-k merging from per-node partial results
- In-process transport for single-JVM testing

## Key Types

- `DistributedVectorCollection` — main facade (builder pattern)
- `ScatterGatherExecutor` — stateless fan-out executor
- `NodeDirectory` — registry mapping node IDs to search clients
- `NodeSearchClient` — RPC interface for querying a remote node
- `ClusterMembership` — cluster topology abstraction
- `ShardRouter` / `ShardOwnership` — document routing
- `TopKMerger` — merges ranked results across nodes

## Status

In progress. Core scatter-gather and in-process transport are functional. Production transports (gRPC, HTTP) are deferred.

## Dependencies

- `vectors-core` — similarity functions
- `vectors-storage` — storage layer
- `vectors-db` — VectorCollection, Document, SearchRequest, SearchResult
