# vectors-server

[![MFCQI](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/integrallis/vectors/main/vectors-server/.github/badges/mfcqi.json)](https://github.com/integrallis/mfcqi-java)

Standalone HTTP server exposing vector collections via a JSON REST API. Built on Helidon SE 4 (Nima virtual-thread runtime) with PicoCLI for command-line entry.

## Responsibility

- REST endpoints for collection lifecycle (create, list, describe, delete)
- Document operations (upsert, delete, get)
- Vector search and batch search
- Server-sent events (SSE) for real-time collection updates
- Collection discovery from persistent storage on startup
- RFC 7807 problem-details error responses

## Key Types

- `VectorsServer` — PicoCLI entry point; `start(ServerConfig)` returns a `ServerHandle`
- `ServerConfig` — port, data directory, max connections, shutdown timeout
- `CollectionRegistry` — in-memory collection instance manager
- `CollectionDiscovery` — opens persistent collections from disk on startup
- `ApiRouting` — assembles route tree from `AdminRoutes`, `CollectionsRoutes`, `DocumentsRoutes`, `SearchRoutes`, `EventsRoutes`

## Running

```bash
cd vectors
./gradlew :vectors-server:run
```

Default port: 8287.

## Dependencies

- `vectors-db` — VectorCollection, Document, SearchRequest, SearchResult
- `vectors-core` — SimilarityFunction
- Helidon 4.3.4 — HTTP server, SSE, Jackson media support
- Jackson 2.18.2 — JSON serialization
- PicoCLI 4.7.6 — CLI argument parsing
- Logback 1.5.15 — logging
