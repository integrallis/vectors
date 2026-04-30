# vectors-server-client

Java HTTP client for the `vectors-server` REST API. Built on the JDK `java.net.http.HttpClient` with Jackson for JSON binding — no third-party HTTP dependency.

## Responsibility

- Collection lifecycle: `createCollection`, `listCollections`, `describe`, `collectionExists`, `deleteCollection`
- Document writes: `upsertDocuments`, `deleteDocument`, `commit`
- Bulk reads (Studio prerequisites): `previewDocuments` (paginated), `vectorsBatch`, `sample`
- Search: `search` (vector), `hybridSearch` (vector + text with server-side fusion)
- Blob retrieval: `getBlob` (returns `Optional<byte[]>`, 404 → empty)
- Health probe: `isHealthy`
- Exception mapping: any 4xx/5xx response is wrapped in a `VectorsServerException` carrying the HTTP status and the raw response body

## Key Types

- `VectorsServerClient` — main entry point. Single-arg constructor `(String baseUrl)` builds an internal `HttpClient` with a 5-second connect timeout and HTTP/1.1; thread-safe; `AutoCloseable`.
- `CollectionInfo` — full collection metadata (name, dimension, metric, indexType, params, size, …); mirrors the `vectors-server` wire format.
- `DocumentPayload` — request/response shape for upsert (id, vector, text, metadata, blob).
- `DocumentPage` — paginated preview response (items, total, offset, limit).
- `VectorsBatchResponse` — bulk vector retrieval response (vectors + missing ids).
- `SampleResponse` — uniform random sample of documents (items, requested, returned).
- `SearchHit` — single ordered search result.
- `VectorsServerException` — runtime exception with `statusCode()` and the server response body as its message.

## Dependencies

- `vectors-core` — `SimilarityFunction`
- `com.fasterxml.jackson.core:jackson-databind:2.18.2`
- `com.fasterxml.jackson.datatype:jackson-datatype-jsr310:2.18.2`
- JDK 25 `java.net.http.HttpClient`
