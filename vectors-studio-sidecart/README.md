# vectors-studio-sidecart

Pluggable sidecart sources for **Vectors Studio**: text, image, audio, and arbitrary binary payloads keyed by external document id and resolved *outside* of the vector index. Lets a thin vector store point at H2, the filesystem, or any HTTP-reachable object store (presigned S3 / GCS / R2 URLs, internal blob services) for the human-readable payload while the embeddings stay in `vectors-server`.

## Responsibility

- `SidecartSource` SPI — `Optional<SidecartRecord> get(String id)`; `AutoCloseable`
- `SidecartRecord(text, blob, mime)` — one row from a source; populates at least one of `text` / `blob`
- `SidecartRegistry` — concurrent map of collection name → `SidecartSource`, with rebind-and-close semantics and bulk `close()`
- `SidecartConfig` (sealed) — typed config carrier (`File`, `H2`, `Http`) that materialises a source through `build()`
- `SidecartSourceException` — uniform wrapper for transport / parse / authentication failures (a missing row is returned as `Optional.empty()`, not raised)

## Concrete sources

- `FileSidecartSource(baseDir, extension, textMode, mimeType)` — flat directory; ids are sanitised so path separators (`/`, `\`, `:`) cannot escape `baseDir`. `textMode=true` decodes as UTF-8.
- `H2SidecartSource(jdbcUrl, user, password, table, idColumn, textColumn?, blobColumn?, mimeColumn?)` — JDBC; identifiers are validated against `[A-Za-z_][A-Za-z0-9_]{0,63}` before interpolation (PreparedStatement only parameterises values, not identifiers).
- `HttpSidecartSource(urlTemplate, textMode, defaultMime, headers)` — generic HTTP fetcher; `urlTemplate` substitutes `{id}` after URL-encoding. Covers presigned S3/GCS/R2 URLs and internal blob services without taking a hard dependency on any cloud SDK.

## Wiring into `vectors-studio-web`

The web layer accepts an optional `SidecartRegistry` on `StudioConfig`. `BlobRoutes` consults the registry first, then falls back to the active `StudioBackend.getBlob(name, id)` (which proxies to `vectors-server`'s `/v1/collections/{name}/blobs/{id}` for the remote backend). Empty registry → today's behaviour preserved.

## Bulk ingest bridge

`SidecartWriterSink` adapts an `H2SidecartWriter` or `D1SidecartWriter` to the `vectors-ingest` `SidecartSink` SPI, so a `BulkIngestor` pipeline can populate the sidecart text/blob rows alongside the vector backend in lock-step with each batch commit. See `:demos:studio-r2-sidecart` for an end-to-end CLI that drives a corpus into R2 + a sidecart in one pass.

## Dependencies

- `vectors-studio-core`
- `vectors-ingest`
- `com.h2database:h2:2.3.232`

## Roadmap

A native `S3SidecartSource` using AWS SDK v2 with IAM auth is planned as a peer of the three above, slotting into the same SPI without touching the registry or web layer.
