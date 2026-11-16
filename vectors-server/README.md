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

### Object storage (durable floor)

Setting a bucket backs every collection with object storage; `--data-dir` becomes the local cache
in front of it. Each flag has an environment variable fallback.

| Flag | Env | S3 / S3-compatible | Azure Blob (`--object-store-provider azure`) |
|------|-----|--------------------|-----------------------------------------------|
| `--object-store-provider` | `VECTORS_OBJECT_STORE_PROVIDER` | `s3` (default) | `azure` |
| `--s3-bucket` | `VECTORS_S3_BUCKET` | bucket | blob container |
| `--s3-endpoint` | `VECTORS_S3_ENDPOINT` | endpoint URI; omit for AWS S3 | blob endpoint override (Azurite, sovereign clouds); omit for `https://<account>.blob.core.windows.net` |
| `--s3-region` | `VECTORS_S3_REGION` | region (`auto` for R2) | unused |
| `--s3-access-key` | `VECTORS_S3_ACCESS_KEY` | access key; omit for the AWS default credential chain | storage account name (required) |
| `--s3-secret-key` | `VECTORS_S3_SECRET_KEY` | secret key; omit for the AWS default credential chain | storage account key (required) |
| `--s3-prefix` | `VECTORS_S3_PREFIX` | key prefix shared by all collections | same |

```bash
# Azure Blob container "vectors" on storage account "acct"
./gradlew :vectors-server:run --args="--data-dir /var/lib/vectors --object-store-provider azure \
  --s3-bucket vectors --s3-access-key acct --s3-secret-key $AZURE_STORAGE_KEY"
```

Both runtimes (`vectors-storage-s3`, `vectors-storage-azure`) ship inside the server; the Azure
client uses the JDK `HttpClient` transport.

## Dependencies

- `vectors-db` — VectorCollection, Document, SearchRequest, SearchResult
- `vectors-core` — SimilarityFunction
- Helidon 4.3.4 — HTTP server, SSE, Jackson media support
- Jackson 2.18.2 — JSON serialization
- PicoCLI 4.7.6 — CLI argument parsing
- Logback 1.5.15 — logging
