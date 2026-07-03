# vectors-studio-web

[![MFCQI](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/integrallis/vectors/main/vectors-studio-web/.github/badges/mfcqi.json)](https://github.com/integrallis/mfcqi-java)

Helidon SE 4 web frontend for **Vectors Studio** — an interactive exploration and dimensionality-reduction workbench for `vectors-db` collections. Server-rendered HTML pages built with JTE + HTMX, plus a 3D projector island powered by Three.js (loaded from a CDN) and an SSE projection API.

## Responsibility

- PicoCLI launcher (`StudioServer`) and programmatic entry point (`StudioServer.start(StudioConfig)`)
- HTTP routing, JTE rendering, and static asset serving
- HTMX-driven pages for collection browse, search, document inspection, and projector
- JSON + SSE API for long-running projection jobs (run on virtual threads via `ProjectionJobManager`)

## Run

```bash
./gradlew :vectors-studio-web:run --args="--connection embedded:/path/to/data"
./gradlew :vectors-studio-web:run --args="--connection http://localhost:8287 --token <bearer>"
```

CLI flags: `--connection` (required, `embedded:/path` or `http(s)://host:port`), `--port` (default `8288`), `--token` (Bearer token forwarded to the remote backend).

## Endpoints

| Method | Path                                              | Source              | Purpose                                                |
| ------ | ------------------------------------------------- | ------------------- | ------------------------------------------------------ |
| GET    | `/`                                               | `HomeRoutes`        | 301 redirect to `/collections`                         |
| GET    | `/collections`                                    | `HomeRoutes`        | Full collections list (with delete-collection actions) |
| DELETE | `/collections/{name}`                             | `HomeRoutes`        | Delete a collection (HTMX-driven, returns empty body)  |
| GET    | `/collections/{name}`                             | `CollectionRoutes`  | Collection overview + first preview page               |
| GET    | `/collections/{name}/preview?offset&limit`        | `CollectionRoutes`  | Paginated preview fragment (HTMX) with total + page-size selector |
| POST   | `/collections/{name}/search`                      | `SearchRoutes`      | Run hybrid search (form-encoded); returns hits-list HTMX fragment |
| GET    | `/collections/{name}/documents/{id}`              | `DocumentRoutes`    | Document inspector with kind-aware viewer              |
| GET    | `/collections/{name}/blobs/{id}`                  | `BlobRoutes`        | Streams document blob (sidecart-first, then backend); content-type sniffed |
| GET    | `/collections/{name}/projector`                   | `ProjectorRoutes`   | 3D projector page (Three.js island)                    |
| POST   | `/api/projections`                                | `ApiRoutes`         | Submit a projection job (JSON)                         |
| GET    | `/api/projections/{id}/events`                    | `ApiRoutes`         | SSE stream of `ProjectionEvent`s; replays terminal state for completed jobs |
| DELETE | `/api/projections/{id}`                           | `ApiRoutes`         | Cancel a running projection                            |
| GET    | `/static/*`                                       | `StaticContentService` | `studio.css`, `projector.js`                        |
| GET    | `/health`                                         | `HealthRoutes`      | Liveness probe                                         |

## Key Types

- `StudioServer` — PicoCLI command and programmatic entry (`start(StudioConfig)` returns `StudioServerHandle`)
- `StudioConfig(int port, StudioSession session, SidecartRegistry sidecart)` — record; `DEFAULT_PORT = 8288`. A 2-arg constructor defaults the registry to empty.
- `BlobRoutes` — sidecart-first blob serving with magic-byte content-type sniffing and metadata-driven MIME hints
- `TemplateSupport` — JTE-callable helpers (`truncate`, `prettyJson`, `pageCount`, `currentPage`)
- `StudioServerHandle` — `port()`, `stop()`
- `StudioRouting` — assembles every route, the JTE engine, and the static-content service
- `JteEngineFactory` — runtime JTE engine over `templates/` on the classpath (no precompilation)
- `ViewRenderer` — `render(res, template, ctx)` and `renderFragment(res, status, template, ctx)`
- `ProjectionJobManager` — submits jobs onto virtual threads; tracks state per job id
- `ProjectionJob` — holds state (`PENDING`/`RUNNING`/`DONE`/`ERROR`) and a `Flow.Publisher<ProjectionEvent>`
- `ProjectionEvent` (sealed): `Started`, `Progress`, `Done`, `Error`
- `ProjectionRequestDto` — wire shape posted to `/api/projections`

## Templates and assets

- `templates/layout.jte` — page shell (HTMX + Alpine via CDN)
- `templates/{collections,collection,document,projector}.jte`
- `templates/partials/` — HTMX fragments (`documentList`, `hitsList`, …)
- `static/studio.css` — themed stylesheet (clean-line tokens distilled from the Brex web app, Inter type ramp, brand-green accent)
- `static/img/{logo.png, favicon*.png, favicon.ico}` — java-vectors brand assets
- `static/projector.js` — hand-written ES module that imports `three` from a CDN, opens the SSE stream, and renders the projection

## Testing

```bash
./gradlew :vectors-studio-web:test
./gradlew :vectors-studio-web:integrationTest
```

`StudioServerSmokeIT` covers the static shell and 404 paths; `ProjectionApiIT` covers the SSE projection lifecycle end-to-end against an embedded backend.

## Dependencies

- `vectors-studio-core`, `vectors-studio-sidecart`, `vectors-db`, `vectors-server-client`
- Helidon 4.3.4 — `helidon-webserver`, `helidon-webserver-sse`, `helidon-webserver-static-content`, `helidon-http-media-jackson`
- Jackson 2.18.2 — `jackson-databind`, `jackson-datatype-jsr310`
- JTE 3.2.3 — `jte`, `jte-runtime` (runtime template parsing, no Gradle precompilation)
- PicoCLI 4.7.6
- `dev.langchain4j:langchain4j-core:1.13.1` (compile-only)
- Logback 1.5.15 (runtime)
