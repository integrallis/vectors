// Umbrella dependency for the embedded java-vectors library. This module carries no code of its
// own; it exists so applications can depend on a single artifact — `com.integrallis:vectors` — that
// transitively brings in the whole embedded engine via `vectors-db` (core, storage, quantization,
// and the HNSW/Vamana/IVF index backends).
description = "Embedded java-vectors — single umbrella dependency (re-exports vectors-db)"

dependencies {
    api(project(":vectors-db"))
}
