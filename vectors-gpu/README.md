# vectors-gpu

Optional GPU backend for java-vectors via Panama-FFM bindings to NVIDIA cuVS (no JNI maintained by
this project). Loads `libcuvs.so` at runtime only when a compatible CUDA device is present;
otherwise reports `GpuAvailability.unavailable(reason)` so callers fall back to the CPU indexes.

## Responsibility

- Runtime detection of compatible NVIDIA GPUs via the `com.nvidia.cuvs` Panama-FFM bindings
- `GpuBackend` SPI loaded via `ServiceLoader` so alternative backends can be plugged in later
- No-crash guarantee on hosts without CUDA (macOS, non-NVIDIA Linux, CI)

## Dependencies

- `vectors-core`
- `vectors-storage`
- `com.nvidia.cuvs:cuvs-java` (pure-Java JAR; native `libcuvs.so` is resolved at runtime)

## Status

Scaffold only. Detection plus SPI wiring are in place; CAGRA / IVF-PQ / brute-force index
builders and search operations are tracked for Phase 3 of the roadmap in `the roadmap`.
