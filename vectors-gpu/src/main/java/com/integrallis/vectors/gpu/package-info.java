/**
 * Optional GPU backend for java-vectors via Panama-FFM bindings to NVIDIA cuVS.
 *
 * <p>The module is pure Java on the JVM side; the native {@code libcuvs.so} is loaded at runtime
 * only if a CUDA device is present. When GPU support is unavailable (no CUDA driver, non-NVIDIA
 * host, or cuVS runtime missing) the backend reports {@link
 * com.integrallis.vectors.gpu.GpuAvailability} with a reason string and callers fall back to the
 * CPU indexes in {@code vectors-hnsw} / {@code vectors-vamana} / {@code vectors-ivf}.
 */
package com.integrallis.vectors.gpu;
