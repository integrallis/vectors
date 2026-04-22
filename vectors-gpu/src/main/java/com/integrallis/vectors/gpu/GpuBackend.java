package com.integrallis.vectors.gpu;

/**
 * SPI for a GPU-accelerated vector search backend. Implementations wrap a native library (e.g.
 * NVIDIA cuVS) and are loaded via {@link java.util.ServiceLoader}.
 *
 * <p>Discovery order (first {@linkplain GpuAvailability#isAvailable() available} wins):
 *
 * <ol>
 *   <li>{@link CuVsGpuBackend} -- Panama-FFM bindings to {@code libcuvs.so}
 * </ol>
 *
 * <p>Implementations MUST NOT throw from {@link #detect()}: any failure to locate or initialise the
 * native runtime is reported as {@link GpuAvailability#unavailable(String)}.
 */
public interface GpuBackend {

  /** Stable identifier for the backend, e.g. {@code "cuvs"}. */
  String name();

  /** Probes the host for GPU support. MUST be side-effect-free beyond loading native libraries. */
  GpuAvailability detect();
}
