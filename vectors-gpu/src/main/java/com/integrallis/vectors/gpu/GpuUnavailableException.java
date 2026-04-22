package com.integrallis.vectors.gpu;

/**
 * Thrown when a GPU-backed index is constructed or used on a host where {@link GpuProvider} reports
 * {@link GpuAvailability#isAvailable()} false. Wraps the best-effort reason string reported by the
 * backend (e.g. {@code "cuvs-java not on classpath"}, {@code "UnsatisfiedLinkError: libcuvs.so"},
 * {@code "no compatible CUDA devices found"}).
 */
public final class GpuUnavailableException extends RuntimeException {

  private static final long serialVersionUID = 1L;

  public GpuUnavailableException(String reason) {
    super("GPU backend not available: " + reason);
  }

  public GpuUnavailableException(String reason, Throwable cause) {
    super("GPU backend not available: " + reason, cause);
  }
}
