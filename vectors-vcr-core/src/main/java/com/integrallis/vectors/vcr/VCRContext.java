package com.integrallis.vectors.vcr;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Per-test-session state for VCR recording/playback.
 *
 * <p>This is the framework-neutral analog of the previous Redis/testcontainers-based VCRContext. It
 * tracks the current test id, per-test call counters, recording statistics, and exposes the
 * pluggable {@link CassetteStore} and {@link VCRRegistry}.
 *
 * <p>The effective mode can be overridden at runtime via the {@code VCR_MODE} environment variable
 * (values: {@code PLAYBACK}, {@code PLAYBACK_OR_RECORD}, {@code RECORD}, {@code RECORD_NEW}, {@code
 * RECORD_FAILED}, {@code OFF}).
 */
public final class VCRContext {

  /** Environment variable name used to override the VCR mode. */
  public static final String VCR_MODE_ENV = "VCR_MODE";

  private final CassetteStore cassetteStore;
  private final VCRRegistry registry;
  private VCRMode effectiveMode;

  private volatile String currentTestId;
  private final List<CassetteKey> currentCassetteKeys = new ArrayList<>();
  private final Map<String, AtomicInteger> callCounters = new ConcurrentHashMap<>();

  private final AtomicLong cacheHits = new AtomicLong();
  private final AtomicLong cacheMisses = new AtomicLong();
  private final AtomicLong apiCalls = new AtomicLong();

  /**
   * Creates a context with explicit dependencies.
   *
   * @param cassetteStore the cassette store (must not be null)
   * @param registry the registry (must not be null)
   * @param annotationMode the mode declared on the test annotation; may be overridden by the {@code
   *     VCR_MODE} environment variable
   */
  public VCRContext(CassetteStore cassetteStore, VCRRegistry registry, VCRMode annotationMode) {
    this.cassetteStore = Objects.requireNonNull(cassetteStore, "cassetteStore");
    this.registry = Objects.requireNonNull(registry, "registry");
    this.effectiveMode = resolveMode(Objects.requireNonNull(annotationMode, "annotationMode"));
  }

  private static VCRMode resolveMode(VCRMode annotationMode) {
    String envMode = System.getenv(VCR_MODE_ENV);
    if (envMode == null || envMode.isEmpty()) {
      return annotationMode;
    }
    try {
      return VCRMode.valueOf(envMode.toUpperCase());
    } catch (IllegalArgumentException e) {
      System.err.println(
          "VCR: invalid "
              + VCR_MODE_ENV
              + " value '"
              + envMode
              + "'; using annotation mode "
              + annotationMode);
      return annotationMode;
    }
  }

  /**
   * @return the cassette store
   */
  public CassetteStore getCassetteStore() {
    return cassetteStore;
  }

  /**
   * @return the registry
   */
  public VCRRegistry getRegistry() {
    return registry;
  }

  /**
   * @return the effective VCR mode for the current session
   */
  public VCRMode getEffectiveMode() {
    return effectiveMode;
  }

  /**
   * Overrides the effective mode (e.g. when per-method annotations upgrade or downgrade it).
   *
   * @param mode the new effective mode
   */
  public void setEffectiveMode(VCRMode mode) {
    this.effectiveMode = Objects.requireNonNull(mode, "mode");
  }

  /** Clears per-test state; call between tests. */
  public void resetCallCounters() {
    callCounters.clear();
    currentCassetteKeys.clear();
  }

  /**
   * Sets the current test identifier.
   *
   * @param testId the test identifier
   */
  public void setCurrentTest(String testId) {
    this.currentTestId = testId;
  }

  /**
   * @return the current test identifier (may be null between tests)
   */
  public String getCurrentTestId() {
    return currentTestId;
  }

  /**
   * Allocates the next cassette key for the given call type under the current test.
   *
   * @param type the call type (e.g. {@code "embedding"}, {@code "batch_embedding"}, {@code "chat"})
   * @return the freshly allocated key
   */
  public CassetteKey generateCassetteKey(String type) {
    Objects.requireNonNull(type, "type");
    if (currentTestId == null) {
      throw new IllegalStateException("No current test id set; call setCurrentTest() first");
    }
    String counterKey = currentTestId + ":" + type;
    int callIndex =
        callCounters.computeIfAbsent(counterKey, k -> new AtomicInteger()).incrementAndGet();
    CassetteKey key = new CassetteKey(type, currentTestId, callIndex);
    currentCassetteKeys.add(key);
    return key;
  }

  /**
   * @return a defensive copy of the cassette keys produced during the current test
   */
  public List<CassetteKey> getCurrentCassetteKeys() {
    return new ArrayList<>(currentCassetteKeys);
  }

  /** Records a cache hit. */
  public void recordCacheHit() {
    cacheHits.incrementAndGet();
  }

  /** Records a cache miss. */
  public void recordCacheMiss() {
    cacheMisses.incrementAndGet();
  }

  /** Records an API call. */
  public void recordApiCall() {
    apiCalls.incrementAndGet();
  }

  /**
   * @return the cache-hit count across all tests handled by this context
   */
  public long getCacheHits() {
    return cacheHits.get();
  }

  /**
   * @return the cache-miss count across all tests handled by this context
   */
  public long getCacheMisses() {
    return cacheMisses.get();
  }

  /**
   * @return the API call count across all tests handled by this context
   */
  public long getApiCalls() {
    return apiCalls.get();
  }
}
