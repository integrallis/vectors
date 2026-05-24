/*
 * Copyright 2025-2026 Integrallis Software, LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.integrallis.vectors.vcr;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
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
 *
 * <p><b>Thread-safety contract:</b> JUnit5 ({@code @Execution(CONCURRENT)}) and TestNG ({@code
 * parallel="methods"}) invoke each test method on whichever framework-worker thread is free. A
 * single {@code VCRContext} is shared per test class across all those methods, so the per-test
 * mutable slots ({@code currentTestId}, the in-flight cassette-key list, the call counters) are
 * kept in a {@link ThreadLocal} {@link TestState} — each worker thread sees only its own method's
 * state. Cross-test aggregate counters ({@code cacheHits} / {@code cacheMisses} / {@code apiCalls})
 * remain shared {@link AtomicLong}s.
 */
public final class VCRContext {

  /** Environment variable name used to override the VCR mode. */
  public static final String VCR_MODE_ENV = "VCR_MODE";

  private final CassetteStore cassetteStore;
  private final VCRRegistry registry;
  private VCRMode effectiveMode;

  private static final class TestState {
    String currentTestId;
    final List<CassetteKey> cassetteKeys = new ArrayList<>();
    final Map<String, AtomicInteger> callCounters = new HashMap<>();
  }

  private final ThreadLocal<TestState> threadState = ThreadLocal.withInitial(TestState::new);

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

  /**
   * Clears per-test state on the calling thread; call between tests. The current test id is
   * preserved — the framework adapter follows this with {@link #setCurrentTest} to install the next
   * test's id.
   */
  public void resetCallCounters() {
    TestState s = threadState.get();
    s.callCounters.clear();
    s.cassetteKeys.clear();
  }

  /**
   * Sets the current test identifier on the calling thread.
   *
   * @param testId the test identifier
   */
  public void setCurrentTest(String testId) {
    threadState.get().currentTestId = testId;
  }

  /**
   * @return the current test identifier on the calling thread (may be null between tests)
   */
  public String getCurrentTestId() {
    return threadState.get().currentTestId;
  }

  /**
   * Allocates the next cassette key for the given call type under the calling thread's current
   * test.
   *
   * @param type the call type (e.g. {@code "embedding"}, {@code "batch_embedding"}, {@code "chat"})
   * @return the freshly allocated key
   */
  public CassetteKey generateCassetteKey(String type) {
    Objects.requireNonNull(type, "type");
    TestState s = threadState.get();
    String testId = s.currentTestId;
    if (testId == null) {
      throw new IllegalStateException("No current test id set; call setCurrentTest() first");
    }
    int callIndex =
        s.callCounters.computeIfAbsent(type, k -> new AtomicInteger()).incrementAndGet();
    CassetteKey key = new CassetteKey(type, testId, callIndex);
    s.cassetteKeys.add(key);
    return key;
  }

  /**
   * @return a defensive copy of the cassette keys produced during the calling thread's current test
   */
  public List<CassetteKey> getCurrentCassetteKeys() {
    return new ArrayList<>(threadState.get().cassetteKeys);
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
