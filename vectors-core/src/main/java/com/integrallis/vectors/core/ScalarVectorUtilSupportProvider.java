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
package com.integrallis.vectors.core;

/**
 * Built-in {@link VectorUtilSupportProvider} for the pure-Java scalar kernels. Lowest-priority of
 * the built-ins and always available, so it is the guaranteed fallback when no SIMD or accelerated
 * provider can run.
 */
public final class ScalarVectorUtilSupportProvider implements VectorUtilSupportProvider {

  /** Public no-arg constructor required by {@link java.util.ServiceLoader}. */
  public ScalarVectorUtilSupportProvider() {}

  @Override
  public String name() {
    return "scalar";
  }

  @Override
  public int priority() {
    return SCALAR_PRIORITY;
  }

  @Override
  public boolean isAvailable() {
    return true;
  }

  @Override
  public VectorUtilSupport create() {
    return new ScalarVectorUtilSupport();
  }
}
