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

import java.lang.reflect.Field;
import java.util.List;
import java.util.Objects;
import java.util.ServiceLoader;

/**
 * Facade that delegates model wrapping to {@link ModelWrapperProvider}s discovered via {@link
 * ServiceLoader}.
 *
 * <p>Use {@link #wrapField} from a framework extension/listener to replace a {@code @VCRModel}
 * field on the test instance with its wrapped counterpart.
 */
public final class VCRModelWrapper {

  private VCRModelWrapper() {}

  /**
   * Wraps the value of {@code field} on {@code testInstance} using the first supporting provider.
   *
   * @param testInstance the test-class instance
   * @param field the {@code @VCRModel}-annotated field
   * @param testId the current test identifier
   * @param mode the effective mode
   * @param modelName the model name override (may be empty)
   * @param cassetteStore the cassette store
   * @return {@code true} if the field was successfully wrapped
   */
  public static boolean wrapField(
      Object testInstance,
      Field field,
      String testId,
      VCRMode mode,
      String modelName,
      CassetteStore cassetteStore) {
    Objects.requireNonNull(testInstance, "testInstance");
    Objects.requireNonNull(field, "field");
    try {
      field.setAccessible(true);
      Object model = field.get(testInstance);
      if (model == null) {
        return false;
      }
      String effectiveModelName =
          modelName == null || modelName.isEmpty() ? field.getName() : modelName;
      Object wrapped = wrapModel(model, testId, mode, effectiveModelName, cassetteStore);
      if (wrapped != null && wrapped != model) {
        field.set(testInstance, wrapped);
        return true;
      }
      return false;
    } catch (IllegalAccessException e) {
      return false;
    }
  }

  /**
   * Wraps {@code model} by iterating over registered {@link ModelWrapperProvider}s until one
   * succeeds.
   *
   * @param model the model instance
   * @param testId the current test identifier
   * @param mode the effective VCR mode
   * @param modelName the model name for cache keys
   * @param cassetteStore the cassette store
   * @return the wrapped model or {@code null} if no provider supports the type
   */
  public static Object wrapModel(
      Object model, String testId, VCRMode mode, String modelName, CassetteStore cassetteStore) {
    Objects.requireNonNull(model, "model");
    for (ModelWrapperProvider provider : providers()) {
      try {
        Object wrapped = provider.wrap(model, testId, mode, modelName, cassetteStore);
        if (wrapped != null) {
          return wrapped;
        }
      } catch (NoClassDefFoundError e) {
        // Provider's framework not on classpath — try the next.
      }
    }
    return null;
  }

  /**
   * @return all registered {@link ModelWrapperProvider}s (materialized list)
   */
  public static List<ModelWrapperProvider> providers() {
    List<ModelWrapperProvider> list = new java.util.ArrayList<>();
    for (ModelWrapperProvider p : ServiceLoader.load(ModelWrapperProvider.class)) {
      list.add(p);
    }
    return list;
  }
}
