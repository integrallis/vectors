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
package com.integrallis.vectors.vcr.junit5;

import com.integrallis.vectors.vcr.CassetteStore;
import com.integrallis.vectors.vcr.ModelWrapperProvider;
import com.integrallis.vectors.vcr.VCRMode;

/**
 * Test-only wrapper provider registered through {@code META-INF/services} that knows how to
 * decorate {@link FakeEmbeddingModel} with a VCR interceptor.
 */
public final class FakeModelWrapperProvider implements ModelWrapperProvider {

  @Override
  public Object wrap(
      Object model, String testId, VCRMode mode, String modelName, CassetteStore cassetteStore) {
    if (model instanceof FakeEmbeddingModel fake && !(model instanceof VCRFakeEmbeddingModel)) {
      return new VCRFakeEmbeddingModel(fake, testId, mode, cassetteStore);
    }
    return null;
  }
}
