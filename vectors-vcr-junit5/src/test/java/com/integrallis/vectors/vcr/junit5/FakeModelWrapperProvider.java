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
