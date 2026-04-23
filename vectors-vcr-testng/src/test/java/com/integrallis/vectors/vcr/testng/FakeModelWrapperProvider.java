package com.integrallis.vectors.vcr.testng;

import com.integrallis.vectors.vcr.CassetteStore;
import com.integrallis.vectors.vcr.ModelWrapperProvider;
import com.integrallis.vectors.vcr.VCRMode;

/** Test-only {@link ModelWrapperProvider} that wraps {@link FakeEmbeddingModel}. */
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
