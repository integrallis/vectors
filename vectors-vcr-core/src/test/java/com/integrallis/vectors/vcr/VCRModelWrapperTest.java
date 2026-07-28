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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatNullPointerException;

import com.integrallis.vectors.storage.backend.HeapStorageBackend;
import java.lang.reflect.Field;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class VCRModelWrapperTest {

  private final CassetteStore store =
      new ExactCassetteStore(new HeapStorageBackend(), new TestSerializer());

  @Test
  void discoversProviderAndWrapsSupportedModel() {
    TestModelWrapperProvider.TestModel model = new TestModelWrapperProvider.TestModel("delegate");

    Object wrapped = VCRModelWrapper.wrapModel(model, "T:model", VCRMode.RECORD, "named", store);

    assertThat(wrapped)
        .isEqualTo(
            new TestModelWrapperProvider.WrappedModel(
                model, "T:model", VCRMode.RECORD, "named", store));
    assertThat(VCRModelWrapper.providers())
        .singleElement()
        .satisfies(provider -> assertThat(provider.name()).isEqualTo("TestModelWrapperProvider"));
  }

  @Test
  void wrapFieldReplacesPrivateValueAndDefaultsModelNameToFieldName() throws Exception {
    ModelHolder holder = new ModelHolder();
    Field field = ModelHolder.class.getDeclaredField("model");

    assertThat(VCRModelWrapper.wrapField(holder, field, "T:field", VCRMode.PLAYBACK, "", store))
        .isTrue();
    assertThat(holder.model)
        .isEqualTo(
            new TestModelWrapperProvider.WrappedModel(
                new TestModelWrapperProvider.TestModel("field"),
                "T:field",
                VCRMode.PLAYBACK,
                "model",
                store));
  }

  @Test
  void wrapFieldReturnsFalseForNullOrUnsupportedValues() throws Exception {
    ModelHolder holder = new ModelHolder();

    assertThat(
            VCRModelWrapper.wrapField(
                holder,
                ModelHolder.class.getDeclaredField("nullModel"),
                "T:null",
                VCRMode.RECORD,
                null,
                store))
        .isFalse();
    assertThat(
            VCRModelWrapper.wrapField(
                holder,
                ModelHolder.class.getDeclaredField("unsupported"),
                "T:unsupported",
                VCRMode.RECORD,
                "unsupported",
                store))
        .isFalse();
  }

  @Test
  void rejectsNullRequiredArguments() throws Exception {
    Field field = ModelHolder.class.getDeclaredField("model");

    assertThatNullPointerException()
        .isThrownBy(
            () -> VCRModelWrapper.wrapField(null, field, "T:null", VCRMode.RECORD, "model", store));
    assertThatNullPointerException()
        .isThrownBy(
            () ->
                VCRModelWrapper.wrapField(
                    new ModelHolder(), null, "T:null", VCRMode.RECORD, "model", store));
    assertThatNullPointerException()
        .isThrownBy(
            () -> VCRModelWrapper.wrapModel(null, "T:null", VCRMode.RECORD, "model", store));
  }

  private static final class ModelHolder {
    private Object model = new TestModelWrapperProvider.TestModel("field");
    private Object nullModel;
    private String unsupported = "plain";
  }
}
