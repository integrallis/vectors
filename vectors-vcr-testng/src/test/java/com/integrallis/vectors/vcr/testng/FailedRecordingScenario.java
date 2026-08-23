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
package com.integrallis.vectors.vcr.testng;

import static org.testng.Assert.fail;

import com.integrallis.vectors.vcr.VCRMode;
import com.integrallis.vectors.vcr.VCRModel;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

/** Scenario used to verify cleanup after a model writes a cassette and the test then fails. */
@VCRTestNG(mode = VCRMode.RECORD, dataDir = "overridden-by-sysprop")
public class FailedRecordingScenario {

  @VCRModel FakeEmbeddingModel embedder;

  @BeforeMethod
  public void setUp() {
    embedder = prompt -> new float[] {(float) prompt.length(), 42f};
  }

  @Test
  public void recordsThenFails() {
    embedder.embed("written before failure");
    fail("intentional failure after cassette write");
  }
}
