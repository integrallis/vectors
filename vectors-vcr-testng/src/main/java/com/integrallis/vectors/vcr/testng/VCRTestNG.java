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

import com.integrallis.vectors.vcr.VCRMode;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Enables VCR recording/playback for a TestNG test class.
 *
 * <p>This is the TestNG analog of {@code @VCRTest} in {@code vectors-vcr-junit5}. {@link
 * VCRListener} is automatically registered for every TestNG run through the {@code
 * META-INF/services/org.testng.ITestNGListener} file shipped with this module; the listener is a
 * no-op for classes that do not carry {@code @VCRTestNG}.
 */
@Target(ElementType.TYPE)
@Retention(RetentionPolicy.RUNTIME)
public @interface VCRTestNG {

  /**
   * @return the default VCR mode for the annotated class
   */
  VCRMode mode() default VCRMode.PLAYBACK_OR_RECORD;

  /**
   * @return the directory under which cassettes are persisted
   */
  String dataDir() default "src/test/resources/vcr-data";
}
