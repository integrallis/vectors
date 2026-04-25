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

import com.integrallis.vectors.vcr.VCRMode;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;
import org.junit.jupiter.api.extension.ExtendWith;

/**
 * Enables VCR recording/playback for a test class.
 *
 * <p>By default the extension builds an {@link com.integrallis.vectors.vcr.ExactCassetteStore}
 * backed by a {@link com.integrallis.vectors.storage.backend.LocalFileStorageBackend} pointed at
 * {@link #dataDir()}. Applications that need semantic matching or custom storage can install a
 * {@link CassetteStoreFactory} through the {@link java.util.ServiceLoader}.
 *
 * <p>The effective mode can also be overridden at runtime via the {@code VCR_MODE} environment
 * variable.
 */
@Target(ElementType.TYPE)
@Retention(RetentionPolicy.RUNTIME)
@ExtendWith(VCRExtension.class)
public @interface VCRTest {

  /**
   * @return the default VCR mode for the annotated test class
   */
  VCRMode mode() default VCRMode.PLAYBACK_OR_RECORD;

  /**
   * @return the directory under which cassettes are persisted (relative paths resolve against the
   *     JVM working directory)
   */
  String dataDir() default "src/test/resources/vcr-data";
}
