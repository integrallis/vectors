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
package com.integrallis.vectors.text.h2;

import com.integrallis.vectors.hybrid.text.TextIndexSpi;
import com.integrallis.vectors.hybrid.text.TextIndexSpiFactory;
import java.nio.file.Path;

/**
 * ServiceLoader provider that creates {@link H2TextIndex} instances.
 *
 * <p>Registered via {@code
 * META-INF/services/com.integrallis.vectors.hybrid.text.TextIndexSpiFactory}.
 */
public final class H2TextIndexFactory implements TextIndexSpiFactory {

  /** Public no-arg constructor required by ServiceLoader. */
  public H2TextIndexFactory() {}

  @Override
  public TextIndexSpi create(String collectionName) {
    return new H2TextIndex(collectionName);
  }

  @Override
  public TextIndexSpi create(String collectionName, Path dataDir) {
    return new H2TextIndex(collectionName, dataDir);
  }
}
