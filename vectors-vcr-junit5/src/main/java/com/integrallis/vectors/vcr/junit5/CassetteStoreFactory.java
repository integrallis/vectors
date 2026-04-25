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
import java.nio.file.Path;

/**
 * SPI for constructing a {@link CassetteStore} from a {@link VCRTest} configuration.
 *
 * <p>Discovered via {@link java.util.ServiceLoader}. The first provider that returns a non-null
 * {@link CassetteStore} from {@link #create(Path)} wins; if no provider is registered, the
 * extension falls back to an {@link com.integrallis.vectors.vcr.ExactCassetteStore} over a {@link
 * com.integrallis.vectors.storage.backend.LocalFileStorageBackend} rooted at the configured {@code
 * dataDir}.
 */
public interface CassetteStoreFactory {

  /**
   * Creates a store for the given data directory, or returns {@code null} to defer to the next
   * provider (or the built-in default if no provider handles the request).
   *
   * @param dataDir the resolved absolute data directory
   * @return the store or {@code null}
   */
  CassetteStore create(Path dataDir);
}
