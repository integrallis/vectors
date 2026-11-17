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
package com.integrallis.vectors.storage.testkit;

import com.integrallis.vectors.storage.backend.StorageBackend;
import java.io.IOException;
import java.util.UUID;

/**
 * The backend under test. Each group of checks extends this; {@link StorageBackendContract} is the
 * union a backend implementation runs.
 */
public interface BackendUnderTest {

  /** The backend under test; may be created per call or shared, the checks use distinct keys. */
  StorageBackend backend() throws IOException;

  /** A key that cannot collide with another check's. */
  default String key(String suffix) {
    return suffix + "-" + UUID.randomUUID();
  }
}
