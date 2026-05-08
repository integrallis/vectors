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
package com.integrallis.vectors.ingest;

import java.util.OptionalLong;

/**
 * SPI for a streaming source of {@link IngestDoc}. Implementations must be re-iterable when {@code
 * startOffset() > 0} so the pipeline can honour a resume cursor.
 */
public interface IngestSource extends Iterable<IngestDoc> {

  /** A short, stable name used for logging and cursor scoping. */
  String name();

  /** Optional total document count for progress reporting. */
  default OptionalLong estimatedSize() {
    return OptionalLong.empty();
  }

  /** Resume offset: the iterator must skip this many leading docs before yielding the first one. */
  default long startOffset() {
    return 0L;
  }
}
