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
package com.integrallis.vectors.storage.io;

import java.io.IOException;

/**
 * Supplier of per-thread {@link RandomAccessInput} instances. Thread-safe; manages shared resources
 * (mapped segments, arenas).
 *
 * <p>Each call to {@link #open()} returns an independent reader with its own position state. The
 * supplier itself manages the shared underlying resource and should be closed when all readers are
 * done.
 */
public interface InputSupplier extends AutoCloseable {

  /** Creates a new independent reader. The caller is responsible for closing it. */
  RandomAccessInput open() throws IOException;

  /** Returns the total length of the underlying data source. */
  long length() throws IOException;

  @Override
  void close() throws IOException;
}
