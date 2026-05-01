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
package com.integrallis.vectors.studio.sidecart;

/**
 * Wraps any transport / parse / authentication failure raised by a {@link SidecartSource}. A
 * missing row is signalled by {@code Optional.empty()} on {@link SidecartSource#get(String)} and is
 * <i>not</i> an exception condition.
 */
public class SidecartSourceException extends RuntimeException {

  private static final long serialVersionUID = 1L;

  public SidecartSourceException(String message) {
    super(message);
  }

  public SidecartSourceException(String message, Throwable cause) {
    super(message, cause);
  }
}
