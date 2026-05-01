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
 * One row from a sidecart source: the human-readable text payload (may be {@code null} when the
 * source only carries a binary blob), the binary blob (may be {@code null}), and an optional MIME
 * type that the UI uses to pick a viewer.
 *
 * <p>A sidecart source <i>always</i> populates at least one of {@code text} / {@code blob}; an
 * "empty" lookup is signalled by the source returning {@code Optional.empty()} from {@code get()}.
 */
public record SidecartRecord(String text, byte[] blob, String mime) {

  public static SidecartRecord ofText(String text, String mime) {
    return new SidecartRecord(text, null, mime);
  }

  public static SidecartRecord ofBlob(byte[] blob, String mime) {
    return new SidecartRecord(null, blob, mime);
  }

  public static SidecartRecord of(String text, byte[] blob, String mime) {
    return new SidecartRecord(text, blob, mime);
  }
}
