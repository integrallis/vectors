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
package com.integrallis.vectors.vcr;

import java.util.Objects;

/**
 * Identifies a single cassette entry inside a {@link CassetteStore}.
 *
 * <p>The on-storage representation is {@code vcr:<type>:<testId>:<callIndex>} where {@code
 * callIndex} is zero-padded to four digits for lexicographic ordering under key-listing.
 *
 * @param type cassette type (e.g. {@code "embedding"}, {@code "chat"})
 * @param testId fully-qualified test identifier (typically {@code ClassName:methodName})
 * @param callIndex 1-based call index within the test
 */
public record CassetteKey(String type, String testId, int callIndex) {

  /** Key prefix used for all cassettes. */
  public static final String KEY_PREFIX = "vcr";

  /** Compact constructor enforcing non-null identifiers. */
  public CassetteKey {
    Objects.requireNonNull(type, "type must not be null");
    Objects.requireNonNull(testId, "testId must not be null");
    if (callIndex < 0) {
      throw new IllegalArgumentException("callIndex must be non-negative: " + callIndex);
    }
  }

  /**
   * Returns the serialized form used as the storage key.
   *
   * @return {@code vcr:<type>:<testId>:<callIndex>} with {@code callIndex} zero-padded to 4 digits
   */
  public String serializedKey() {
    return String.format("%s:%s:%s:%04d", KEY_PREFIX, type, testId, callIndex);
  }

  /**
   * Parses a serialized key back into a {@link CassetteKey}.
   *
   * @param serialized the serialized key
   * @return the parsed key or {@code null} if the input does not match the expected format
   */
  public static CassetteKey parse(String serialized) {
    if (serialized == null) {
      return null;
    }
    int firstSep = serialized.indexOf(':');
    if (firstSep < 0 || !KEY_PREFIX.equals(serialized.substring(0, firstSep))) {
      return null;
    }
    int secondSep = serialized.indexOf(':', firstSep + 1);
    int lastSep = serialized.lastIndexOf(':');
    if (secondSep < 0 || lastSep <= secondSep) {
      return null;
    }
    String type = serialized.substring(firstSep + 1, secondSep);
    String testId = serialized.substring(secondSep + 1, lastSep);
    String callIndexStr = serialized.substring(lastSep + 1);
    if (type.isEmpty() || testId.isEmpty() || callIndexStr.isEmpty()) {
      return null;
    }
    try {
      int idx = Integer.parseInt(callIndexStr);
      return new CassetteKey(type, testId, idx);
    } catch (NumberFormatException e) {
      return null;
    }
  }
}
