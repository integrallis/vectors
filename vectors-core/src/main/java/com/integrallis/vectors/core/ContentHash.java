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
package com.integrallis.vectors.core;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.Objects;

/**
 * SHA-256 of the input a vector was produced from.
 *
 * <p><b>Hash the embedded input, not the raw source.</b> The question this answers is "would
 * embedding this again produce the same vector", so it must cover what actually reached the model:
 * after chunking, after any role prefix, after truncation. A hash of the original document cannot
 * answer that — change a prefix or a chunk size and the source is byte-identical while every vector
 * differs.
 *
 * <p>The algorithm and encoding are deliberately boring, because this becomes a persisted identity
 * that has to stay comparable across versions. SHA-256, lowercase hex, 64 characters. A shorter or
 * faster hash would save bytes nobody is counting and foreclose that comparison.
 */
public final class ContentHash {

  /** Length of a valid hash string: SHA-256 as lowercase hex. */
  public static final int LENGTH = 64;

  private ContentHash() {}

  /** Hashes the exact string that was, or will be, handed to the embedding model. */
  public static String of(String embeddedInput) {
    Objects.requireNonNull(embeddedInput, "embeddedInput");
    return of(embeddedInput.getBytes(StandardCharsets.UTF_8));
  }

  /** Hashes bytes, for inputs that are not text — an image or audio clip being embedded. */
  public static String of(byte[] embeddedInput) {
    Objects.requireNonNull(embeddedInput, "embeddedInput");
    try {
      return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(embeddedInput));
    } catch (NoSuchAlgorithmException impossible) {
      // Every conforming JRE ships SHA-256.
      throw new IllegalStateException("SHA-256 unavailable", impossible);
    }
  }

  /**
   * Validates a caller-supplied hash.
   *
   * <p>Required because a document whose source is not stored must supply its own hash, and a
   * malformed one there would silently defeat every staleness check downstream.
   */
  public static String validated(String hash) {
    Objects.requireNonNull(hash, "contentHash");
    if (hash.length() != LENGTH) {
      throw new IllegalArgumentException(
          "contentHash must be " + LENGTH + " lowercase hex characters, got " + hash.length());
    }
    for (int index = 0; index < LENGTH; index++) {
      char character = hash.charAt(index);
      boolean hex =
          (character >= '0' && character <= '9') || (character >= 'a' && character <= 'f');
      if (!hex) {
        throw new IllegalArgumentException(
            "contentHash must be lowercase hex; found '" + character + "' at " + index);
      }
    }
    return hash;
  }
}
