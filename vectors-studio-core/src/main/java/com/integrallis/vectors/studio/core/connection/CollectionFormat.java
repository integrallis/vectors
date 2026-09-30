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
package com.integrallis.vectors.studio.core.connection;

/**
 * The on-disk manifest format a collection was written with, relative to what this build writes.
 *
 * <p>Shown because the version alone tells a user nothing. "Format 4" is only meaningful next to
 * the version in front of it, and what they actually need to know is which of three situations they
 * are in: nothing to do, a migration is available, or this Studio is too old to open the
 * collection. {@link Status} is that answer, and the two numbers are the evidence for it.
 *
 * <p>{@link Status#OLDER} is not a fault. A collection written by an earlier build is intact and
 * its vectors are unchanged; only the manifest header grew. It is offered as a migration rather
 * than an error because the upgrade rewrites a few hundred bytes per generation and re-embeds
 * nothing.
 */
public record CollectionFormat(int version, int current, Status status) {

  /** Where a collection's format sits relative to this build. */
  public enum Status {
    /** Written by an older build. Readable after an in-place manifest migration. */
    OLDER,
    /** Written by this format version. Nothing to do. */
    CURRENT,
    /** Written by a newer build. This one cannot read it and must be upgraded. */
    NEWER,
    /** The manifest could not be read, so the format is genuinely not known. */
    UNKNOWN
  }

  /** Compact constructor. */
  public CollectionFormat {
    if (status == null) {
      status = Status.UNKNOWN;
    }
  }

  /**
   * Classifies a version found on disk against the version this build writes.
   *
   * @param version the manifest version read from the collection, or a non-positive value if it
   *     could not be read
   * @param current the manifest version this build writes
   * @return the classified format
   */
  public static CollectionFormat of(int version, int current) {
    if (version <= 0) {
      return new CollectionFormat(version, current, Status.UNKNOWN);
    }
    Status status =
        version == current ? Status.CURRENT : (version < current ? Status.OLDER : Status.NEWER);
    return new CollectionFormat(version, current, status);
  }

  /** A collection whose format could not be determined. */
  public static CollectionFormat unknown(int current) {
    return new CollectionFormat(-1, current, Status.UNKNOWN);
  }

  /**
   * Whether an in-place migration would make this collection readable here.
   *
   * @return true only when the collection is older than this build
   */
  public boolean migratable() {
    return status == Status.OLDER;
  }

  /**
   * A short label for display, e.g. {@code "v5 (current)"} or {@code "v4 (older)"}.
   *
   * @return the label
   */
  public String label() {
    if (status == Status.UNKNOWN) {
      return "format unknown";
    }
    return "v"
        + version
        + (status == Status.CURRENT
            ? ""
            : " (" + status.name().toLowerCase(java.util.Locale.ROOT) + ")");
  }

  /**
   * What this format means for the user, and what to do about it.
   *
   * @return a sentence for a tooltip
   */
  public String explanation() {
    return switch (status) {
      case CURRENT ->
          "Written with manifest format v" + version + ", which is what this build reads.";
      case OLDER ->
          "Written with manifest format v"
              + version
              + "; this build reads v"
              + current
              + ". The vectors are intact — only the manifest header changed — so an in-place"
              + " migration makes it readable without re-embedding anything.";
      case NEWER ->
          "Written with manifest format v"
              + version
              + "; this build only reads v"
              + current
              + ". Upgrade vectors to open it. It must not be downgraded, because fields this build"
              + " does not know about would be dropped.";
      case UNKNOWN ->
          "The manifest could not be read, so the format is not known. The directory may not be a"
              + " collection.";
    };
  }

  /**
   * CSS class for the display pill.
   *
   * @return the class name
   */
  public String cssClass() {
    return switch (status) {
      case CURRENT -> "pill-format-current";
      case OLDER -> "pill-format-older";
      case NEWER -> "pill-format-newer";
      case UNKNOWN -> "pill-format-unknown";
    };
  }
}
