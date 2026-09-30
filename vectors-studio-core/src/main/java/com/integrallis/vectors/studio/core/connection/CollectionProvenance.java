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

import com.integrallis.vectors.core.EmbeddingRecipe;
import java.util.Optional;

/**
 * What Studio can honestly say about where a collection's vectors came from.
 *
 * <p>Three states, kept distinct on purpose. Collapsing them would hide exactly the decision a user
 * needs to see: whether the numbers in front of them can be trusted to have come from the model
 * they think, and whether the collection can tell which of its vectors are stale.
 *
 * <p>{@link Level#UNKNOWN} is not an error and must not be styled as one — a collection used as
 * plain vector storage is a legitimate choice. It is, however, a collection whose vectors cannot be
 * attributed or re-derived, and a user deserves to know that before drawing a conclusion from a
 * ranking.
 */
public record CollectionProvenance(
    Level level, Optional<String> model, Optional<String> version, Optional<String> digest) {

  /** How much is known about the model behind the vectors. */
  public enum Level {
    /** A recipe with a weights digest: a stored vector can be re-derived and compared. */
    ATTESTED,
    /** A recipe naming a model, but no digest: a swap behind that name would go unnoticed. */
    DECLARED,
    /** No recipe. Vectors were supplied from outside and cannot be attributed or checked. */
    UNKNOWN
  }

  /** Reads provenance off a collection's recipe, if it has one. */
  public static CollectionProvenance of(Optional<EmbeddingRecipe> recipe) {
    return recipe
        .map(
            r ->
                new CollectionProvenance(
                    r.attestation() == EmbeddingRecipe.Attestation.ATTESTED
                        ? Level.ATTESTED
                        : Level.DECLARED,
                    Optional.of(r.modelId()),
                    Optional.of(r.modelVersion()),
                    r.modelDigest()))
        .orElseGet(CollectionProvenance::unknown);
  }

  /** A collection with no recipe. */
  public static CollectionProvenance unknown() {
    return new CollectionProvenance(
        Level.UNKNOWN, Optional.empty(), Optional.empty(), Optional.empty());
  }

  /** Short label for a UI pill. */
  public String label() {
    return switch (level) {
      case ATTESTED, DECLARED -> model.orElse("unknown model");
      case UNKNOWN -> "no recipe";
    };
  }

  /**
   * The sentence a user should read on hover, written to state the consequence rather than the
   * state.
   *
   * <p>"DECLARED" tells a user nothing; "a different model behind this name would not be detected"
   * tells them what to do about it.
   */
  public String explanation() {
    return switch (level) {
      case ATTESTED ->
          "Attested: the recipe pins this model by content digest ("
              + digest.map(d -> d.substring(0, Math.min(12, d.length())) + "…").orElse("?")
              + "), so a stored vector can be re-derived and compared bit for bit.";
      case DECLARED ->
          "Declared: the recipe names "
              + model.orElse("a model")
              + " "
              + version.orElse("")
              + " but records no content digest, so a different model served behind that name would"
              + " not be detected.";
      case UNKNOWN ->
          "No embedding recipe. These vectors were supplied from outside, so Studio cannot say which"
              + " model produced them, cannot tell which are stale, and cannot re-derive them. This is"
              + " a valid way to use a collection — it just cannot be audited.";
    };
  }

  /** CSS modifier, so the three states are visually distinct rather than three shades of grey. */
  public String cssClass() {
    return switch (level) {
      case ATTESTED -> "pill-provenance-attested";
      case DECLARED -> "pill-provenance-declared";
      case UNKNOWN -> "pill-provenance-unknown";
    };
  }
}
