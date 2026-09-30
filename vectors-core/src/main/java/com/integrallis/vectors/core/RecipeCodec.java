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

import java.io.IOException;

/**
 * Serialises an {@link EmbeddingRecipe} to and from the text stored in {@code recipe.json}.
 *
 * <p>Pluggable through {@link java.util.ServiceLoader} so an application can use the JSON stack it
 * already standardises on rather than inheriting ours. A built-in dependency-free codec is the
 * default; {@code vectors-db-jackson} provides a Jackson binding, and a custom implementation needs
 * only these three methods.
 *
 * <h2>Why swapping this is safe</h2>
 *
 * <p>{@link EmbeddingRecipe#recipeHash()} is computed from a <b>canonical field rendering</b>, not
 * from the serialised text. So the identity anchored in the manifest is independent of which codec
 * wrote the file: a collection written with one codec verifies under another, and changing codecs
 * cannot invalidate stored vectors or break attestation. Only the file's textual shape changes.
 *
 * <p>That is the whole reason this is a safe extension point rather than a format fork.
 *
 * <h2>What an implementation owes</h2>
 *
 * <ul>
 *   <li><b>Round-trip fidelity.</b> {@code decode(encode(r)).equals(r)} for every recipe, including
 *       absent prefixes, an absent digest and an empty {@code extra} map. A field silently dropped
 *       here becomes a recipe that cannot be reproduced.
 *   <li><b>Determinism.</b> The same recipe must encode to the same bytes every time — on any host
 *       and in any JVM. The file is diffed between environments and read by auditors; bytes that
 *       wobble with map iteration order or the platform line separator defeat both. Sort map keys,
 *       and do not use a platform-dependent line separator.
 *   <li><b>Every field written, including nulls.</b> An omitted key cannot be distinguished from a
 *       writer that did not know the field, and that distinction is what separates attested
 *       provenance from declared.
 *   <li><b>Tolerating unknown fields on decode.</b> A newer writer may have added some; failing
 *       would turn a forward-compatible file into an unopenable collection. Attestation is withheld
 *       elsewhere, based on {@link #schemaVersionOf}.
 * </ul>
 *
 * <p>{@code RecipeCodecContractTest} in {@code vectors-db} exercises all four against any codec.
 */
public interface RecipeCodec {

  /** Schema version this codec writes. */
  int schemaVersion();

  /** Encodes a recipe. Must be deterministic; see the class documentation. */
  String encode(EmbeddingRecipe recipe) throws IOException;

  /** Decodes a recipe, tolerating fields it does not recognise. */
  EmbeddingRecipe decode(String text) throws IOException;

  /**
   * Reads only the schema version, which decides whether this build may attest to the recipe.
   *
   * <p>Separate from {@link #decode} because it must succeed on a document whose other fields this
   * build cannot fully interpret.
   */
  int schemaVersionOf(String text) throws IOException;
}
