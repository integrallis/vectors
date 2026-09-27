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

import java.util.ArrayList;
import java.util.List;
import java.util.ServiceLoader;

/**
 * Selects the {@link RecipeCodec} in use.
 *
 * <p>Resolution order, highest first:
 *
 * <ol>
 *   <li>the codec passed explicitly to the API that needs one — always wins, because a test or a
 *       multi-tenant host must be able to be specific rather than depend on classpath order;
 *   <li>{@code -Dvectors.recipeCodec=<class name>}, for choosing between two codecs that are both
 *       present without repackaging;
 *   <li>a single {@link ServiceLoader} provider on the classpath;
 *   <li>{@link BuiltinRecipeCodec}.
 * </ol>
 *
 * <p><b>Two providers with no property set is an error, not a coin toss.</b> Silently picking one
 * by classpath order would mean a collection's file shape depended on jar ordering, which is the
 * kind of thing that works in a test and differs in production. The failure names both so the
 * choice can be made deliberately.
 *
 * <p>Note that this only decides the file's textual shape. {@link EmbeddingRecipe#recipeHash()} is
 * computed from a canonical field rendering, so identity and verification are unaffected by which
 * codec is in use — a collection written by one reads and verifies under another.
 */
public final class RecipeCodecs {

  /** System property naming the codec implementation class to use. */
  public static final String CODEC_PROPERTY = "vectors.recipeCodec";

  private static final RecipeCodec BUILTIN = new BuiltinRecipeCodec();

  private RecipeCodecs() {}

  /** The codec this JVM should use. */
  public static RecipeCodec discover() {
    String requested = System.getProperty(CODEC_PROPERTY);
    List<RecipeCodec> providers = new ArrayList<>();
    for (RecipeCodec codec : ServiceLoader.load(RecipeCodec.class)) {
      providers.add(codec);
    }
    if (requested != null && !requested.isBlank()) {
      for (RecipeCodec codec : providers) {
        if (codec.getClass().getName().equals(requested)) {
          return codec;
        }
      }
      if (BuiltinRecipeCodec.class.getName().equals(requested)) {
        return BUILTIN;
      }
      throw new IllegalStateException(
          CODEC_PROPERTY
              + '='
              + requested
              + " but no such RecipeCodec was found on the classpath. Available: "
              + providers.stream().map(c -> c.getClass().getName()).toList()
              + " plus the built-in "
              + BuiltinRecipeCodec.class.getName());
    }
    if (providers.size() > 1) {
      throw new IllegalStateException(
          "several RecipeCodec providers are on the classpath and none was chosen: "
              + providers.stream().map(c -> c.getClass().getName()).toList()
              + ". Set -D"
              + CODEC_PROPERTY
              + " to pick one. Choosing by classpath order would make a collection's file shape depend"
              + " on jar ordering.");
    }
    return providers.isEmpty() ? BUILTIN : providers.get(0);
  }

  /** The dependency-free default, for callers that want it regardless of the classpath. */
  public static RecipeCodec builtin() {
    return BUILTIN;
  }
}
