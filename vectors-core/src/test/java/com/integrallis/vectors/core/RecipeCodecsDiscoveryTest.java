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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.io.IOException;
import java.net.URLClassLoader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.api.parallel.Isolated;

@Isolated("Changes the recipe-codec property and thread context class loader")
public class RecipeCodecsDiscoveryTest {
  @TempDir Path temporary;

  @Test
  void emptyClasspathUsesBuiltinIncludingBlankProperty() throws Exception {
    withProviders(
        new Class<?>[0],
        () -> {
          assertThat(RecipeCodecs.discover()).isSameAs(RecipeCodecs.builtin());
          System.setProperty(RecipeCodecs.CODEC_PROPERTY, "   ");
          assertThat(RecipeCodecs.discover()).isSameAs(RecipeCodecs.builtin());
        });
  }

  @Test
  void singleProviderIsSelectedThroughRealServiceLoader() throws Exception {
    withProviders(
        new Class<?>[] {FirstCodec.class},
        () -> assertThat(RecipeCodecs.discover()).isInstanceOf(FirstCodec.class));
  }

  @Test
  void ambiguityIsRejectedAndExplicitChoiceWinsIndependentlyOfOrder() throws Exception {
    withProviders(
        new Class<?>[] {SecondCodec.class, FirstCodec.class},
        () -> {
          assertThatThrownBy(RecipeCodecs::discover)
              .isInstanceOf(IllegalStateException.class)
              .hasMessageContaining(FirstCodec.class.getName())
              .hasMessageContaining(SecondCodec.class.getName());
          System.setProperty(RecipeCodecs.CODEC_PROPERTY, FirstCodec.class.getName());
          assertThat(RecipeCodecs.discover()).isInstanceOf(FirstCodec.class);
          System.setProperty(RecipeCodecs.CODEC_PROPERTY, BuiltinRecipeCodec.class.getName());
          assertThat(RecipeCodecs.discover()).isSameAs(RecipeCodecs.builtin());
          System.setProperty(RecipeCodecs.CODEC_PROPERTY, "missing.Codec");
          assertThatThrownBy(RecipeCodecs::discover)
              .isInstanceOf(IllegalStateException.class)
              .hasMessageContaining("missing.Codec")
              .hasMessageContaining(FirstCodec.class.getName())
              .hasMessageContaining(SecondCodec.class.getName())
              .hasMessageContaining(BuiltinRecipeCodec.class.getName());
        });
  }

  private void withProviders(Class<?>[] providers, Runnable assertion) throws Exception {
    Path descriptor = temporary.resolve("META-INF/services/" + RecipeCodec.class.getName());
    Files.createDirectories(descriptor.getParent());
    Files.writeString(
        descriptor, Arrays.stream(providers).map(Class::getName).collect(Collectors.joining("\n")));
    ClassLoader previousLoader = Thread.currentThread().getContextClassLoader();
    String previousProperty = System.getProperty(RecipeCodecs.CODEC_PROPERTY);
    try (URLClassLoader loader =
        new URLClassLoader(
            new java.net.URL[] {temporary.toUri().toURL()}, RecipeCodec.class.getClassLoader())) {
      Thread.currentThread().setContextClassLoader(loader);
      System.clearProperty(RecipeCodecs.CODEC_PROPERTY);
      assertion.run();
    } finally {
      Thread.currentThread().setContextClassLoader(previousLoader);
      if (previousProperty == null) System.clearProperty(RecipeCodecs.CODEC_PROPERTY);
      else System.setProperty(RecipeCodecs.CODEC_PROPERTY, previousProperty);
    }
  }

  public static class FirstCodec implements RecipeCodec {
    public FirstCodec() {}

    @Override
    public int schemaVersion() {
      return RecipeCodecs.builtin().schemaVersion();
    }

    @Override
    public String encode(EmbeddingRecipe recipe) throws IOException {
      return RecipeCodecs.builtin().encode(recipe);
    }

    @Override
    public EmbeddingRecipe decode(String text) throws IOException {
      return RecipeCodecs.builtin().decode(text);
    }

    @Override
    public int schemaVersionOf(String text) throws IOException {
      return RecipeCodecs.builtin().schemaVersionOf(text);
    }
  }

  public static class SecondCodec extends FirstCodec {
    public SecondCodec() {}
  }
}
