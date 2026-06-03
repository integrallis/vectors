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
package com.integrallis.vectors.hybrid.text;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class TextIndexSpiFactoryTest {

  @Test
  void defaultPersistentCreateFailsFastWhenDataDirIsProvided(@TempDir Path dataDir) {
    TextIndexSpiFactory factory = new InMemoryOnlyFactory();

    assertThatThrownBy(() -> factory.create("collection", dataDir))
        .isInstanceOf(UnsupportedOperationException.class)
        .hasMessageContaining("does not implement persistent text indexes");
  }

  @Test
  void defaultPersistentCreateAllowsExplicitInMemoryMode() {
    TextIndexSpiFactory factory = new InMemoryOnlyFactory();

    TextIndexSpi index = factory.create("collection", null);

    assertThat(index).isInstanceOf(NoOpTextIndex.class);
  }

  private static final class InMemoryOnlyFactory implements TextIndexSpiFactory {
    @Override
    public TextIndexSpi create(String collectionName) {
      return new NoOpTextIndex();
    }
  }

  private static final class NoOpTextIndex implements TextIndexSpi {
    @Override
    public void index(List<TextDocument> documents) {}

    @Override
    public TextSearchOutcome search(String query, int k) {
      return TextSearchOutcome.empty();
    }

    @Override
    public Optional<StoredContent> get(String id) {
      return Optional.empty();
    }

    @Override
    public Optional<byte[]> getBlob(String id) {
      return Optional.empty();
    }

    @Override
    public void remove(String id) {}

    @Override
    public void clear() {}

    @Override
    public int size() {
      return 0;
    }

    @Override
    public void close() {}
  }
}
