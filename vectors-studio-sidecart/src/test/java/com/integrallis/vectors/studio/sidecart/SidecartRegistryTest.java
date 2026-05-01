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

import static org.assertj.core.api.Assertions.assertThat;

import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.jupiter.api.Test;

class SidecartRegistryTest {

  @Test
  void emptyRegistryReturnsEmptyForAnyCollection() {
    SidecartRegistry r = SidecartRegistry.empty();
    assertThat(r.isEmpty()).isTrue();
    assertThat(r.get("anything")).isEmpty();
  }

  @Test
  void bindsAndResolvesByCollectionName() {
    SidecartSource fixed = id -> Optional.of(SidecartRecord.ofText("x", "text/plain"));
    SidecartRegistry r = SidecartRegistry.empty().bind("docs", fixed);
    assertThat(r.get("docs")).containsSame(fixed);
    assertThat(r.get("other")).isEmpty();
  }

  @Test
  void rebindClosesPriorSource() {
    AtomicBoolean closed = new AtomicBoolean();
    SidecartSource first =
        new SidecartSource() {
          @Override
          public Optional<SidecartRecord> get(String id) {
            return Optional.empty();
          }

          @Override
          public void close() {
            closed.set(true);
          }
        };
    SidecartSource second = id -> Optional.empty();

    SidecartRegistry r = SidecartRegistry.empty().bind("docs", first).bind("docs", second);

    assertThat(closed.get()).isTrue();
    assertThat(r.get("docs")).containsSame(second);
  }

  @Test
  void closeReleasesAllSources() {
    AtomicBoolean a = new AtomicBoolean();
    AtomicBoolean b = new AtomicBoolean();
    SidecartSource srcA = closingSource(a);
    SidecartSource srcB = closingSource(b);
    SidecartRegistry r = SidecartRegistry.empty().bind("a", srcA).bind("b", srcB);

    r.close();

    assertThat(a.get()).isTrue();
    assertThat(b.get()).isTrue();
    assertThat(r.isEmpty()).isTrue();
  }

  private static SidecartSource closingSource(AtomicBoolean flag) {
    return new SidecartSource() {
      @Override
      public Optional<SidecartRecord> get(String id) {
        return Optional.empty();
      }

      @Override
      public void close() {
        flag.set(true);
      }
    };
  }
}
