package com.integrallis.vectors.db.id;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatIllegalArgumentException;
import static org.assertj.core.api.Assertions.assertThatNullPointerException;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class InMemoryIdMapperTest {

  @Test
  void putAssignsSequentialOrdinals() {
    InMemoryIdMapper m = new InMemoryIdMapper();
    assertThat(m.put("a")).isZero();
    assertThat(m.put("b")).isEqualTo(1);
    assertThat(m.put("c")).isEqualTo(2);
    assertThat(m.size()).isEqualTo(3);
  }

  @Test
  void putDuplicateThrows() {
    InMemoryIdMapper m = new InMemoryIdMapper();
    m.put("a");
    assertThatIllegalArgumentException().isThrownBy(() -> m.put("a")).withMessageContaining("a");
  }

  @Test
  void copyOfProducesIndependentSuccessor() {
    InMemoryIdMapper src = new InMemoryIdMapper();
    src.put("a");
    src.put("b");

    InMemoryIdMapper copy = InMemoryIdMapper.copyOf(src);
    assertThat(copy.size()).isEqualTo(2);
    assertThat(copy.contains("a")).isTrue();
    assertThat(copy.contains("b")).isTrue();
    assertThat(copy.ordinalOf("a")).isZero();
    assertThat(copy.ordinalOf("b")).isEqualTo(1);

    // Mutating the copy must not affect the source.
    copy.put("c");
    assertThat(copy.size()).isEqualTo(3);
    assertThat(src.size()).isEqualTo(2);
    assertThat(src.contains("c")).isFalse();

    // Mutating the source must not affect the copy.
    src.put("d");
    assertThat(src.size()).isEqualTo(3);
    assertThat(copy.contains("d")).isFalse();
  }

  @Test
  void copyOfEmptyMapperIsEmpty() {
    InMemoryIdMapper src = new InMemoryIdMapper();
    InMemoryIdMapper copy = InMemoryIdMapper.copyOf(src);
    assertThat(copy.size()).isZero();
    copy.put("a");
    assertThat(copy.ordinalOf("a")).isZero();
  }

  @Test
  void copyOfNullThrows() {
    assertThatNullPointerException().isThrownBy(() -> InMemoryIdMapper.copyOf(null));
  }
}
