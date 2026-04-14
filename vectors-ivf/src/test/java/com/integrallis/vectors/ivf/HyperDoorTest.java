package com.integrallis.vectors.ivf;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class HyperDoorTest {

  @Test
  void strideArithmetic_t0BitOffset_linearInOrdinal() {
    int dim = 128;
    HyperDoor d0 = HyperDoor.full(0, dim);
    HyperDoor d1 = HyperDoor.full(1, dim);
    HyperDoor d5 = HyperDoor.full(5, dim);

    assertThat(d0.t0BitOffset()).isEqualTo(0L);
    assertThat(d1.t0BitOffset()).isEqualTo((long) dim);
    assertThat(d5.t0BitOffset()).isEqualTo((long) 5 * dim);
  }

  @Test
  void strideArithmetic_t1ByteOffset_linearInOrdinal() {
    int dim = 128;
    HyperDoor d3 = HyperDoor.full(3, dim);
    assertThat(d3.t1ByteOffset()).isEqualTo(3 * dim);
  }

  @Test
  void strideArithmetic_t2FileOffset_linearInOrdinal() {
    int dim = 128;
    HyperDoor d2 = HyperDoor.full(2, dim);
    assertThat(d2.t2FileOffset()).isEqualTo((long) 2 * dim * Float.BYTES);
  }

  @Test
  void strideArithmetic_t3ObjectOffset_linearInOrdinal() {
    int dim = 64;
    HyperDoor d4 = HyperDoor.full(4, dim);
    assertThat(d4.t3ObjectOffset()).isEqualTo((long) 4 * dim * Float.BYTES);
  }

  @Test
  void fullDoor_hasBothT1AndT2() {
    HyperDoor door = HyperDoor.full(0, 128);
    assertThat(door.hasT1()).isTrue();
    assertThat(door.hasT2()).isTrue();
  }

  @Test
  void t0AndT3Only_missingT1AndT2() {
    HyperDoor door = HyperDoor.t0AndT3Only(0, 128);
    assertThat(door.hasT1()).isFalse();
    assertThat(door.hasT2()).isFalse();
    assertThat(door.t1ByteOffset()).isEqualTo(-1);
    assertThat(door.t2FileOffset()).isEqualTo(-1L);
  }

  @Test
  void t0AlwaysPresent_evenForT0AndT3Only() {
    HyperDoor door = HyperDoor.t0AndT3Only(7, 64);
    assertThat(door.t0BitOffset()).isGreaterThanOrEqualTo(0L);
    assertThat(door.t3ObjectOffset()).isGreaterThanOrEqualTo(0L);
  }

  @Test
  void clusterOrdinalPreserved() {
    HyperDoor door = HyperDoor.full(42, 128);
    assertThat(door.clusterOrdinal()).isEqualTo(42);
  }
}
