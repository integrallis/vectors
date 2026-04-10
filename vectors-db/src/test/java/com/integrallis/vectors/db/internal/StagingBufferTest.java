package com.integrallis.vectors.db.internal;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatNullPointerException;

import com.integrallis.vectors.db.Document;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class StagingBufferTest {

  private static Document doc(String id) {
    return Document.of(id, new float[] {1f, 2f, 3f, 4f});
  }

  @Test
  void emptyBufferReportsZero() {
    StagingBuffer buf = new StagingBuffer();
    assertThat(buf.size()).isZero();
    assertThat(buf.isEmpty()).isTrue();
    assertThat(buf.contains("a")).isFalse();
    assertThat(buf.documents()).isEmpty();
  }

  @Test
  void appendTracksSizeAndOrder() {
    StagingBuffer buf = new StagingBuffer();
    assertThat(buf.append(doc("a"))).isTrue();
    assertThat(buf.append(doc("b"))).isTrue();
    assertThat(buf.size()).isEqualTo(2);
    assertThat(buf.isEmpty()).isFalse();
    assertThat(buf.contains("a")).isTrue();
    assertThat(buf.contains("b")).isTrue();
    assertThat(buf.documents().get(0).id()).isEqualTo("a");
    assertThat(buf.documents().get(1).id()).isEqualTo("b");
  }

  @Test
  void appendDuplicateReturnsFalseAndDoesNotMutate() {
    StagingBuffer buf = new StagingBuffer();
    buf.append(doc("a"));
    assertThat(buf.append(doc("a"))).isFalse();
    assertThat(buf.size()).isEqualTo(1);
    assertThat(buf.documents()).hasSize(1);
  }

  @Test
  void clearWipesDocumentsAndIds() {
    StagingBuffer buf = new StagingBuffer();
    buf.append(doc("a"));
    buf.append(doc("b"));
    buf.clear();
    assertThat(buf.size()).isZero();
    assertThat(buf.isEmpty()).isTrue();
    assertThat(buf.contains("a")).isFalse();
    // After clear the same ids can be re-appended.
    assertThat(buf.append(doc("a"))).isTrue();
  }

  @Test
  void appendNullThrows() {
    StagingBuffer buf = new StagingBuffer();
    assertThatNullPointerException().isThrownBy(() -> buf.append(null));
  }
}
