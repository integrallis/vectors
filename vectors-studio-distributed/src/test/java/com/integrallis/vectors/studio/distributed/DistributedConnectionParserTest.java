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
package com.integrallis.vectors.studio.distributed;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.vectors.core.SimilarityFunction;
import org.junit.jupiter.api.Test;

class DistributedConnectionParserTest {

  @Test
  void happyPathParsesAllFields() {
    DistributedConnectionConfig cfg =
        DistributedConnectionParser.parse(
            "r2://my-bucket/demo/?wal=/tmp/wal&dim=384"
                + "&endpoint=https://acct.r2.cloudflarestorage.com"
                + "&region=auto&metric=COSINE&name=demo-corpus&t1=10&t2=3",
            "AK",
            "SK");
    assertThat(cfg.collectionName()).isEqualTo("demo-corpus");
    assertThat(cfg.s3Bucket()).isEqualTo("my-bucket");
    assertThat(cfg.s3Prefix()).isEqualTo("demo/");
    assertThat(cfg.s3Endpoint()).isEqualTo("https://acct.r2.cloudflarestorage.com");
    assertThat(cfg.s3Region()).isEqualTo("auto");
    assertThat(cfg.s3AccessKey()).isEqualTo("AK");
    assertThat(cfg.s3SecretKey()).isEqualTo("SK");
    assertThat(cfg.dim()).isEqualTo(384);
    assertThat(cfg.metric()).isEqualTo(SimilarityFunction.COSINE);
    assertThat(cfg.walDir().toString()).isEqualTo("/tmp/wal");
    assertThat(cfg.tierPolicy().t1Threshold()).isEqualTo(10);
    assertThat(cfg.tierPolicy().t2Threshold()).isEqualTo(3);
  }

  @Test
  void defaultsRegionMetricAndName() {
    DistributedConnectionConfig cfg =
        DistributedConnectionParser.parse(
            "r2://demo-bucket/?wal=/tmp/wal&dim=8&endpoint=https://x.r2.cloudflarestorage.com",
            "k",
            "s");
    assertThat(cfg.s3Region()).isEqualTo("auto");
    assertThat(cfg.metric()).isEqualTo(SimilarityFunction.COSINE);
    assertThat(cfg.collectionName()).isEqualTo("demo-bucket");
  }

  @Test
  void prefixWithoutTrailingSlashKeepsRawForm() {
    DistributedConnectionConfig cfg =
        DistributedConnectionParser.parse(
            "r2://b/sub/path?wal=/w&dim=4&endpoint=https://x", "k", "s");
    assertThat(cfg.s3Prefix()).isEqualTo("sub/path");
  }

  @Test
  void rejectsNonR2Url() {
    assertThatThrownBy(() -> DistributedConnectionParser.parse("http://example.com", "k", "s"))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void rejectsMissingWal() {
    assertThatThrownBy(
            () -> DistributedConnectionParser.parse("r2://b/?dim=4&endpoint=https://x", "k", "s"))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("wal");
  }

  @Test
  void rejectsMissingDim() {
    assertThatThrownBy(
            () -> DistributedConnectionParser.parse("r2://b/?wal=/w&endpoint=https://x", "k", "s"))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("dim");
  }

  @Test
  void rejectsMissingEndpoint() {
    assertThatThrownBy(() -> DistributedConnectionParser.parse("r2://b/?wal=/w&dim=4", "k", "s"))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("endpoint");
  }

  @Test
  void rejectsMissingBucket() {
    assertThatThrownBy(
            () ->
                DistributedConnectionParser.parse(
                    "r2://?wal=/w&dim=4&endpoint=https://x", "k", "s"))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("bucket");
  }

  @Test
  void rejectsNonNumericDim() {
    assertThatThrownBy(
            () ->
                DistributedConnectionParser.parse(
                    "r2://b/?wal=/w&dim=NaN&endpoint=https://x", "k", "s"))
        .isInstanceOf(IllegalArgumentException.class);
  }
}
