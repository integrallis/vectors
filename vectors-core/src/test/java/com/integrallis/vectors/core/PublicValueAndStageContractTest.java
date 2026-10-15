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

import static org.junit.jupiter.api.Assertions.*;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

/** Value-object validation only; no fabricated capabilities are installed or executed. */
class PublicValueAndStageContractTest {
  @Test
  void factoriesPreserveTypedValuesAndDocumentedDoublePrecision() {
    Map<String, MetadataValue> m =
        Map.of(
            "s",
            MetadataValue.of("alpha"),
            "i",
            MetadataValue.of(42L),
            "f",
            MetadataValue.of(1.25),
            "b",
            MetadataValue.of(true),
            "t",
            MetadataValue.tags("a", "b"));
    assertEquals(new MetadataValue.Str("alpha"), m.get("s"));
    assertEquals(new MetadataValue.Num(42), m.get("i"));
    assertEquals(new MetadataValue.Num(1.25), m.get("f"));
    assertEquals(new MetadataValue.Bool(true), m.get("b"));
    assertEquals(new MetadataValue.Tags(List.of("a", "b")), m.get("t"));
    long n = (1L << 53) + 1;
    assertEquals((double) n, MetadataValue.Num.of(n).value());
    assertEquals(
        Double.doubleToRawLongBits(-0.0),
        Double.doubleToRawLongBits(MetadataValue.Num.of(-0.0).value()));
    assertNotEquals(MetadataValue.of(-0.0), MetadataValue.of(0.0));
    assertFalse(((MetadataValue.Bool) MetadataValue.of(false)).value());
  }

  @Test
  void tagFactoriesOwnInputAndExposeImmutableValues() {
    String[] a = {"a", "b"};
    MetadataValue.Tags tags = (MetadataValue.Tags) MetadataValue.tags(a);
    a[0] = "changed";
    assertEquals(List.of("a", "b"), tags.values());
    assertThrows(UnsupportedOperationException.class, () -> tags.values().add("c"));
    List<String> values = new ArrayList<>(List.of("owned"));
    MetadataValue.Tags copied = new MetadataValue.Tags(values);
    values.clear();
    assertEquals(List.of("owned"), copied.values());
    assertEquals(List.of(), ((MetadataValue.Tags) MetadataValue.tags()).values());
    assertThrows(NullPointerException.class, () -> MetadataValue.of((String) null));
    assertThrows(NullPointerException.class, () -> MetadataValue.tags((String[]) null));
    assertThrows(NullPointerException.class, () -> MetadataValue.tags("a", null));
  }

  @Test
  void legacyCapabilityConstructorPreservesFactsAndIntegerKernelDefault() {
    var a = VectorUtil.runtimeCapabilities();
    var legacy =
        new VectorRuntimeCapabilities(
            "  " + a.providerName() + "  ",
            a.vectorApi(),
            a.configuredMaxVectorBits(),
            a.preferredVectorBits(),
            a.activeVectorBits(),
            a.fastVectorFma(),
            a.fastScalarFma(),
            a.sve(),
            a.q4ShortPairwiseSupported(),
            a.q4UnsignedPairwiseSupported(),
            a.ggufParallel(),
            "  " + a.ggufExecutor() + "  ",
            a.ggufThreads(),
            a.ggufChunksPerThread());
    var expected =
        cap(
            a.providerName(),
            a.vectorApi(),
            a.configuredMaxVectorBits(),
            a.preferredVectorBits(),
            a.activeVectorBits(),
            a.ggufExecutor(),
            a.ggufThreads(),
            a.ggufChunksPerThread(),
            "integer");
    assertEquals(expected, legacy);
    assertEquals(expected.hashCode(), legacy.hashCode());
  }

  @Test
  void capabilityValuesRejectAmbiguousAndContradictoryReports() {
    var a = VectorUtil.runtimeCapabilities();
    for (String bad : Arrays.asList(null, "", " ")) {
      assertThrows(
          IllegalArgumentException.class,
          () ->
              cap(
                  bad,
                  a.vectorApi(),
                  a.configuredMaxVectorBits(),
                  a.preferredVectorBits(),
                  a.activeVectorBits(),
                  a.ggufExecutor(),
                  a.ggufThreads(),
                  a.ggufChunksPerThread(),
                  "integer"));
      assertThrows(
          IllegalArgumentException.class,
          () ->
              cap(
                  a.providerName(),
                  a.vectorApi(),
                  a.configuredMaxVectorBits(),
                  a.preferredVectorBits(),
                  a.activeVectorBits(),
                  bad,
                  a.ggufThreads(),
                  a.ggufChunksPerThread(),
                  "integer"));
      assertThrows(
          IllegalArgumentException.class,
          () ->
              cap(
                  a.providerName(),
                  a.vectorApi(),
                  a.configuredMaxVectorBits(),
                  a.preferredVectorBits(),
                  a.activeVectorBits(),
                  a.ggufExecutor(),
                  a.ggufThreads(),
                  a.ggufChunksPerThread(),
                  bad));
    }
    for (int bad : new int[] {0, -1}) {
      assertThrows(
          IllegalArgumentException.class,
          () ->
              cap(
                  a.providerName(),
                  a.vectorApi(),
                  bad,
                  a.preferredVectorBits(),
                  a.activeVectorBits(),
                  a.ggufExecutor(),
                  a.ggufThreads(),
                  a.ggufChunksPerThread(),
                  "integer"));
      assertThrows(
          IllegalArgumentException.class,
          () ->
              cap(
                  a.providerName(),
                  a.vectorApi(),
                  a.configuredMaxVectorBits(),
                  bad,
                  a.activeVectorBits(),
                  a.ggufExecutor(),
                  a.ggufThreads(),
                  a.ggufChunksPerThread(),
                  "integer"));
      assertThrows(
          IllegalArgumentException.class,
          () ->
              cap(
                  a.providerName(),
                  a.vectorApi(),
                  a.configuredMaxVectorBits(),
                  a.preferredVectorBits(),
                  a.activeVectorBits(),
                  a.ggufExecutor(),
                  bad,
                  a.ggufChunksPerThread(),
                  "integer"));
      assertThrows(
          IllegalArgumentException.class,
          () ->
              cap(
                  a.providerName(),
                  a.vectorApi(),
                  a.configuredMaxVectorBits(),
                  a.preferredVectorBits(),
                  a.activeVectorBits(),
                  a.ggufExecutor(),
                  a.ggufThreads(),
                  bad,
                  "integer"));
    }
    assertThrows(
        IllegalArgumentException.class,
        () ->
            cap(
                a.providerName(),
                a.vectorApi(),
                a.configuredMaxVectorBits(),
                a.preferredVectorBits(),
                -1,
                a.ggufExecutor(),
                a.ggufThreads(),
                a.ggufChunksPerThread(),
                "integer"));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            cap(
                a.providerName(),
                true,
                a.configuredMaxVectorBits(),
                a.preferredVectorBits(),
                0,
                a.ggufExecutor(),
                a.ggufThreads(),
                a.ggufChunksPerThread(),
                "integer"));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            cap(
                a.providerName(),
                false,
                a.configuredMaxVectorBits(),
                a.preferredVectorBits(),
                a.preferredVectorBits(),
                a.ggufExecutor(),
                a.ggufThreads(),
                a.ggufChunksPerThread(),
                "integer"));
  }

  private static VectorRuntimeCapabilities cap(
      String p,
      boolean v,
      int max,
      int preferred,
      int active,
      String ex,
      int threads,
      int chunks,
      String kernel) {
    var a = VectorUtil.runtimeCapabilities();
    return new VectorRuntimeCapabilities(
        p,
        v,
        max,
        preferred,
        active,
        a.fastVectorFma(),
        a.fastScalarFma(),
        a.sve(),
        a.q4ShortPairwiseSupported(),
        a.q4UnsignedPairwiseSupported(),
        a.ggufParallel(),
        ex,
        threads,
        chunks,
        kernel);
  }

  @Test
  void recipesRejectNonpositiveDimensionsAndTokenBudgets() {
    for (int bad : new int[] {0, -1, Integer.MIN_VALUE}) {
      assertThrows(IllegalArgumentException.class, () -> recipe(bad, 512));
      assertThrows(IllegalArgumentException.class, () -> recipe(8, bad));
    }
    assertThrows(
        IllegalArgumentException.class,
        () -> EmbeddingRecipe.declared(" ", "v1", 8, SimilarityFunction.COSINE));
    var one = recipe(1, 1);
    assertEquals(1, one.dimension());
    assertEquals(1, one.maxInputTokens());
    assertEquals(Map.of(), one.extra());
    assertEquals("input", one.documentInput("input"));
    assertEquals("input", one.queryInput("input"));
  }

  private static EmbeddingRecipe recipe(int d, int t) {
    return new EmbeddingRecipe(
        "model",
        "v1",
        Optional.empty(),
        d,
        SimilarityFunction.COSINE,
        true,
        EmbeddingRecipe.Pooling.MEAN,
        Optional.empty(),
        Optional.empty(),
        t,
        EmbeddingRecipe.Truncation.REJECT,
        null);
  }

  @Test
  void publicStagePlanOwnsStagesAndPreservesOrderWhenReused() {
    AtomicInteger count = new AtomicInteger();
    GgufStagePlan.Stage[] stages = {
      GgufStagePlan.stage(
          1,
          (a, b) -> {
            assertEquals(0, a);
            assertEquals(1, b);
            assertEquals(0, count.getAndIncrement());
          }),
      GgufStagePlan.stage(
          1,
          (a, b) -> {
            assertEquals(0, a);
            assertEquals(1, b);
            assertEquals(1, count.getAndIncrement());
          })
    };
    var plan = GgufStagePlan.of(stages);
    stages[0] = GgufStagePlan.stage(1, (a, b) -> fail("caller changed immutable plan"));
    assertEquals(2, plan.stageCount());
    plan.execute();
    assertEquals(2, count.get());
    count.set(0);
    plan.execute();
    assertEquals(2, count.get());
  }

  @Test
  void invalidStagesFailBeforeCallbacksAndStageFailureStopsFollowingStages() {
    AtomicInteger calls = new AtomicInteger();
    assertThrows(NullPointerException.class, () -> GgufStagePlan.of((GgufStagePlan.Stage[]) null));
    assertThrows(IllegalArgumentException.class, GgufStagePlan::of);
    assertThrows(NullPointerException.class, () -> GgufStagePlan.of((GgufStagePlan.Stage) null));
    assertThrows(
        IllegalArgumentException.class,
        () -> GgufStagePlan.stage(0, (a, b) -> calls.incrementAndGet()));
    assertThrows(
        IllegalArgumentException.class,
        () -> GgufStagePlan.stage(-1, (a, b) -> calls.incrementAndGet()));
    assertThrows(NullPointerException.class, () -> GgufStagePlan.stage(1, null));
    RuntimeException failure = new RuntimeException("stage failed");
    var plan =
        GgufStagePlan.of(
            GgufStagePlan.stage(
                1,
                (a, b) -> {
                  calls.incrementAndGet();
                  throw failure;
                }),
            GgufStagePlan.stage(1, (a, b) -> fail("ran after failed stage")));
    assertSame(failure, assertThrows(RuntimeException.class, plan::execute));
    assertEquals(1, calls.get());
  }
}
