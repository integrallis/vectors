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
package com.integrallis.vectors.bench;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.concurrent.TimeUnit;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.LongVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;

/**
 * Roofline probe: sequential main-memory read bandwidth with a trivial Panama loop.
 *
 * <p>One op is one full pass over a {@code sizeMb} off-heap segment (pre-touched, well beyond the
 * last-level cache). The per-byte compute is a single vector add, so a pass is bounded by how fast
 * memory streams. GB/s = {@code sizeMb / 1024 / (ms/op / 1000)}. Run with {@code -t N} for
 * multi-threaded bandwidth (each thread scans the whole segment; total = N x per-thread GB/s).
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
@Fork(
    value = 1,
    jvmArgsPrepend = {"--add-modules", "jdk.incubator.vector"})
@Warmup(iterations = 3, time = 2)
@Measurement(iterations = 5, time = 2)
public class MemoryBandwidthBenchmark {

  private static final VectorSpecies<Long> LONGS = LongVector.SPECIES_PREFERRED;
  private static final VectorSpecies<Byte> BYTES = ByteVector.SPECIES_PREFERRED;

  @Param({"2048"})
  int sizeMb;

  private Arena arena;
  private MemorySegment segment;
  private long vectorBytes;

  @Setup(Level.Trial)
  public void setUp() {
    long size = (long) sizeMb * 1024 * 1024;
    arena = Arena.ofShared();
    segment = arena.allocate(size, 64);
    long seed = 0x9E3779B97F4A7C15L;
    for (long offset = 0; offset + Long.BYTES <= size; offset += Long.BYTES) {
      seed ^= seed << 13;
      seed ^= seed >>> 7;
      seed ^= seed << 17;
      segment.set(ValueLayout.JAVA_LONG_UNALIGNED, offset, seed);
    }
    vectorBytes = size - size % LONGS.vectorByteSize();
    System.out.printf(
        "memory-bandwidth sizeMb=%d longLanes=%d byteLanes=%d%n",
        sizeMb, LONGS.length(), BYTES.length());
  }

  @TearDown(Level.Trial)
  public void tearDown() {
    arena.close();
  }

  /** 64-bit lanes: fewest ops per byte, closest to pure streaming. */
  @Benchmark
  public long sequentialLongSum() {
    LongVector acc = LongVector.zero(LONGS);
    int step = LONGS.vectorByteSize();
    for (long offset = 0; offset < vectorBytes; offset += step) {
      acc = acc.add(LongVector.fromMemorySegment(LONGS, segment, offset, ByteOrder.LITTLE_ENDIAN));
    }
    return acc.reduceLanes(VectorOperators.ADD);
  }

  /** Byte lanes: the element width the quantized dequantization loops read. */
  @Benchmark
  public long sequentialByteXor() {
    ByteVector acc = ByteVector.zero(BYTES);
    int step = BYTES.vectorByteSize();
    for (long offset = 0; offset < vectorBytes; offset += step) {
      acc =
          acc.lanewise(
              VectorOperators.XOR,
              ByteVector.fromMemorySegment(BYTES, segment, offset, ByteOrder.LITTLE_ENDIAN));
    }
    return acc.reduceLanes(VectorOperators.XOR);
  }
}
