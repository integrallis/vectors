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

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * What the topology probe is allowed to answer, on whatever machine runs the suite.
 *
 * <p>Asserted as invariants rather than as this machine's numbers: the suite runs on developer
 * laptops, CI containers and bench boxes with different core counts, and a test that hard-coded one
 * of them would fail everywhere else. The invariants are the part that must hold on all of them,
 * and the container clamp is the one that would otherwise go unnoticed until a two-CPU cgroup
 * started twenty-four workers.
 */
class CpuTopologyTest {

  @Test
  void computeThreadsNeverExceedsTheJvmProcessorBudget() {
    int available = Runtime.getRuntime().availableProcessors();

    // The clamp that matters: core topology is read from the host, so inside a CPU-limited
    // container the host's cores are not this process's budget.
    assertThat(CpuTopology.computeThreads()).isBetween(1, available);
  }

  @Test
  void computeThreadsIsPositiveEvenWhenDetectionFails() {
    assertThat(CpuTopology.computeThreads()).isPositive();
  }

  @Test
  void physicalCoresIsEitherUnknownOrPlausible() {
    int cores = CpuTopology.physicalCores();

    if (cores < 0) {
      // An unfamiliar platform is allowed to decline, and must then fall back cleanly.
      assertThat(CpuTopology.computeThreads())
          .isEqualTo(Math.max(1, Runtime.getRuntime().availableProcessors()));
      return;
    }
    // No machine has more physical cores than hardware threads.
    assertThat(cores).isPositive().isLessThanOrEqualTo(totalHardwareThreads());
  }

  @Test
  void smtIsReportedConsistentlyWithTheCoreCount() {
    if (CpuTopology.physicalCores() < 0) {
      assertThat(CpuTopology.simultaneousMultithreading()).isFalse();
      return;
    }
    assertThat(CpuTopology.simultaneousMultithreading())
        .isEqualTo(Runtime.getRuntime().availableProcessors() > CpuTopology.physicalCores());
  }

  /**
   * Hardware threads as the OS sees them, which on a container-limited JVM is larger than {@code
   * availableProcessors()} -- so the plausibility bound above must not use that.
   */
  private static int totalHardwareThreads() {
    int cores = CpuTopology.physicalCores();
    return Math.max(cores * 8, Runtime.getRuntime().availableProcessors());
  }

  /**
   * An Intel hybrid part as sysfs describes it, built rather than borrowed.
   *
   * <p>Shaped like an i9-14900HX: eight performance cores with two hardware threads each on cpus
   * 0-15, and sixteen single-threaded efficiency cores on cpus 16-31. The right answer is eight --
   * not 32 (hardware threads), not 24 (all physical cores), not 16 (performance threads). Each
   * wrong answer is a plausible one some other reading of this tree would produce, which is why the
   * fixture carries all three distinctions at once.
   */
  private static Path intelHybridSysfs(Path root) throws IOException {
    Path cpu = Files.createDirectories(root.resolve("cpu"));
    for (int logical = 0; logical < 32; logical++) {
      Path topology = Files.createDirectories(cpu.resolve("cpu" + logical).resolve("topology"));
      // cpus 0-15 are two threads of each of cores 0-7; cpus 16-31 are cores 8-23, one thread each.
      int core = logical < 16 ? logical / 2 : 8 + (logical - 16);
      Files.writeString(topology.resolve("core_id"), core + "\n");
      Files.writeString(topology.resolve("physical_package_id"), "0\n");
    }
    Path core = Files.createDirectories(cpu.resolve("types").resolve("intel_core_0"));
    Files.writeString(core.resolve("cpulist"), "0-15\n");
    Path atom = Files.createDirectories(cpu.resolve("types").resolve("intel_atom_0"));
    Files.writeString(atom.resolve("cpulist"), "16-31\n");
    return cpu;
  }

  @Test
  void countsPerformanceCoresOnAnIntelHybridTree(@TempDir Path root) throws IOException {
    Path cpu = intelHybridSysfs(root);

    // Eight: efficiency cores excluded, and the performance cores' SMT siblings folded away.
    assertThat(CpuTopology.linuxIntelPerformanceCpus(cpu)).isEqualTo(8);
  }

  @Test
  void declinesWhenTheTreeHasOnlyOneCoreClass(@TempDir Path root) throws IOException {
    Path cpu = intelHybridSysfs(root);
    // Remove the efficiency class: a machine with one class must answer "unknown" rather than
    // report its whole core count as though it were the fast subset.
    Files.delete(cpu.resolve("types").resolve("intel_atom_0").resolve("cpulist"));
    Files.delete(cpu.resolve("types").resolve("intel_atom_0"));

    assertThat(CpuTopology.linuxIntelPerformanceCpus(cpu)).isEqualTo(-1);
  }

  @Test
  void declinesWhenThereIsNoTypesDirectory(@TempDir Path root) throws IOException {
    Path cpu = Files.createDirectories(root.resolve("cpu"));

    assertThat(CpuTopology.linuxIntelPerformanceCpus(cpu)).isEqualTo(-1);
  }

  /**
   * The i9-14900HX exactly as its own kernel reports it, with no {@code types} directory.
   *
   * <p>Three frequency groups on purpose, because that machine has three: sixteen efficiency cores
   * at 4.1 GHz, twelve performance threads at 5.6 and four favoured ones at 5.8. A detector that
   * took the highest group would answer two.
   */
  private static Path i9HybridByFrequency(Path root) throws IOException {
    Path cpu = Files.createDirectories(root.resolve("cpu"));
    for (int logical = 0; logical < 32; logical++) {
      Path topology = Files.createDirectories(cpu.resolve("cpu" + logical).resolve("topology"));
      Path cpufreq = Files.createDirectories(cpu.resolve("cpu" + logical).resolve("cpufreq"));
      boolean performance = logical < 16;
      int core = performance ? (logical / 2) * 4 : 32 + (logical - 16);
      long ceiling =
          performance ? (logical >= 8 && logical <= 11 ? 5_800_000L : 5_600_000L) : 4_100_000L;
      Files.writeString(topology.resolve("core_id"), core + "\n");
      Files.writeString(topology.resolve("physical_package_id"), "0\n");
      Files.writeString(cpufreq.resolve("cpuinfo_max_freq"), ceiling + "\n");
    }
    return cpu;
  }

  @Test
  void countsPerformanceCoresByFrequencyWhenThereIsNoTypesDirectory(@TempDir Path root)
      throws IOException {
    Path cpu = i9HybridByFrequency(root);

    // Eight. Not 2 (the top frequency group), not 16 (performance threads), not 24 (all cores).
    assertThat(CpuTopology.linuxPerformanceCoresByMaxFrequency(cpu)).isEqualTo(8);
  }

  @Test
  void frequencyReadDeclinesOnAHomogeneousMachine(@TempDir Path root) throws IOException {
    Path cpu = Files.createDirectories(root.resolve("cpu"));
    for (int logical = 0; logical < 8; logical++) {
      Path topology = Files.createDirectories(cpu.resolve("cpu" + logical).resolve("topology"));
      Path cpufreq = Files.createDirectories(cpu.resolve("cpu" + logical).resolve("cpufreq"));
      Files.writeString(topology.resolve("core_id"), (logical / 2) + "\n");
      Files.writeString(topology.resolve("physical_package_id"), "0\n");
      Files.writeString(cpufreq.resolve("cpuinfo_max_freq"), "3600000\n");
    }

    // One frequency group is not a performance subset, and must not be reported as one.
    assertThat(CpuTopology.linuxPerformanceCoresByMaxFrequency(cpu)).isEqualTo(-1);
  }
}
