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

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.TimeUnit;

/**
 * How many threads this machine can actually run compute-bound kernel work on.
 *
 * <p>{@link Runtime#availableProcessors()} counts hardware threads, and a GGUF matmul does not get
 * faster for being given two threads that share one core's vector units. MEASURED on a CCX33 (8
 * vCPU, 4 physical cores): batched decision prefill halves from one thread to two and again to four
 * and then saturates -- at the physical core count, not the processor count. Sizing a pool to
 * {@code availableProcessors()} on any SMT machine therefore asks for roughly twice the threads
 * that can do work, and the surplus is not free: they contend for the same units and each one still
 * arrives at every barrier.
 *
 * <p>The count is clamped to {@code availableProcessors()} and that clamp is the load-bearing part.
 * Core topology is read from the host, and inside a container with a CPU limit the host's core
 * count is not this process's budget: a two-CPU cgroup on a 64-core machine must size to two.
 * Reading {@code /sys} alone would hand it thirty-two. Everything here is a floor-and-ceiling
 * around what the JVM already knows rather than a replacement for it.
 *
 * <p>Detection is best-effort by construction. An unreadable topology, an unfamiliar platform or a
 * surprising answer all fall back to {@code availableProcessors()}, because being wrong about the
 * hardware must cost a little parallelism and never a failure to start.
 */
final class CpuTopology {

  /**
   * Resolved once: this cannot change while the JVM runs, and probing costs a syscall or a fork.
   */
  private static final int PHYSICAL_CORES = detectPhysicalCores();

  /** Logical processors on the fastest class of core, or -1 when there is only one class. */
  private static final int PERFORMANCE_CORES = detectPerformanceCores();

  private CpuTopology() {}

  /**
   * Threads to run compute-bound kernel work on: physical cores, never more than the JVM's own
   * processor budget, never fewer than one.
   */
  static int computeThreads() {
    int available = Math.max(1, Runtime.getRuntime().availableProcessors());
    // Performance cores first, physical cores second, and the processor count only if neither is
    // knowable. The reference CPU implementation we measure against resolves the same two
    // quantities in the same order, and its reasons are the ones that apply here: it skips
    // efficiency cores because "efficiency cores harm lockstep threading", and skips SMT siblings
    // because "hyperthreading isn't useful for linear algebra" (llama.cpp,
    // common/common.cpp cpu_count_math_cpus). Both are statements about a barrier-synchronised
    // pool doing dense arithmetic, which is exactly this one, and our own sweep agrees on the
    // second: on a 16-core/32-thread EPYC, 16 threads beat 32.
    int preferred = PERFORMANCE_CORES > 0 ? PERFORMANCE_CORES : PHYSICAL_CORES;
    if (preferred < 1) {
      return available;
    }
    return Math.max(1, Math.min(preferred, available));
  }

  /** Physical cores as detected, or {@code -1} when this platform did not answer. */
  static int physicalCores() {
    return PHYSICAL_CORES;
  }

  /** Whether this machine reports more hardware threads than physical cores. */
  static boolean simultaneousMultithreading() {
    return PHYSICAL_CORES > 0 && Runtime.getRuntime().availableProcessors() > PHYSICAL_CORES;
  }

  /**
   * Logical processors on this machine's fastest class of core, or {@code -1} when it has only one
   * class or the platform does not say.
   *
   * <p>Reported and deliberately <b>not</b> used to size anything. A hybrid CPU -- Intel P and E
   * cores, ARM big.LITTLE -- breaks the assumption that a chunk costs the same wherever it lands,
   * and a barrier with equal chunks would wait for its slowest member every dispatch. But this
   * executor does not hand out equal chunks: it over-decomposes and lets workers steal, so a slow
   * core simply takes fewer of them. Whether the slow cores still cost more than they contribute is
   * a measurement, not something to reason out, so this exists to make that measurement possible
   * and to let a report say which kind of machine produced a number.
   */
  static int performanceCores() {
    return PERFORMANCE_CORES;
  }

  /** Whether this machine mixes core classes of different speed. */
  static boolean heterogeneous() {
    return PERFORMANCE_CORES > 0
        && PERFORMANCE_CORES < Math.max(1, Runtime.getRuntime().availableProcessors());
  }

  /**
   * Counts logical processors on the fastest core class.
   *
   * <p>Three sources, because three vendors publish it three ways: Intel hybrid parts list one
   * {@code cpulist} per core type under {@code /sys/devices/system/cpu/types}, ARM publishes a
   * per-cpu {@code cpu_capacity} whose largest value marks the big cores, and macOS numbers its
   * classes {@code hw.perflevelN} with 0 fastest. A machine with one class answers -1 rather than
   * its whole processor count, so "heterogeneous" and "unknown" stay distinguishable.
   */
  private static int detectPerformanceCores() {
    try {
      String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
      if (os.contains("linux")) {
        int byType = linuxIntelPerformanceCpus();
        if (byType > 0) {
          return byType;
        }
        // Checked against a real i9-14900HX: that kernel exposes no
        // /sys/devices/system/cpu/types at all, so the typed read above answers -1 on exactly the
        // hardware it was written for. Maximum frequency is the reading that works there.
        int byFrequency = linuxPerformanceCoresByMaxFrequency();
        return byFrequency > 0 ? byFrequency : linuxBigCpusByCapacity();
      }
      if (os.contains("mac") || os.contains("darwin")) {
        return macPerformanceCpus();
      }
      return -1;
    } catch (RuntimeException | IOException | InterruptedException failure) {
      if (failure instanceof InterruptedException) {
        Thread.currentThread().interrupt();
      }
      return -1;
    }
  }

  /** The {@code intel_core} type's cpulist size, a directory that exists only on hybrid parts. */
  private static int linuxIntelPerformanceCpus() {
    return linuxIntelPerformanceCpus(Path.of("/sys/devices/system/cpu"));
  }

  /**
   * The same read against an explicit sysfs root.
   *
   * <p>Separate so the hybrid path can be exercised on a machine that is not the one being
   * described. It is the branch that matters most -- every Apple Silicon part and every recent
   * Intel laptop takes it -- and it would otherwise be covered only on hardware nobody runs the
   * suite on.
   */
  static int linuxIntelPerformanceCpus(Path cpuRoot) {
    Path types = cpuRoot.resolve("types");
    if (!Files.isDirectory(types)) {
      return -1;
    }
    int performance = -1;
    boolean sawSlowerClass = false;
    try (var entries = Files.list(types)) {
      for (Path type : entries.toList()) {
        Path fileName = type.getFileName();
        if (fileName == null) {
          continue;
        }
        String name = fileName.toString().toLowerCase(Locale.ROOT);
        int count = countCpuList(readFirstLine(type.resolve("cpulist")));
        if (count <= 0) {
          continue;
        }
        if (name.contains("core")) {
          // The cpulist is logical cpus; P-cores are hyperthreaded, so fold siblings away.
          int physical = physicalCoresAmong(cpuRoot, readFirstLine(type.resolve("cpulist")));
          performance = physical > 0 ? physical : count;
        } else {
          sawSlowerClass = true;
        }
      }
    } catch (IOException | RuntimeException failure) {
      return -1;
    }
    // Only "the fast ones" if there is a slower class to contrast with.
    return sawSlowerClass ? performance : -1;
  }

  /**
   * Distinct (package, core) pairs among the logical cpus of a cpulist.
   *
   * <p>A hybrid part's performance cpulist names hardware threads, two per core, and running two
   * threads of one core through a dense matmul does not do two cores of work. Folding them here is
   * what makes the performance count comparable to a physical core count.
   */
  private static int physicalCoresAmong(Path cpus, String cpuList) {
    if (cpuList == null || cpuList.isBlank()) {
      return -1;
    }
    Set<String> cores = new HashSet<>();
    for (String part : cpuList.split(",")) {
      String range = part.strip();
      if (range.isEmpty()) {
        continue;
      }
      int dash = range.indexOf('-');
      int from = parseIndex(dash < 0 ? range : range.substring(0, dash));
      int to = dash < 0 ? from : parseIndex(range.substring(dash + 1));
      if (from < 0 || to < from) {
        continue;
      }
      for (int cpu = from; cpu <= to; cpu++) {
        Path topology = cpus.resolve("cpu" + cpu).resolve("topology");
        String core = readFirstLine(topology.resolve("core_id"));
        String socket = readFirstLine(topology.resolve("physical_package_id"));
        if (core != null) {
          cores.add((socket == null ? "0" : socket) + ':' + core);
        }
      }
    }
    return cores.isEmpty() ? -1 : cores.size();
  }

  private static int linuxPerformanceCoresByMaxFrequency() {
    return linuxPerformanceCoresByMaxFrequency(Path.of("/sys/devices/system/cpu"));
  }

  /**
   * Physical cores that are not in the slowest maximum-frequency group.
   *
   * <p>Everything <i>above</i> the lowest group counts, rather than the single highest group,
   * because performance cores are not all alike: Turbo Boost Max 3.0 marks a couple of favoured
   * cores with a higher ceiling than their siblings. On the i9-14900HX this reads 16 cpus at 4.1
   * GHz, 12 at 5.6 and 4 at 5.8 -- taking the maximum alone would answer two cores, and taking
   * everything above the minimum answers the eight that are really there. Siblings are then folded,
   * because two threads of one core do not do two cores of arithmetic.
   *
   * <p>A machine with one frequency group answers -1: it is homogeneous, or its kernel does not
   * report per-cpu ceilings, and neither is a performance-core count.
   */
  static int linuxPerformanceCoresByMaxFrequency(Path cpus) {
    if (!Files.isDirectory(cpus)) {
      return -1;
    }
    Map<Integer, Long> ceilings = new LinkedHashMap<>();
    for (int cpu = 0; cpu < 4096; cpu++) {
      Path max = cpus.resolve("cpu" + cpu).resolve("cpufreq").resolve("cpuinfo_max_freq");
      if (!Files.isReadable(max)) {
        if (cpu > Runtime.getRuntime().availableProcessors() + 8) {
          break;
        }
        continue;
      }
      String value = readFirstLine(max);
      try {
        long ceiling = Long.parseLong(value == null ? "" : value.strip());
        if (ceiling > 0) {
          ceilings.put(cpu, ceiling);
        }
      } catch (NumberFormatException ignored) {
        // A cpu that does not report a ceiling simply does not vote.
      }
    }
    if (ceilings.isEmpty()) {
      return -1;
    }
    long slowest = ceilings.values().stream().mapToLong(Long::longValue).min().orElse(0L);
    if (ceilings.values().stream().allMatch(ceiling -> ceiling == slowest)) {
      return -1;
    }
    Set<String> cores = new HashSet<>();
    for (Map.Entry<Integer, Long> entry : ceilings.entrySet()) {
      if (entry.getValue() == slowest) {
        continue;
      }
      Path topology = cpus.resolve("cpu" + entry.getKey()).resolve("topology");
      String core = readFirstLine(topology.resolve("core_id"));
      String socket = readFirstLine(topology.resolve("physical_package_id"));
      if (core != null) {
        cores.add((socket == null ? "0" : socket) + ':' + core);
      }
    }
    return cores.isEmpty() ? -1 : cores.size();
  }

  /** Logical cpus sharing the largest {@code cpu_capacity}, which is ARM's big cluster. */
  private static int linuxBigCpusByCapacity() {
    Path cpus = Path.of("/sys/devices/system/cpu");
    if (!Files.isDirectory(cpus)) {
      return -1;
    }
    Set<Integer> capacities = new HashSet<>();
    int best = -1;
    int bestCount = 0;
    for (int cpu = 0; cpu < 4096; cpu++) {
      Path capacity = cpus.resolve("cpu" + cpu).resolve("cpu_capacity");
      if (!Files.isReadable(capacity)) {
        if (cpu > Runtime.getRuntime().availableProcessors() + 8) {
          break;
        }
        continue;
      }
      int value = parsePositive(readFirstLine(capacity));
      if (value <= 0) {
        continue;
      }
      capacities.add(value);
      if (value > best) {
        best = value;
        bestCount = 1;
      } else if (value == best) {
        bestCount++;
      }
    }
    return capacities.size() > 1 ? bestCount : -1;
  }

  /** {@code hw.perflevel0.logicalcpu}, meaningful only when macOS reports several levels. */
  private static int macPerformanceCpus() throws IOException, InterruptedException {
    if (sysctlInt("hw.nperflevels") <= 1) {
      return -1;
    }
    // Physical, not logical: Apple Silicon has no SMT so they are equal today, but physical is the
    // quantity meant and stays correct if that ever changes.
    int performance = sysctlInt("hw.perflevel0.physicalcpu");
    return performance > 0 ? performance : sysctlInt("hw.perflevel0.logicalcpu");
  }

  /** Counts entries in a Linux cpulist such as {@code 0-15,24} without materialising them. */
  private static int countCpuList(String cpuList) {
    if (cpuList == null || cpuList.isBlank()) {
      return -1;
    }
    int total = 0;
    for (String part : cpuList.split(",")) {
      String range = part.strip();
      if (range.isEmpty()) {
        continue;
      }
      int dash = range.indexOf('-');
      if (dash < 0) {
        total += parseIndex(range) >= 0 ? 1 : 0;
        continue;
      }
      int from = parseIndex(range.substring(0, dash));
      int to = parseIndex(range.substring(dash + 1));
      if (from >= 0 && to >= from) {
        total += to - from + 1;
      }
    }
    return total > 0 ? total : -1;
  }

  private static int parseIndex(String value) {
    try {
      int parsed = Integer.parseInt(value.strip());
      return parsed >= 0 ? parsed : -1;
    } catch (NumberFormatException failure) {
      return -1;
    }
  }

  private static int detectPhysicalCores() {
    try {
      String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
      if (os.contains("linux")) {
        int fromSysfs = linuxCoresFromSysfs();
        return fromSysfs > 0 ? fromSysfs : linuxCoresFromProcCpuinfo();
      }
      if (os.contains("mac") || os.contains("darwin")) {
        return macCores();
      }
      return -1;
    } catch (RuntimeException | IOException | InterruptedException failure) {
      // Never let a topology probe stop a process from starting.
      if (failure instanceof InterruptedException) {
        Thread.currentThread().interrupt();
      }
      return -1;
    }
  }

  /**
   * Counts distinct (package, core) pairs under {@code /sys}.
   *
   * <p>Two hardware threads of one core report the same {@code core_id} <i>within</i> a package, so
   * the pair and not the {@code core_id} alone is what identifies a core -- on a two-socket machine
   * both sockets have a core 0, and counting ids alone would halve the answer.
   */
  private static int linuxCoresFromSysfs() {
    Path cpus = Path.of("/sys/devices/system/cpu");
    if (!Files.isDirectory(cpus)) {
      return -1;
    }
    Set<String> cores = new HashSet<>();
    for (int cpu = 0; cpu < 4096; cpu++) {
      Path topology = cpus.resolve("cpu" + cpu).resolve("topology");
      if (!Files.isDirectory(topology)) {
        if (cpu > Runtime.getRuntime().availableProcessors() + 8) {
          break;
        }
        continue;
      }
      String core = readFirstLine(topology.resolve("core_id"));
      String socket = readFirstLine(topology.resolve("physical_package_id"));
      if (core != null) {
        cores.add((socket == null ? "0" : socket) + ':' + core);
      }
    }
    return cores.isEmpty() ? -1 : cores.size();
  }

  /**
   * The same count from {@code /proc/cpuinfo}, for kernels that do not expose the sysfs topology.
   */
  private static int linuxCoresFromProcCpuinfo() throws IOException {
    Path cpuinfo = Path.of("/proc/cpuinfo");
    if (!Files.isReadable(cpuinfo)) {
      return -1;
    }
    Set<String> cores = new HashSet<>();
    String socket = "0";
    String core = null;
    for (String line : Files.readAllLines(cpuinfo, StandardCharsets.UTF_8)) {
      if (line.startsWith("physical id")) {
        socket = valueAfterColon(line);
      } else if (line.startsWith("core id")) {
        core = valueAfterColon(line);
      } else if (line.isBlank() && core != null) {
        cores.add(socket + ':' + core);
        socket = "0";
        core = null;
      }
    }
    if (core != null) {
      cores.add(socket + ':' + core);
    }
    return cores.isEmpty() ? -1 : cores.size();
  }

  /** {@code sysctl hw.physicalcpu}, the only place macOS publishes this. */
  private static int macCores() throws IOException, InterruptedException {
    return sysctlInt("hw.physicalcpu");
  }

  /** One bounded {@code sysctl -n} read, so a wedged fork cannot hold up class initialisation. */
  private static int sysctlInt(String name) throws IOException, InterruptedException {
    ProcessBuilder builder = new ProcessBuilder("/usr/sbin/sysctl", "-n", name);
    builder.redirectErrorStream(true);
    Process process = builder.start();
    String output;
    try (var stream = process.getInputStream()) {
      output = new String(stream.readAllBytes(), StandardCharsets.UTF_8).strip();
    }
    if (!process.waitFor(2, TimeUnit.SECONDS)) {
      process.destroyForcibly();
      return -1;
    }
    return process.exitValue() == 0 ? parsePositive(output) : -1;
  }

  private static String readFirstLine(Path file) {
    try {
      if (!Files.isReadable(file)) {
        return null;
      }
      List<String> lines = Files.readAllLines(file, StandardCharsets.UTF_8);
      return lines.isEmpty() ? null : lines.getFirst().strip();
    } catch (IOException | RuntimeException failure) {
      return null;
    }
  }

  private static String valueAfterColon(String line) {
    int colon = line.indexOf(':');
    return colon < 0 ? "0" : line.substring(colon + 1).strip();
  }

  private static int parsePositive(String value) {
    try {
      int parsed = Integer.parseInt(value.strip());
      return parsed > 0 ? parsed : -1;
    } catch (NumberFormatException failure) {
      return -1;
    }
  }
}
