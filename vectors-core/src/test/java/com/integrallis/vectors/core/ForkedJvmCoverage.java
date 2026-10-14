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
import java.lang.management.ManagementFactory;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/** Carries the parent's coverage agent into functional forks, without sharing execution files. */
final class ForkedJvmCoverage {
  private static final AtomicLong NEXT = new AtomicLong();

  private ForkedJvmCoverage() {}

  static ProcessBuilder process(String... command) throws IOException {
    return process(List.of(command));
  }

  static ProcessBuilder process(List<String> command) throws IOException {
    List<String> result = new ArrayList<>(command);
    String directory = System.getProperty("vectors.test.forkCoverageDir");
    if (directory == null) return new ProcessBuilder(result);
    for (String argument : ManagementFactory.getRuntimeMXBean().getInputArguments()) {
      if (!argument.startsWith("-javaagent:")) continue;
      int separator = argument.indexOf('=');
      String agent = separator < 0 ? argument : argument.substring(0, separator);
      if (!agent.toLowerCase(java.util.Locale.ROOT).contains("jacoco")) continue;
      Path destination = Path.of(directory).toAbsolutePath();
      Files.createDirectories(destination);
      destination =
          destination.resolve(
              "fork-" + ProcessHandle.current().pid() + "-" + NEXT.incrementAndGet() + ".exec");
      List<String> options = new ArrayList<>();
      if (separator >= 0)
        for (String option : argument.substring(separator + 1).split(",")) {
          String name = option.split("=", 2)[0];
          if (!List.of("destfile", "append", "output", "dumponexit").contains(name))
            options.add(option);
        }
      options.add("destfile=" + destination);
      options.add("append=false");
      options.add("output=file");
      options.add("dumponexit=true");
      result.add(1, agent + "=" + String.join(",", options));
      return new ProcessBuilder(result);
    }
    return new ProcessBuilder(result);
  }
}
