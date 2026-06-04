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

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class PanamaConstantsTest {

  @Test
  void x86ArchDetectionAcceptsCommonAliases() {
    assertThat(PanamaConstants.isX86Arch("amd64")).isTrue();
    assertThat(PanamaConstants.isX86Arch("x86_64")).isTrue();
    assertThat(PanamaConstants.isX86Arch("i686")).isTrue();
    assertThat(PanamaConstants.isX86Arch("aarch64")).isFalse();
  }

  @Test
  void cpuInfoParserDetectsAuthenticAmdWithFmaFlag() {
    PanamaConstants.CpuInfo info =
        PanamaConstants.parseCpuInfo(
            """
            processor   : 0
            vendor_id   : AuthenticAMD
            model name  : AMD Ryzen 9 5950X 16-Core Processor
            flags       : fpu vme de pse tsc msr pae mce cx8 apic sep mtrr fma sse4_2 avx
            """);

    assertThat(PanamaConstants.isAmdWithFma(info)).isTrue();
  }

  @Test
  void cpuInfoParserDoesNotMatchPartialFmaFlag() {
    PanamaConstants.CpuInfo info =
        PanamaConstants.parseCpuInfo(
            """
            vendor_id   : AuthenticAMD
            flags       : fpu fma4 avx
            """);

    assertThat(PanamaConstants.isAmdWithFma(info)).isFalse();
  }

  @Test
  void cpuInfoParserRejectsIntelEvenWhenFmaIsPresent() {
    PanamaConstants.CpuInfo info =
        PanamaConstants.parseCpuInfo(
            """
            vendor_id   : GenuineIntel
            model name  : Intel(R) Core(TM) i9
            flags       : fpu sse4_2 avx fma
            """);

    assertThat(PanamaConstants.isAmdWithFma(info)).isFalse();
  }
}
