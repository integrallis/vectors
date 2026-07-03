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

  // ─── P3.5 SVE detection ────────────────────────────────────────────────────

  @Test
  void armArchDetectionAcceptsAarch64Aliases() {
    assertThat(PanamaConstants.isArmArch("aarch64")).isTrue();
    assertThat(PanamaConstants.isArmArch("arm64")).isTrue();
    assertThat(PanamaConstants.isArmArch("AARCH64")).isTrue();
    assertThat(PanamaConstants.isArmArch("amd64")).isFalse();
    assertThat(PanamaConstants.isArmArch("x86_64")).isFalse();
  }

  @Test
  void cpuInfoParserDetectsSveOnArmFeaturesLine() {
    // Graviton 3 cpuinfo shape: Features line lists scalable-vector flags including 'sve'.
    PanamaConstants.CpuInfo info =
        PanamaConstants.parseCpuInfo(
            """
            processor   : 0
            BogoMIPS    : 2100.00
            Features    : fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp \
                          cpuid asimdrdm jscvt fcma lrcpc dcpop sha3 sm3 sm4 asimddp sha512 \
                          sve asimdfhm
            CPU implementer : 0x41
            """);

    assertThat(PanamaConstants.hasSveCpuFlag(info)).isTrue();
  }

  @Test
  void cpuInfoParserDoesNotMatchPartialSveToken() {
    // 'sve2-only' / 'sve3' substrings should not satisfy the bare 'sve' token check (we use
    // exact-token matching, so a CPU that genuinely supports sve will list it as a separate
    // token alongside sve2).
    PanamaConstants.CpuInfo info =
        PanamaConstants.parseCpuInfo(
            """
            Features    : fp asimd sve2 asimddp
            """);

    assertThat(PanamaConstants.hasSveCpuFlag(info))
        .as("sve2 without bare 'sve' must not register as SVE support")
        .isFalse();
  }

  @Test
  void cpuInfoParserRejectsSveOnX86CpuInfo() {
    // A regular x86 cpuinfo has no Features: line and no sve token — must return false.
    PanamaConstants.CpuInfo info =
        PanamaConstants.parseCpuInfo(
            """
            vendor_id   : GenuineIntel
            flags       : fpu sse4_2 avx fma avx2 avx512f
            """);

    assertThat(PanamaConstants.hasSveCpuFlag(info)).isFalse();
  }

  @Test
  void hasSveIsFalseOnNonArmHosts() {
    // This test must run on every CI box including x86_64 — the JVM-resolved HAS_SVE on those
    // platforms must be false. The complement (HAS_SVE true on an ARM box with SVE) is genuine
    // hardware coverage that only ARM-with-SVE CI can validate.
    if (PanamaConstants.isArmArch(System.getProperty("os.arch", ""))) {
      // Skip: ARM platform may legitimately have SVE; can't assert false.
      return;
    }
    assertThat(PanamaConstants.HAS_SVE).as("non-ARM hosts must report HAS_SVE=false").isFalse();
  }
}
