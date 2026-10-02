```
label=genoa-256-on-avx512 date=2026-09-17T04:17:47Z commit=db41481d dirty=0
phases=dequant ab arms=scalar simd-byte simd-int simd-split rounds=1 batches=1,2,4,8,512 jmh=-f 2 -wi 3 -i 5 -w 2 -r 2
Linux modeljars-granite41-hybrid-20260916 6.8.0-138-generic #138-Ubuntu SMP PREEMPT_DYNAMIC Fri Jul 31 22:41:49 UTC 2026 x86_64 x86_64 x86_64 GNU/Linux
Architecture:                            x86_64 CPU op-mode(s):                          32-bit, 64-bit Address sizes:                           40 bits physical, 57 bits virtual Byte Order:                              Little Endian CPU(s):                                  16 On-line CPU(s) list:                     0-15 Vendor ID:                               AuthenticAMD BIOS Vendor ID:                          QEMU Model name:                              AMD EPYC-Genoa Processor BIOS Model name:                         NotSpecified  CPU @ 2.0GHz BIOS CPU family:                         1 CPU family:                              25 Model:                                   17 Thread(s) per core:                      1 Core(s) per socket:                      16 Socket(s):                               1 Stepping:                                0 BogoMIPS:                                4799.99 Flags:                                   fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm rep_good nopl cpuid extd_apicid tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm cmp_legacy cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw topoext perfctr_core ssbd ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves avx512_bf16 clzero xsaveerptr wbnoinvd arat avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid fsrm Hypervisor vendor:                       KVM Virtualization type:                     full L1d cache:                               512 KiB (16 instances) L1i cache:                               512 KiB (16 instances) L2 cache:                                16 MiB (16 instances) L3 cache:                                32 MiB (1 instance) NUMA node(s):                            1 NUMA node0 CPU(s):                       0-15 Vulnerability Gather data sampling:      Not affected Vulnerability Indirect target selection: Not affected Vulnerability Itlb multihit:             Not affected Vulnerability L1tf:                      Not affected Vulnerability Mds:                       Not affected Vulnerability Meltdown:                  Not affected Vulnerability Mmio stale data:           Not affected Vulnerability Reg file data sampling:    Not affected Vulnerability Retbleed:                  Not affected Vulnerability Spec rstack overflow:      Mitigation; Safe RET Vulnerability Spec store bypass:         Mitigation; Speculative Store Bypass disabled via prctl Vulnerability Spectre v1:                Mitigation; usercopy/swapgs barriers and __user pointer sanitization Vulnerability Spectre v2:                Mitigation; Enhanced / Automatic IBRS; IBPB conditional; STIBP disabled; PBRSB-eIBRS Not affected; BHI Not affected Vulnerability Srbds:                     Not affected Vulnerability Tsa:                       Vulnerable: Clear CPU buffers attempted, no microcode Vulnerability Tsx async abort:           Not affected Vulnerability Vmscape:                   Not affected 
fma avx avx2 avx512f avx512dq avx512ifma avx512cd avx512bw avx512vl avx512_bf16 avx512vbmi avx512_vbmi2 avx512_vnni avx512_bitalg avx512_vpopcntdq 
openjdk version "25.0.4.1" 2026-08-18 LTS
OpenJDK Runtime Environment Temurin-25.0.4.1+1 (build 25.0.4.1+1-LTS)
OpenJDK 64-Bit Server VM Temurin-25.0.4.1+1 (build 25.0.4.1+1-LTS, mixed mode, sharing)
jvm=--add-modules jdk.incubator.vector -Xmx8g -Xms8g -XX:+AlwaysPreTouch -XX:+UseG1GC
```

### Dequantisation alone (single thread; ms/op, M F32 elements/s, speedup vs scalar)

| format | shape | scalar | simd-byte | simd-int | simd-split |
|---|---|---:|---:|---:|---:|
| Q4_K | 8192x2560 | 15.64 ± 0.55 / 1340 M / **1.00x** | 3.86 ± 0.13 / 5434 M / **4.05x** | 3.39 ± 0.13 / 6184 M / **4.61x** | 3.43 ± 0.18 / 6116 M / **4.56x** |
| Q4_K | 2560x8192 | 15.61 ± 0.37 / 1343 M / **1.00x** | 3.89 ± 0.12 / 5395 M / **4.02x** | 3.28 ± 0.04 / 6393 M / **4.76x** | 3.47 ± 0.11 / 6038 M / **4.50x** |
| Q6_K | 8192x2560 | 35.04 ± 0.74 / 599 M / **1.00x** | 8.56 ± 0.12 / 2451 M / **4.09x** | 4.93 ± 0.10 / 4256 M / **7.11x** | 5.70 ± 0.12 / 3677 M / **6.14x** |
| Q6_K | 2560x8192 | 34.82 ± 0.32 / 602 M / **1.00x** | 8.64 ± 0.22 / 2428 M / **4.03x** | 5.00 ± 0.12 / 4192 M / **6.96x** | 5.78 ± 0.19 / 3626 M / **6.02x** |

Geometric-mean dequant speedup vs scalar: scalar 1.00x, simd-byte 4.05x, simd-int 5.74x, simd-split 5.25x

Candidate (best geomean in THIS run; the rule fixes it from Genoa 256-bit): simd-int

**(b) dequant >= 2.0x scalar, all 4 cells: PASS** — Q4_K 8192x2560 4.61x, Q4_K 2560x8192 4.76x, Q6_K 8192x2560 7.11x, Q6_K 2560x8192 6.96x

### Band A/B, multi-threaded (executor default)

ms/op ± 99.9% half-width. `new/old` = scalar band ms / arm band ms (>1 = faster than scalar dequant). `vs int` = integer ms / band ms.

| format | shape | batch | integer | band scalar | band simd-byte | band simd-int | band simd-split | simd-byte new/old | simd-int new/old | simd-split new/old | scalar vs int | simd-byte vs int | simd-int vs int | simd-split vs int |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q4_K | 2560x8192 | 1 | 0.56 ± 0.02 | 1.30 ± 0.02 | 0.39 ± 0.01 | 0.39 ± 0.01 | 0.41 ± 0.02 | **3.31x** | **3.36x** | **3.20x** | 0.43x | 1.44x | 1.46x | 1.39x |
| Q4_K | 2560x8192 | 2 | 1.25 ± 0.13 | 1.37 ± 0.03 | 0.50 ± 0.03 | 0.48 ± 0.01 | 0.49 ± 0.02 | **2.76x** | **2.85x** | **2.80x** | 0.91x | 2.51x | 2.59x | 2.55x |
| Q4_K | 2560x8192 | 4 | 1.67 ± 0.08 | 1.45 ± 0.02 | 0.59 ± 0.02 | 0.57 ± 0.06 | 0.54 ± 0.04 | **2.44x** | **2.52x** | **2.69x** | 1.15x | 2.82x | 2.91x | 3.10x |
| Q4_K | 2560x8192 | 8 | 3.26 ± 0.32 | 2.87 ± 0.05 | 1.11 ± 0.03 | 1.13 ± 0.03 | 1.03 ± 0.04 | **2.59x** | **2.54x** | **2.80x** | 1.14x | 2.93x | 2.89x | 3.18x |
| Q4_K | 2560x8192 | 512 | 220.10 ± 11.14 | 28.86 ± 0.44 | 28.17 ± 0.73 | 28.96 ± 1.20 | 28.32 ± 1.22 | **1.02x** | **1.00x** | **1.02x** | 7.63x | 7.81x | 7.60x | 7.77x |
| Q4_K | 8192x2560 | 1 | 0.55 ± 0.03 | 1.27 ± 0.06 | 0.40 ± 0.01 | 0.42 ± 0.02 | 0.43 ± 0.03 | **3.19x** | **3.04x** | **2.94x** | 0.43x | 1.38x | 1.32x | 1.27x |
| Q4_K | 8192x2560 | 2 | 1.11 ± 0.04 | 1.36 ± 0.03 | 0.49 ± 0.01 | 0.48 ± 0.03 | 0.48 ± 0.04 | **2.77x** | **2.80x** | **2.81x** | 0.82x | 2.26x | 2.29x | 2.30x |
| Q4_K | 8192x2560 | 4 | 1.68 ± 0.10 | 1.39 ± 0.03 | 0.60 ± 0.03 | 0.56 ± 0.01 | 0.56 ± 0.01 | **2.32x** | **2.48x** | **2.48x** | 1.21x | 2.80x | 3.00x | 3.00x |
| Q4_K | 8192x2560 | 8 | 3.14 ± 0.11 | 1.63 ± 0.03 | 0.86 ± 0.08 | 0.86 ± 0.07 | 0.77 ± 0.03 | **1.90x** | **1.90x** | **2.11x** | 1.93x | 3.65x | 3.66x | 4.06x |
| Q4_K | 8192x2560 | 512 | 202.46 ± 14.67 | 28.25 ± 0.89 | 28.98 ± 0.98 | 30.22 ± 0.98 | 28.09 ± 0.51 | **0.97x** | **0.93x** | **1.01x** | 7.17x | 6.99x | 6.70x | 7.21x |
| Q6_K | 2560x8192 | 1 | 0.82 ± 0.09 | 2.71 ± 0.31 | 0.82 ± 0.02 | 0.58 ± 0.05 | 0.64 ± 0.02 | **3.30x** | **4.70x** | **4.26x** | 0.30x | 1.00x | 1.42x | 1.29x |
| Q6_K | 2560x8192 | 2 | 1.63 ± 0.08 | 2.95 ± 0.04 | 0.94 ± 0.04 | 0.66 ± 0.03 | 0.69 ± 0.02 | **3.16x** | **4.47x** | **4.27x** | 0.55x | 1.74x | 2.46x | 2.35x |
| Q6_K | 2560x8192 | 4 | 1.76 ± 0.10 | 2.82 ± 0.32 | 1.02 ± 0.04 | 0.76 ± 0.02 | 0.77 ± 0.03 | **2.75x** | **3.70x** | **3.66x** | 0.62x | 1.71x | 2.31x | 2.29x |
| Q6_K | 2560x8192 | 8 | 3.53 ± 0.23 | 5.56 ± 0.65 | 1.95 ± 0.05 | 1.42 ± 0.04 | 1.49 ± 0.02 | **2.85x** | **3.92x** | **3.73x** | 0.63x | 1.81x | 2.49x | 2.36x |
| Q6_K | 2560x8192 | 512 | 237.53 ± 6.58 | 31.64 ± 1.48 | 28.67 ± 1.79 | 28.65 ± 1.04 | 28.27 ± 0.40 | **1.10x** | **1.10x** | **1.12x** | 7.51x | 8.29x | 8.29x | 8.40x |
| Q6_K | 8192x2560 | 1 | 0.83 ± 0.07 | 2.39 ± 0.03 | 0.81 ± 0.02 | 0.58 ± 0.03 | 0.66 ± 0.04 | **2.94x** | **4.14x** | **3.62x** | 0.35x | 1.02x | 1.44x | 1.26x |
| Q6_K | 8192x2560 | 2 | 1.56 ± 0.11 | 2.43 ± 0.04 | 0.95 ± 0.05 | 0.67 ± 0.03 | 0.71 ± 0.01 | **2.57x** | **3.61x** | **3.44x** | 0.64x | 1.65x | 2.32x | 2.21x |
| Q6_K | 8192x2560 | 4 | 1.76 ± 0.11 | 2.52 ± 0.04 | 1.08 ± 0.05 | 0.78 ± 0.08 | 0.77 ± 0.02 | **2.34x** | **3.23x** | **3.28x** | 0.70x | 1.63x | 2.25x | 2.29x |
| Q6_K | 8192x2560 | 8 | 3.35 ± 0.19 | 2.75 ± 0.04 | 1.31 ± 0.04 | 1.11 ± 0.03 | 1.03 ± 0.05 | **2.11x** | **2.48x** | **2.67x** | 1.22x | 2.57x | 3.03x | 3.26x |
| Q6_K | 8192x2560 | 512 | 219.33 ± 10.48 | 29.50 ± 0.47 | 29.82 ± 1.42 | 30.13 ± 1.82 | 28.15 ± 1.27 | **0.99x** | **0.98x** | **1.05x** | 7.44x | 7.36x | 7.28x | 7.79x |

**(c) simd-int: batch 1/4 faster than scalar in every cell: PASS (8/8); batch 512 <= 1.03x scalar in every cell: FAIL (worst 1.070x)**
