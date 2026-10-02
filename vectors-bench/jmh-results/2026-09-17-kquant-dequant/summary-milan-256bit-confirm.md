```
label=milan-256bit-confirm date=2026-10-02T18:19:24Z commit=ac14df2 dirty=0
phases=dequant ab arms=scalar simd-split rounds=3 batches=1,2,4,8,512 jmh=-f 2 -wi 3 -i 5 -w 2 -r 2
Linux modeljars-bench-cpu-20261002-kquant 6.8.0-138-generic #138-Ubuntu SMP PREEMPT_DYNAMIC Fri Jul 31 22:41:49 UTC 2026 x86_64 x86_64 x86_64 GNU/Linux
Architecture:                            x86_64 CPU op-mode(s):                          32-bit, 64-bit Address sizes:                           40 bits physical, 48 bits virtual Byte Order:                              Little Endian CPU(s):                                  8 On-line CPU(s) list:                     0-7 Vendor ID:                               AuthenticAMD BIOS Vendor ID:                          QEMU Model name:                              AMD EPYC-Milan Processor BIOS Model name:                         NotSpecified  CPU @ 2.0GHz BIOS CPU family:                         1 CPU family:                              25 Model:                                   1 Thread(s) per core:                      2 Core(s) per socket:                      4 Socket(s):                               1 Stepping:                                1 BogoMIPS:                                4800.00 Flags:                                   fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm rep_good nopl cpuid extd_apicid tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm cmp_legacy cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw topoext perfctr_core ssbd ibrs ibpb stibp vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid rdseed adx smap clflushopt clwb sha_ni xsaveopt xsavec xgetbv1 xsaves clzero xsaveerptr wbnoinvd arat umip pku ospke rdpid fsrm Hypervisor vendor:                       KVM Virtualization type:                     full L1d cache:                               128 KiB (4 instances) L1i cache:                               128 KiB (4 instances) L2 cache:                                2 MiB (4 instances) L3 cache:                                32 MiB (1 instance) NUMA node(s):                            1 NUMA node0 CPU(s):                       0-7 Vulnerability Gather data sampling:      Not affected Vulnerability Indirect target selection: Not affected Vulnerability Itlb multihit:             Not affected Vulnerability L1tf:                      Not affected Vulnerability Mds:                       Not affected Vulnerability Meltdown:                  Not affected Vulnerability Mmio stale data:           Not affected Vulnerability Reg file data sampling:    Not affected Vulnerability Retbleed:                  Not affected Vulnerability Spec rstack overflow:      Vulnerable: Safe RET, no microcode Vulnerability Spec store bypass:         Mitigation; Speculative Store Bypass disabled via prctl Vulnerability Spectre v1:                Mitigation; usercopy/swapgs barriers and __user pointer sanitization Vulnerability Spectre v2:                Mitigation; Retpolines; IBPB conditional; IBRS_FW; STIBP conditional; RSB filling; PBRSB-eIBRS Not affected; BHI Not affected Vulnerability Srbds:                     Not affected Vulnerability Tsa:                       Vulnerable: Clear CPU buffers attempted, no microcode Vulnerability Tsx async abort:           Not affected Vulnerability Vmscape:                   Not affected 
fma avx avx2 
openjdk version "25.0.4.1" 2026-08-18 LTS
OpenJDK Runtime Environment Temurin-25.0.4.1+1 (build 25.0.4.1+1-LTS)
OpenJDK 64-Bit Server VM Temurin-25.0.4.1+1 (build 25.0.4.1+1-LTS, mixed mode, sharing)
jvm=--add-modules jdk.incubator.vector -Xmx8g -Xms8g -XX:+AlwaysPreTouch -XX:+UseG1GC
```

### Dequantisation alone (single thread; ms/op, M F32 elements/s, speedup vs scalar)

| format | shape | scalar | simd-split |
|---|---|---:|---:|
| Q4_K | 8192x2560 | 15.48 ± 0.51 / 1354 M / **1.00x** | 3.18 ± 0.03 / 6595 M / **4.87x** |
| Q4_K | 2560x8192 | 15.25 ± 0.26 / 1376 M / **1.00x** | 3.26 ± 0.13 / 6442 M / **4.68x** |
| Q6_K | 8192x2560 | 33.44 ± 0.74 / 627 M / **1.00x** | 5.81 ± 0.31 / 3609 M / **5.75x** |
| Q6_K | 2560x8192 | 33.41 ± 0.62 / 628 M / **1.00x** | 5.62 ± 0.05 / 3733 M / **5.95x** |

Geometric-mean dequant speedup vs scalar: scalar 1.00x, simd-split 5.29x

Candidate (given): simd-split

**(b) dequant >= 2.0x scalar, all 4 cells: PASS** — Q4_K 8192x2560 4.87x, Q4_K 2560x8192 4.68x, Q6_K 8192x2560 5.75x, Q6_K 2560x8192 5.95x

### Band A/B, multi-threaded (executor default) round 1

ms/op ± 99.9% half-width. `new/old` = scalar band ms / arm band ms (>1 = faster than scalar dequant). `vs int` = integer ms / band ms.

| format | shape | batch | integer | band scalar | band simd-split | simd-split new/old | scalar vs int | simd-split vs int |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Q4_K | 2560x8192 | 1 | 1.61 ± 0.02 | 4.53 ± 0.02 | 1.20 ± 0.02 | **3.78x** | 0.35x | 1.34x |
| Q4_K | 2560x8192 | 2 | 3.61 ± 0.05 | 4.75 ± 0.04 | 1.45 ± 0.03 | **3.27x** | 0.76x | 2.48x |
| Q4_K | 2560x8192 | 4 | 5.00 ± 0.02 | 5.14 ± 0.04 | 1.92 ± 0.02 | **2.68x** | 0.97x | 2.61x |
| Q4_K | 2560x8192 | 8 | 10.08 ± 0.15 | 6.06 ± 0.11 | 2.94 ± 0.04 | **2.06x** | 1.66x | 3.43x |
| Q4_K | 2560x8192 | 512 | 643.13 ± 3.50 | 133.25 ± 1.21 | 124.68 ± 0.58 | **1.07x** | 4.83x | 5.16x |
| Q4_K | 8192x2560 | 1 | 1.61 ± 0.03 | 4.33 ± 0.06 | 1.20 ± 0.02 | **3.62x** | 0.37x | 1.34x |
| Q4_K | 8192x2560 | 2 | 3.58 ± 0.13 | 4.54 ± 0.02 | 1.43 ± 0.04 | **3.18x** | 0.79x | 2.51x |
| Q4_K | 8192x2560 | 4 | 4.96 ± 0.00 | 4.92 ± 0.04 | 1.93 ± 0.11 | **2.55x** | 1.01x | 2.56x |
| Q4_K | 8192x2560 | 8 | 9.96 ± 0.32 | 5.85 ± 0.06 | 2.87 ± 0.05 | **2.04x** | 1.70x | 3.47x |
| Q4_K | 8192x2560 | 512 | 637.16 ± 2.75 | 125.28 ± 0.95 | 118.56 ± 1.42 | **1.06x** | 5.09x | 5.37x |
| Q6_K | 2560x8192 | 1 | 2.44 ± 0.04 | 9.73 ± 0.03 | 1.86 ± 0.03 | **5.22x** | 0.25x | 1.31x |
| Q6_K | 2560x8192 | 2 | 4.90 ± 0.01 | 9.92 ± 0.05 | 2.12 ± 0.03 | **4.68x** | 0.49x | 2.31x |
| Q6_K | 2560x8192 | 4 | 6.04 ± 0.12 | 10.33 ± 0.23 | 2.58 ± 0.05 | **3.99x** | 0.59x | 2.34x |
| Q6_K | 2560x8192 | 8 | 11.95 ± 0.13 | 11.03 ± 0.04 | 3.62 ± 0.05 | **3.05x** | 1.08x | 3.30x |
| Q6_K | 2560x8192 | 512 | 787.87 ± 9.11 | 137.15 ± 2.30 | 129.72 ± 0.97 | **1.06x** | 5.74x | 6.07x |
| Q6_K | 8192x2560 | 1 | 2.42 ± 0.03 | 9.19 ± 0.04 | 1.85 ± 0.07 | **4.96x** | 0.26x | 1.30x |
| Q6_K | 8192x2560 | 2 | 4.88 ± 0.17 | 9.41 ± 0.08 | 2.08 ± 0.05 | **4.52x** | 0.52x | 2.34x |
| Q6_K | 8192x2560 | 4 | 6.01 ± 0.23 | 9.79 ± 0.05 | 2.51 ± 0.04 | **3.90x** | 0.61x | 2.40x |
| Q6_K | 8192x2560 | 8 | 12.00 ± 0.30 | 10.55 ± 0.05 | 3.52 ± 0.04 | **2.99x** | 1.14x | 3.41x |
| Q6_K | 8192x2560 | 512 | 768.67 ± 13.88 | 128.59 ± 0.83 | 122.57 ± 0.85 | **1.05x** | 5.98x | 6.27x |

**(c) simd-split round 1: batch 1/4 faster than scalar in every cell: PASS (8/8); batch 512 <= 1.03x scalar in every cell: PASS (worst 0.953x)**

### Band A/B, multi-threaded (executor default) round 2

ms/op ± 99.9% half-width. `new/old` = scalar band ms / arm band ms (>1 = faster than scalar dequant). `vs int` = integer ms / band ms.

| format | shape | batch | integer | band scalar | band simd-split | simd-split new/old | scalar vs int | simd-split vs int |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Q4_K | 2560x8192 | 1 | 1.61 ± 0.02 | 4.53 ± 0.02 | 1.21 ± 0.02 | **3.76x** | 0.35x | 1.33x |
| Q4_K | 2560x8192 | 2 | 3.63 ± 0.04 | 4.74 ± 0.03 | 1.44 ± 0.02 | **3.28x** | 0.77x | 2.51x |
| Q4_K | 2560x8192 | 4 | 5.01 ± 0.02 | 5.10 ± 0.01 | 1.91 ± 0.03 | **2.67x** | 0.98x | 2.62x |
| Q4_K | 2560x8192 | 8 | 10.01 ± 0.01 | 6.01 ± 0.03 | 2.93 ± 0.02 | **2.06x** | 1.66x | 3.42x |
| Q4_K | 2560x8192 | 512 | 644.28 ± 14.46 | 133.89 ± 3.02 | 127.01 ± 2.22 | **1.05x** | 4.81x | 5.07x |
| Q4_K | 8192x2560 | 1 | 1.61 ± 0.03 | 4.34 ± 0.05 | 1.18 ± 0.01 | **3.69x** | 0.37x | 1.37x |
| Q4_K | 8192x2560 | 2 | 3.58 ± 0.09 | 4.56 ± 0.05 | 1.41 ± 0.04 | **3.22x** | 0.79x | 2.53x |
| Q4_K | 8192x2560 | 4 | 4.96 ± 0.01 | 4.89 ± 0.03 | 1.90 ± 0.04 | **2.57x** | 1.01x | 2.61x |
| Q4_K | 8192x2560 | 8 | 9.90 ± 0.00 | 5.80 ± 0.11 | 2.84 ± 0.05 | **2.04x** | 1.71x | 3.48x |
| Q4_K | 8192x2560 | 512 | 637.33 ± 0.70 | 122.60 ± 1.02 | 121.05 ± 1.28 | **1.01x** | 5.20x | 5.26x |
| Q6_K | 2560x8192 | 1 | 2.44 ± 0.04 | 9.73 ± 0.02 | 1.88 ± 0.03 | **5.17x** | 0.25x | 1.29x |
| Q6_K | 2560x8192 | 2 | 4.96 ± 0.17 | 9.94 ± 0.07 | 2.12 ± 0.03 | **4.68x** | 0.50x | 2.34x |
| Q6_K | 2560x8192 | 4 | 6.16 ± 0.32 | 10.28 ± 0.04 | 2.60 ± 0.06 | **3.95x** | 0.60x | 2.36x |
| Q6_K | 2560x8192 | 8 | 11.92 ± 0.03 | 11.14 ± 0.08 | 3.62 ± 0.02 | **3.08x** | 1.07x | 3.30x |
| Q6_K | 2560x8192 | 512 | 789.56 ± 11.85 | 138.97 ± 2.32 | 127.06 ± 1.06 | **1.09x** | 5.68x | 6.21x |
| Q6_K | 8192x2560 | 1 | 2.44 ± 0.11 | 9.19 ± 0.01 | 1.84 ± 0.05 | **4.98x** | 0.27x | 1.32x |
| Q6_K | 8192x2560 | 2 | 4.85 ± 0.01 | 9.39 ± 0.02 | 2.08 ± 0.05 | **4.51x** | 0.52x | 2.33x |
| Q6_K | 8192x2560 | 4 | 5.91 ± 0.02 | 9.75 ± 0.05 | 2.51 ± 0.05 | **3.88x** | 0.61x | 2.35x |
| Q6_K | 8192x2560 | 8 | 11.97 ± 0.50 | 10.59 ± 0.10 | 3.51 ± 0.06 | **3.02x** | 1.13x | 3.41x |
| Q6_K | 8192x2560 | 512 | 773.78 ± 13.83 | 127.10 ± 0.76 | 120.40 ± 2.95 | **1.06x** | 6.09x | 6.43x |

**(c) simd-split round 2: batch 1/4 faster than scalar in every cell: PASS (8/8); batch 512 <= 1.03x scalar in every cell: PASS (worst 0.987x)**

### Band A/B, multi-threaded (executor default) round 3

ms/op ± 99.9% half-width. `new/old` = scalar band ms / arm band ms (>1 = faster than scalar dequant). `vs int` = integer ms / band ms.

| format | shape | batch | integer | band scalar | band simd-split | simd-split new/old | scalar vs int | simd-split vs int |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Q4_K | 2560x8192 | 1 | 1.64 ± 0.06 | 4.52 ± 0.03 | 1.19 ± 0.03 | **3.80x** | 0.36x | 1.38x |
| Q4_K | 2560x8192 | 2 | 3.60 ± 0.04 | 4.73 ± 0.02 | 1.45 ± 0.01 | **3.26x** | 0.76x | 2.48x |
| Q4_K | 2560x8192 | 4 | 5.00 ± 0.02 | 5.12 ± 0.01 | 1.90 ± 0.03 | **2.69x** | 0.98x | 2.63x |
| Q4_K | 2560x8192 | 8 | 9.97 ± 0.02 | 6.03 ± 0.03 | 2.98 ± 0.06 | **2.03x** | 1.65x | 3.35x |
| Q4_K | 2560x8192 | 512 | 641.33 ± 2.85 | 132.39 ± 1.47 | 126.22 ± 1.27 | **1.05x** | 4.84x | 5.08x |
| Q4_K | 8192x2560 | 1 | 1.60 ± 0.02 | 4.32 ± 0.05 | 1.17 ± 0.02 | **3.70x** | 0.37x | 1.37x |
| Q4_K | 8192x2560 | 2 | 3.55 ± 0.00 | 4.52 ± 0.06 | 1.39 ± 0.02 | **3.25x** | 0.78x | 2.55x |
| Q4_K | 8192x2560 | 4 | 4.95 ± 0.01 | 4.89 ± 0.02 | 1.87 ± 0.03 | **2.62x** | 1.01x | 2.65x |
| Q4_K | 8192x2560 | 8 | 9.88 ± 0.00 | 5.76 ± 0.08 | 2.79 ± 0.05 | **2.06x** | 1.72x | 3.54x |
| Q4_K | 8192x2560 | 512 | 636.88 ± 1.08 | 122.60 ± 0.82 | 119.15 ± 1.63 | **1.03x** | 5.19x | 5.35x |
| Q6_K | 2560x8192 | 1 | 2.45 ± 0.10 | 9.71 ± 0.02 | 1.85 ± 0.06 | **5.26x** | 0.25x | 1.33x |
| Q6_K | 2560x8192 | 2 | 4.92 ± 0.19 | 9.91 ± 0.07 | 2.10 ± 0.08 | **4.72x** | 0.50x | 2.34x |
| Q6_K | 2560x8192 | 4 | 5.98 ± 0.01 | 10.26 ± 0.04 | 2.56 ± 0.05 | **4.01x** | 0.58x | 2.34x |
| Q6_K | 2560x8192 | 8 | 11.95 ± 0.02 | 11.04 ± 0.04 | 3.70 ± 0.02 | **2.98x** | 1.08x | 3.23x |
| Q6_K | 2560x8192 | 512 | 785.56 ± 8.60 | 136.72 ± 1.27 | 131.62 ± 1.92 | **1.04x** | 5.75x | 5.97x |
| Q6_K | 8192x2560 | 1 | 2.42 ± 0.03 | 9.18 ± 0.07 | 1.77 ± 0.02 | **5.18x** | 0.26x | 1.36x |
| Q6_K | 8192x2560 | 2 | 4.84 ± 0.01 | 9.35 ± 0.03 | 2.02 ± 0.04 | **4.64x** | 0.52x | 2.40x |
| Q6_K | 8192x2560 | 4 | 5.94 ± 0.19 | 9.74 ± 0.05 | 2.44 ± 0.06 | **3.99x** | 0.61x | 2.43x |
| Q6_K | 8192x2560 | 8 | 11.92 ± 0.34 | 10.50 ± 0.06 | 3.51 ± 0.09 | **2.99x** | 1.13x | 3.40x |
| Q6_K | 8192x2560 | 512 | 778.29 ± 10.51 | 127.85 ± 1.54 | 122.28 ± 2.01 | **1.05x** | 6.09x | 6.36x |

**(c) simd-split round 3: batch 1/4 faster than scalar in every cell: PASS (8/8); batch 512 <= 1.03x scalar in every cell: PASS (worst 0.972x)**
