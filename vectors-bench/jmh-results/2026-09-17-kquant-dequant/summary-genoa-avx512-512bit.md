```
label=genoa-avx512-512bit date=2026-09-17T05:25:30Z commit=db41481d dirty=0
phases=dequant ab arms=scalar simd-byte simd-int simd-split rounds=1 batches=1,2,4,8,512 jmh=-f 2 -wi 3 -i 5 -w 2 -r 2
Linux modeljars-granite41-hybrid-20260916 6.8.0-138-generic #138-Ubuntu SMP PREEMPT_DYNAMIC Fri Jul 31 22:41:49 UTC 2026 x86_64 x86_64 x86_64 GNU/Linux
Architecture:                            x86_64 CPU op-mode(s):                          32-bit, 64-bit Address sizes:                           40 bits physical, 57 bits virtual Byte Order:                              Little Endian CPU(s):                                  16 On-line CPU(s) list:                     0-15 Vendor ID:                               AuthenticAMD BIOS Vendor ID:                          QEMU Model name:                              AMD EPYC-Genoa Processor BIOS Model name:                         NotSpecified  CPU @ 2.0GHz BIOS CPU family:                         1 CPU family:                              25 Model:                                   17 Thread(s) per core:                      1 Core(s) per socket:                      16 Socket(s):                               1 Stepping:                                0 BogoMIPS:                                4799.99 Flags:                                   fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm rep_good nopl cpuid extd_apicid tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm cmp_legacy cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw topoext perfctr_core ssbd ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves avx512_bf16 clzero xsaveerptr wbnoinvd arat avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid fsrm Hypervisor vendor:                       KVM Virtualization type:                     full L1d cache:                               512 KiB (16 instances) L1i cache:                               512 KiB (16 instances) L2 cache:                                16 MiB (16 instances) L3 cache:                                32 MiB (1 instance) NUMA node(s):                            1 NUMA node0 CPU(s):                       0-15 Vulnerability Gather data sampling:      Not affected Vulnerability Indirect target selection: Not affected Vulnerability Itlb multihit:             Not affected Vulnerability L1tf:                      Not affected Vulnerability Mds:                       Not affected Vulnerability Meltdown:                  Not affected Vulnerability Mmio stale data:           Not affected Vulnerability Reg file data sampling:    Not affected Vulnerability Retbleed:                  Not affected Vulnerability Spec rstack overflow:      Mitigation; Safe RET Vulnerability Spec store bypass:         Mitigation; Speculative Store Bypass disabled via prctl Vulnerability Spectre v1:                Mitigation; usercopy/swapgs barriers and __user pointer sanitization Vulnerability Spectre v2:                Mitigation; Enhanced / Automatic IBRS; IBPB conditional; STIBP disabled; PBRSB-eIBRS Not affected; BHI Not affected Vulnerability Srbds:                     Not affected Vulnerability Tsa:                       Vulnerable: Clear CPU buffers attempted, no microcode Vulnerability Tsx async abort:           Not affected Vulnerability Vmscape:                   Not affected 
fma avx avx2 avx512f avx512dq avx512ifma avx512cd avx512bw avx512vl avx512_bf16 avx512vbmi avx512_vbmi2 avx512_vnni avx512_bitalg avx512_vpopcntdq 
openjdk version "25.0.4.1" 2026-08-18 LTS
OpenJDK Runtime Environment Temurin-25.0.4.1+1 (build 25.0.4.1+1-LTS)
OpenJDK 64-Bit Server VM Temurin-25.0.4.1+1 (build 25.0.4.1+1-LTS, mixed mode, sharing)
jvm=--add-modules jdk.incubator.vector -Xmx8g -Xms8g -XX:+AlwaysPreTouch -XX:+UseG1GC -Dvectors.maxBits=512
```

### Dequantisation alone (single thread; ms/op, M F32 elements/s, speedup vs scalar)

| format | shape | scalar | simd-byte | simd-int | simd-split |
|---|---|---:|---:|---:|---:|
| Q4_K | 8192x2560 | 15.49 ± 0.18 / 1354 M / **1.00x** | 3.95 ± 0.28 / 5310 M / **3.92x** | 3.53 ± 0.18 / 5938 M / **4.39x** | 3.54 ± 0.10 / 5916 M / **4.37x** |
| Q4_K | 2560x8192 | 15.46 ± 0.27 / 1357 M / **1.00x** | 3.74 ± 0.06 / 5614 M / **4.14x** | 3.73 ± 0.25 / 5621 M / **4.14x** | 3.55 ± 0.06 / 5911 M / **4.36x** |
| Q6_K | 8192x2560 | 34.50 ± 0.40 / 608 M / **1.00x** | 5.95 ± 0.15 / 3526 M / **5.80x** | 4.28 ± 0.13 / 4902 M / **8.07x** | 5.77 ± 0.14 / 3632 M / **5.98x** |
| Q6_K | 2560x8192 | 34.74 ± 0.57 / 604 M / **1.00x** | 5.79 ± 0.08 / 3624 M / **6.00x** | 4.24 ± 0.11 / 4941 M / **8.18x** | 5.79 ± 0.13 / 3619 M / **5.99x** |

Geometric-mean dequant speedup vs scalar: scalar 1.00x, simd-byte 4.88x, simd-int 5.89x, simd-split 5.11x

Candidate (given): simd-int

**(b) dequant >= 2.0x scalar, all 4 cells: PASS** — Q4_K 8192x2560 4.39x, Q4_K 2560x8192 4.14x, Q6_K 8192x2560 8.07x, Q6_K 2560x8192 8.18x

### Band A/B, multi-threaded (executor default)

ms/op ± 99.9% half-width. `new/old` = scalar band ms / arm band ms (>1 = faster than scalar dequant). `vs int` = integer ms / band ms.

| format | shape | batch | integer | band scalar | band simd-byte | band simd-int | band simd-split | simd-byte new/old | simd-int new/old | simd-split new/old | scalar vs int | simd-byte vs int | simd-int vs int | simd-split vs int |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q4_K | 2560x8192 | 1 | 0.54 ± 0.03 | 1.32 ± 0.06 | 0.37 ± 0.02 | 0.34 ± 0.01 | 0.36 ± 0.01 | **3.53x** | **3.82x** | **3.70x** | 0.41x | 1.44x | 1.56x | 1.51x |
| Q4_K | 2560x8192 | 2 | 1.20 ± 0.08 | 1.40 ± 0.03 | 0.45 ± 0.01 | 0.44 ± 0.02 | 0.44 ± 0.01 | **3.08x** | **3.16x** | **3.15x** | 0.86x | 2.65x | 2.72x | 2.71x |
| Q4_K | 2560x8192 | 4 | 1.70 ± 0.09 | 1.39 ± 0.02 | 0.48 ± 0.00 | 0.45 ± 0.01 | 0.45 ± 0.01 | **2.91x** | **3.13x** | **3.08x** | 1.22x | 3.55x | 3.82x | 3.76x |
| Q4_K | 2560x8192 | 8 | 3.37 ± 0.18 | 1.55 ± 0.02 | 0.64 ± 0.03 | 0.60 ± 0.01 | 0.63 ± 0.03 | **2.43x** | **2.56x** | **2.45x** | 2.18x | 5.30x | 5.57x | 5.34x |
| Q4_K | 2560x8192 | 512 | 224.07 ± 8.97 | 26.10 ± 0.78 | 25.04 ± 0.52 | 25.71 ± 0.90 | 25.10 ± 0.47 | **1.04x** | **1.02x** | **1.04x** | 8.59x | 8.95x | 8.72x | 8.93x |
| Q4_K | 8192x2560 | 1 | 0.53 ± 0.03 | 1.31 ± 0.07 | 0.38 ± 0.02 | 0.36 ± 0.01 | 0.40 ± 0.01 | **3.46x** | **3.67x** | **3.29x** | 0.40x | 1.39x | 1.48x | 1.33x |
| Q4_K | 8192x2560 | 2 | 1.15 ± 0.11 | 1.37 ± 0.09 | 0.51 ± 0.02 | 0.46 ± 0.02 | 0.47 ± 0.02 | **2.70x** | **3.00x** | **2.92x** | 0.84x | 2.26x | 2.51x | 2.45x |
| Q4_K | 8192x2560 | 4 | 1.64 ± 0.12 | 1.38 ± 0.09 | 0.50 ± 0.03 | 0.49 ± 0.03 | 0.48 ± 0.02 | **2.77x** | **2.84x** | **2.88x** | 1.18x | 3.28x | 3.36x | 3.41x |
| Q4_K | 8192x2560 | 8 | 3.13 ± 0.15 | 1.58 ± 0.07 | 0.71 ± 0.02 | 0.69 ± 0.02 | 0.69 ± 0.03 | **2.25x** | **2.29x** | **2.29x** | 1.98x | 4.44x | 4.54x | 4.53x |
| Q4_K | 8192x2560 | 512 | 211.40 ± 14.27 | 26.87 ± 1.35 | 25.56 ± 1.12 | 26.92 ± 1.20 | 26.28 ± 1.41 | **1.05x** | **1.00x** | **1.02x** | 7.87x | 8.27x | 7.85x | 8.04x |
| Q6_K | 2560x8192 | 1 | 0.80 ± 0.06 | 2.84 ± 0.03 | 0.59 ± 0.01 | 0.46 ± 0.04 | 0.56 ± 0.01 | **4.83x** | **6.20x** | **5.11x** | 0.28x | 1.37x | 1.76x | 1.45x |
| Q6_K | 2560x8192 | 2 | 1.71 ± 0.09 | 2.71 ± 0.34 | 0.67 ± 0.03 | 0.55 ± 0.03 | 0.65 ± 0.02 | **4.02x** | **4.93x** | **4.19x** | 0.63x | 2.53x | 3.10x | 2.64x |
| Q6_K | 2560x8192 | 4 | 1.79 ± 0.10 | 2.91 ± 0.06 | 0.69 ± 0.03 | 0.56 ± 0.03 | 0.68 ± 0.02 | **4.20x** | **5.17x** | **4.29x** | 0.62x | 2.59x | 3.19x | 2.64x |
| Q6_K | 2560x8192 | 8 | 3.40 ± 0.20 | 3.08 ± 0.03 | 0.84 ± 0.01 | 0.71 ± 0.01 | 0.84 ± 0.02 | **3.65x** | **4.33x** | **3.68x** | 1.10x | 4.03x | 4.78x | 4.06x |
| Q6_K | 2560x8192 | 512 | 233.40 ± 7.83 | 27.43 ± 0.42 | 25.21 ± 0.42 | 25.35 ± 1.18 | 25.41 ± 0.46 | **1.09x** | **1.08x** | **1.08x** | 8.51x | 9.26x | 9.21x | 9.19x |
| Q6_K | 8192x2560 | 1 | 0.85 ± 0.08 | 2.48 ± 0.22 | 0.58 ± 0.02 | 0.47 ± 0.03 | 0.58 ± 0.01 | **4.25x** | **5.22x** | **4.30x** | 0.34x | 1.45x | 1.79x | 1.47x |
| Q6_K | 8192x2560 | 2 | 1.56 ± 0.09 | 2.57 ± 0.13 | 0.67 ± 0.03 | 0.59 ± 0.05 | 0.68 ± 0.03 | **3.83x** | **4.35x** | **3.76x** | 0.61x | 2.32x | 2.63x | 2.28x |
| Q6_K | 8192x2560 | 4 | 1.81 ± 0.20 | 2.46 ± 0.08 | 0.69 ± 0.02 | 0.60 ± 0.03 | 0.69 ± 0.03 | **3.54x** | **4.09x** | **3.56x** | 0.74x | 2.61x | 3.01x | 2.62x |
| Q6_K | 8192x2560 | 8 | 3.41 ± 0.17 | 2.84 ± 0.16 | 0.89 ± 0.02 | 0.81 ± 0.06 | 0.91 ± 0.04 | **3.19x** | **3.49x** | **3.12x** | 1.20x | 3.82x | 4.18x | 3.74x |
| Q6_K | 8192x2560 | 512 | 217.83 ± 6.84 | 28.59 ± 1.14 | 25.91 ± 1.42 | 26.53 ± 0.81 | 26.94 ± 1.42 | **1.10x** | **1.08x** | **1.06x** | 7.62x | 8.41x | 8.21x | 8.09x |

**(c) simd-int: batch 1/4 faster than scalar in every cell: PASS (8/8); batch 512 <= 1.03x scalar in every cell: PASS (worst 1.002x)**
