```
label=intel-i7-9750h-avx2-screen-st date=2026-09-17T04:10:09Z commit=cb069be dirty=1
phases=ab-st arms=scalar simd-split simd-int rounds=1 batches=1,4 jmh=-f 1 -wi 2 -i 3 -w 1 -r 1
Darwin Brians-MacBook-Pro.local 25.6.0 Darwin Kernel Version 25.6.0: Fri Jul 31 19:11:49 PDT 2026; root:xnu-12377.161.14~5/RELEASE_X86_64 x86_64
Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz 12 32768 262144 34359738368 
FMA AVX1.0 AVX2 
openjdk version "25.0.3" 2026-04-21 LTS
OpenJDK Runtime Environment Temurin-25.0.3+9 (build 25.0.3+9-LTS)
OpenJDK 64-Bit Server VM Temurin-25.0.3+9 (build 25.0.3+9-LTS, mixed mode)
jvm=--add-modules jdk.incubator.vector -Xmx8g -Xms8g -XX:+AlwaysPreTouch -XX:+UseG1GC  -Dvectors.gguf.threads=12
```

Candidate (given): simd-split

### Band A/B, single-threaded

ms/op ± 99.9% half-width. `new/old` = scalar band ms / arm band ms (>1 = faster than scalar dequant). `vs int` = integer ms / band ms.

| format | shape | batch | integer | band scalar | band simd-int | band simd-split | simd-int new/old | simd-split new/old | scalar vs int | simd-int vs int | simd-split vs int |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q4_K | 2560x8192 | 1 | 37.29 ± 56.98 | 60.71 ± 352.89 | 19.07 ± 20.80 | 14.19 ± 27.01 | **3.18x** | **4.28x** | 0.61x | 1.96x | 2.63x |
| Q4_K | 2560x8192 | 4 | 108.79 ± 151.17 | 55.34 ± 331.43 | 22.95 ± 44.74 | 33.79 ± 26.96 | **2.41x** | **1.64x** | 1.97x | 4.74x | 3.22x |
| Q4_K | 8192x2560 | 1 | 38.39 ± 103.03 | 53.30 ± 19.56 | 17.35 ± 9.83 | 15.34 ± 4.55 | **3.07x** | **3.48x** | 0.72x | 2.21x | 2.50x |
| Q4_K | 8192x2560 | 4 | 132.50 ± 766.65 | 49.07 ± 21.89 | 21.61 ± 33.91 | 37.52 ± 134.43 | **2.27x** | **1.31x** | 2.70x | 6.13x | 3.53x |
| Q6_K | 2560x8192 | 1 | 56.68 ± 56.34 | 111.54 ± 609.93 | 24.97 ± 28.32 | 28.69 ± 107.29 | **4.47x** | **3.89x** | 0.51x | 2.27x | 1.98x |
| Q6_K | 2560x8192 | 4 | 101.89 ± 243.51 | 105.15 ± 804.19 | 50.78 ± 191.94 | 34.72 ± 52.69 | **2.07x** | **3.03x** | 0.97x | 2.01x | 2.94x |
| Q6_K | 8192x2560 | 1 | 61.60 ± 27.46 | 150.61 ± 912.75 | 24.53 ± 20.12 | 34.21 ± 36.30 | **6.14x** | **4.40x** | 0.41x | 2.51x | 1.80x |
| Q6_K | 8192x2560 | 4 | 107.93 ± 348.09 | 113.10 ± 404.44 | 29.30 ± 111.54 | 38.05 ± 142.89 | **3.86x** | **2.97x** | 0.95x | 3.68x | 2.84x |
