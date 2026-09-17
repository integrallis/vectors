## genoa-avx512-512bit

### Sequential read bandwidth (MemoryBandwidthBenchmark)

| probe | threads | GB/s |
|---|---:|---:|
| sequentialByteXor (1 thread) | 1 | 33.79 |
| sequentialLongSum (1 thread) | 1 | 33.39 |
| sequentialByteXor (all threads) | 16 | 372.71 |
| sequentialLongSum (all threads) | 16 | 344.24 |

### Single-threaded (-Dvectors.gguf.parallel=false), existing shape

| format | shape (rows x cols) | batch | storage | integer ms/op | band ms/op | band speedup | weights GB/s int / band | % bandwidth int / band | mul-adds/s int / band (G) |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| Q4_K | 1024x2048 | 1 | heap | 0.675 ± 0.053 | 1.703 ± 0.061 | **0.40x** | 1.75 / 0.69 | 5.2 / 2.1 | 3.11 / 1.23 |
| Q4_K | 1024x2048 | 8 | heap | 4.265 ± 0.415 | 1.950 ± 0.038 | **2.19x** | 0.28 / 0.61 | 0.8 / 1.8 | 3.93 / 8.60 |
| Q4_K | 1024x2048 | 32 | heap | 17.184 ± 1.373 | 3.246 ± 0.211 | **5.29x** | 0.07 / 0.36 | 0.2 / 1.1 | 3.91 / 20.68 |
| Q6_K | 1024x2048 | 1 | heap | 1.017 ± 0.077 | 3.673 ± 0.256 | **0.28x** | 1.69 / 0.47 | 5.1 / 1.4 | 2.06 / 0.57 |
| Q6_K | 1024x2048 | 8 | heap | 4.482 ± 0.180 | 3.874 ± 0.137 | **1.16x** | 0.38 / 0.44 | 1.1 / 1.3 | 3.74 / 4.33 |
| Q6_K | 1024x2048 | 32 | heap | 17.787 ± 0.870 | 5.261 ± 0.277 | **3.38x** | 0.10 / 0.33 | 0.3 / 1.0 | 3.77 / 12.76 |
| Q8_0 | 1024x2048 | 1 | heap | 0.479 ± 0.027 | 0.846 ± 0.076 | **0.57x** | 4.65 / 2.63 | 13.9 / 7.9 | 4.37 / 2.48 |
| Q8_0 | 1024x2048 | 8 | heap | 2.206 ± 0.122 | 1.132 ± 0.052 | **1.95x** | 1.01 / 1.97 | 3.0 / 5.9 | 7.61 / 14.83 |
| Q8_0 | 1024x2048 | 32 | heap | 14.698 ± 0.739 | 2.390 ± 0.035 | **6.15x** | 0.15 / 0.93 | 0.5 / 2.8 | 4.57 / 28.08 |

### Single-threaded, Granite 4.1 3B FFN shapes

| format | shape (rows x cols) | batch | storage | integer ms/op | band ms/op | band speedup | weights GB/s int / band | % bandwidth int / band | mul-adds/s int / band (G) |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| Q4_K | 2560x8192 | 1 | heap | 6.761 ± 4.633 | 18.522 ± 9.223 | **0.37x** | 1.74 / 0.64 | 5.2 / 1.9 | 3.10 / 1.13 |
| Q4_K | 2560x8192 | 512 | heap | 2737.740 ± 832.914 | 303.247 ± 47.751 | **9.03x** | 0.00 / 0.04 | 0.0 / 0.1 | 3.92 / 35.41 |
| Q6_K | 2560x8192 | 1 | heap | 10.406 ± 7.896 | 38.324 ± 14.796 | **0.27x** | 1.65 / 0.45 | 5.0 / 1.3 | 2.02 / 0.55 |
| Q6_K | 2560x8192 | 512 | heap | 2826.767 ± 61.878 | 322.806 ± 55.860 | **8.76x** | 0.01 / 0.05 | 0.0 / 0.2 | 3.80 / 33.26 |
| Q8_0 | 2560x8192 | 1 | heap | 4.730 ± 2.230 | 9.823 ± 3.947 | **0.48x** | 4.71 / 2.27 | 14.1 / 6.8 | 4.43 / 2.13 |
| Q8_0 | 2560x8192 | 512 | heap | 1940.486 ± 272.195 | 287.491 ± 31.446 | **6.75x** | 0.01 / 0.08 | 0.0 / 0.2 | 5.53 / 37.35 |
| Q4_K | 8192x2560 | 1 | heap | 6.557 ± 0.279 | 18.557 ± 3.344 | **0.35x** | 1.80 / 0.64 | 5.4 / 1.9 | 3.20 / 1.13 |
| Q4_K | 8192x2560 | 512 | heap | 2743.089 ± 532.550 | 296.046 ± 15.844 | **9.27x** | 0.00 / 0.04 | 0.0 / 0.1 | 3.91 / 36.27 |
| Q6_K | 8192x2560 | 1 | heap | 10.094 ± 1.392 | 37.618 ± 0.356 | **0.27x** | 1.70 / 0.46 | 5.1 / 1.4 | 2.08 / 0.56 |
| Q6_K | 8192x2560 | 512 | heap | 2868.638 ± 169.294 | 310.261 ± 77.720 | **9.25x** | 0.01 / 0.06 | 0.0 / 0.2 | 3.74 / 34.61 |
| Q8_0 | 8192x2560 | 1 | heap | 4.678 ± 1.571 | 9.425 ± 13.904 | **0.50x** | 4.76 / 2.36 | 14.3 / 7.1 | 4.48 / 2.23 |
| Q8_0 | 8192x2560 | 512 | heap | 2001.272 ± 329.073 | 283.908 ± 67.910 | **7.05x** | 0.01 / 0.08 | 0.0 / 0.2 | 5.37 / 37.82 |

### Band dequantization only (single thread)

| format | shape | ms/op | F32 elements/s (M) | quantized GB/s | % of 1-thread bandwidth |
|---|---|---:|---:|---:|---:|
| Q4_K | 8192x2560 | 15.85 | 1323 | 0.744 | 2.2 |
| Q4_K | 2560x8192 | 15.43 | 1359 | 0.764 | 2.3 |
| Q6_K | 8192x2560 | 34.55 | 607 | 0.498 | 1.5 |
| Q6_K | 2560x8192 | 33.87 | 619 | 0.508 | 1.5 |
| Q8_0 | 8192x2560 | 5.55 | 3779 | 4.015 | 12.0 |
| Q8_0 | 2560x8192 | 5.57 | 3764 | 3.999 | 12.0 |

### Pre-registration 1 (single kernel replaces integer) on this run's data

- LLM-shape configurations with band >= 1.10x: 6 of 12
- configurations anywhere with band slower by > 5% (speedup < 0.952x): 9 of 21
  - ab-st-existing Q4_K 1024x2048 batch=1 heap: 0.40x
  - ab-st-existing Q6_K 1024x2048 batch=1 heap: 0.28x
  - ab-st-existing Q8_0 1024x2048 batch=1 heap: 0.57x
  - ab-st-llm Q4_K 2560x8192 batch=1 heap: 0.37x
  - ab-st-llm Q6_K 2560x8192 batch=1 heap: 0.27x
  - ab-st-llm Q8_0 2560x8192 batch=1 heap: 0.48x
  - ab-st-llm Q4_K 8192x2560 batch=1 heap: 0.35x
  - ab-st-llm Q6_K 8192x2560 batch=1 heap: 0.27x
  - ab-st-llm Q8_0 8192x2560 batch=1 heap: 0.50x
- verdict for this ISA: **FAILS** (>=10% LLM-shape win AND no >5% loss elsewhere)

### raw/genoa-avx512-512bit/crossover.json  (forks=2, jvmArgs=--add-modules jdk.incubator.vector --add-modules jdk.incubator.vector -Xmx8g -Xms8g -XX:+AlwaysPreTouch -XX:+UseG1GC -Dvectors.maxBits=512, jdk=25.0.4.1)

| format | shape | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 512 | T |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q4_K | 2560x8192 | 0.38 | 0.87 | 1.18* | 2.20* | 3.57* | 4.86* | 6.41* | 8.63* | 9.92* | 4 |
| Q4_K | 8192x2560 | 0.38 | 0.80 | 1.12* | 2.15* | 3.31* | 4.38* | 5.71* | 6.46* | 6.88* | 4 |
| Q6_K | 2560x8192 | 0.30 | 0.57 | 0.60 | 1.22* | 2.06* | 3.48* | 5.10* | 7.60* | 9.20* | 8 |
| Q6_K | 8192x2560 | 0.30 | 0.52 | 0.63 | 1.17 | 1.86* | 3.12* | 4.92* | 6.10* | 10.18* | 16 |
| Q8_0 | 2560x8192 | 0.62 | 1.29* | 1.67* | 2.45* | 3.39* | 4.09* | 4.84* | 5.91* | 6.37* | 2 |
| Q8_0 | 8192x2560 | 0.45 | 0.75 | 1.47* | 2.30* | 3.78* | 5.15* | 4.25* | 4.73* | 7.02* | 4 |
