## intel-i7-9750h-avx2-rep2

### Multi-threaded, existing benchmark shape

| format | shape (rows x cols) | batch | storage | integer ms/op | band ms/op | band speedup | weights GB/s int / band | % bandwidth int / band | mul-adds/s int / band (G) |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| Q4_K | 1024x2048 | 1 | heap | 1.062 ± 0.298 | 1.892 ± 1.347 | **0.56x** | 1.11 / 0.62 | n/a | 1.97 / 1.11 |
| Q4_K | 1024x2048 | 2 | heap | 1.908 ± 4.323 | 5.542 ± 13.397 | **0.34x** | 0.62 / 0.21 | n/a | 2.20 / 0.76 |
| Q4_K | 1024x2048 | 4 | heap | 3.287 ± 0.723 | 2.521 ± 2.481 | **1.30x** | 0.36 / 0.47 | n/a | 2.55 / 3.33 |
| Q4_K | 1024x2048 | 8 | heap | 7.987 ± 11.243 | 4.881 ± 2.455 | **1.64x** | 0.15 / 0.24 | n/a | 2.10 / 3.44 |
| Q4_K | 1024x2048 | 32 | heap | 12.944 ± 3.397 | 6.123 ± 1.271 | **2.11x** | 0.09 / 0.19 | n/a | 5.18 / 10.96 |
| Q6_K | 1024x2048 | 1 | heap | 1.673 ± 0.846 | 2.864 ± 1.231 | **0.58x** | 1.03 / 0.60 | n/a | 1.25 / 0.73 |
| Q6_K | 1024x2048 | 2 | heap | 2.493 ± 0.593 | 3.072 ± 0.687 | **0.81x** | 0.69 / 0.56 | n/a | 1.68 / 1.37 |
| Q6_K | 1024x2048 | 4 | heap | 2.941 ± 0.832 | 3.741 ± 0.496 | **0.79x** | 0.58 / 0.46 | n/a | 2.85 / 2.24 |
| Q6_K | 1024x2048 | 8 | heap | 5.410 ± 0.881 | 9.089 ± 4.853 | **0.60x** | 0.32 / 0.19 | n/a | 3.10 / 1.85 |
| Q6_K | 1024x2048 | 32 | heap | 23.433 ± 9.425 | 9.300 ± 9.533 | **2.52x** | 0.07 / 0.18 | n/a | 2.86 / 7.22 |
| Q8_0 | 1024x2048 | 1 | heap | 3.039 ± 0.498 | 3.967 ± 0.557 | **0.77x** | 0.73 / 0.56 | n/a | 0.69 / 0.53 |
| Q8_0 | 1024x2048 | 2 | heap | 1.649 ± 0.424 | 4.894 ± 1.116 | **0.34x** | 1.35 / 0.46 | n/a | 2.54 / 0.86 |
| Q8_0 | 1024x2048 | 4 | heap | 3.550 ± 3.167 | 7.664 ± 1.531 | **0.46x** | 0.63 / 0.29 | n/a | 2.36 / 1.09 |
| Q8_0 | 1024x2048 | 8 | heap | 3.514 ± 4.896 | 7.805 ± 2.766 | **0.45x** | 0.63 / 0.29 | n/a | 4.77 / 2.15 |
| Q8_0 | 1024x2048 | 32 | heap | 8.974 ± 2.899 | 24.637 ± 37.941 | **0.36x** | 0.25 / 0.09 | n/a | 7.48 / 2.72 |

### Multi-threaded, Granite 4.1 3B FFN shapes

| format | shape (rows x cols) | batch | storage | integer ms/op | band ms/op | band speedup | weights GB/s int / band | % bandwidth int / band | mul-adds/s int / band (G) |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| Q4_K | 2560x8192 | 1 | heap | 6.627 ± 1.213 | 10.753 ± 1.905 | **0.62x** | 1.78 / 1.10 | n/a | 3.16 / 1.95 |
| Q4_K | 2560x8192 | 512 | heap | 1386.784 ± 114.362 | 249.821 ± 20.244 | **5.55x** | 0.01 / 0.05 | n/a | 7.74 / 42.98 |
| Q6_K | 2560x8192 | 1 | heap | 7.055 ± 3.444 | 16.315 ± 1.970 | **0.43x** | 2.44 / 1.05 | n/a | 2.97 / 1.29 |
| Q6_K | 2560x8192 | 512 | heap | 2015.019 ± 208.074 | 258.227 ± 5.371 | **7.80x** | 0.01 / 0.07 | n/a | 5.33 / 41.58 |
| Q8_0 | 2560x8192 | 1 | heap | 4.166 ± 0.436 | 6.211 ± 0.390 | **0.67x** | 5.35 / 3.59 | n/a | 5.03 / 3.38 |
| Q8_0 | 2560x8192 | 512 | heap | 1556.813 ± 203.139 | 241.066 ± 8.684 | **6.46x** | 0.01 / 0.09 | n/a | 6.90 / 44.54 |
| Q4_K | 8192x2560 | 1 | heap | 8.393 ± 20.526 | 11.855 ± 2.147 | **0.71x** | 1.41 / 1.00 | n/a | 2.50 / 1.77 |
| Q4_K | 8192x2560 | 512 | heap | 1432.584 ± 188.525 | 227.794 ± 28.810 | **6.29x** | 0.01 / 0.05 | n/a | 7.50 / 47.14 |
| Q6_K | 8192x2560 | 1 | heap | 13.642 ± 17.929 | 17.537 ± 3.287 | **0.78x** | 1.26 / 0.98 | n/a | 1.54 / 1.20 |
| Q6_K | 8192x2560 | 512 | heap | 2079.373 ± 200.964 | 233.114 ± 25.320 | **8.92x** | 0.01 / 0.07 | n/a | 5.16 / 46.06 |
| Q8_0 | 8192x2560 | 1 | heap | 4.054 ± 0.623 | 7.344 ± 1.437 | **0.55x** | 5.50 / 3.03 | n/a | 5.17 / 2.86 |
| Q8_0 | 8192x2560 | 512 | heap | 1501.112 ± 117.749 | 211.282 ± 23.927 | **7.10x** | 0.01 / 0.11 | n/a | 7.15 / 50.82 |

### Pre-registration 1 (single kernel replaces integer) on this run's data

- LLM-shape configurations with band >= 1.10x: 6 of 12
- configurations anywhere with band slower by > 5% (speedup < 0.952x): 17 of 27
  - ab-mt-existing Q4_K 1024x2048 batch=1 heap: 0.56x
  - ab-mt-existing Q4_K 1024x2048 batch=2 heap: 0.34x
  - ab-mt-existing Q6_K 1024x2048 batch=1 heap: 0.58x
  - ab-mt-existing Q6_K 1024x2048 batch=2 heap: 0.81x
  - ab-mt-existing Q6_K 1024x2048 batch=4 heap: 0.79x
  - ab-mt-existing Q6_K 1024x2048 batch=8 heap: 0.60x
  - ab-mt-existing Q8_0 1024x2048 batch=1 heap: 0.77x
  - ab-mt-existing Q8_0 1024x2048 batch=2 heap: 0.34x
  - ab-mt-existing Q8_0 1024x2048 batch=4 heap: 0.46x
  - ab-mt-existing Q8_0 1024x2048 batch=8 heap: 0.45x
  - ab-mt-existing Q8_0 1024x2048 batch=32 heap: 0.36x
  - ab-mt-llm Q4_K 2560x8192 batch=1 heap: 0.62x
  - ab-mt-llm Q6_K 2560x8192 batch=1 heap: 0.43x
  - ab-mt-llm Q8_0 2560x8192 batch=1 heap: 0.67x
  - ab-mt-llm Q4_K 8192x2560 batch=1 heap: 0.71x
  - ab-mt-llm Q6_K 8192x2560 batch=1 heap: 0.78x
  - ab-mt-llm Q8_0 8192x2560 batch=1 heap: 0.55x
- verdict for this ISA: **FAILS** (>=10% LLM-shape win AND no >5% loss elsewhere)
