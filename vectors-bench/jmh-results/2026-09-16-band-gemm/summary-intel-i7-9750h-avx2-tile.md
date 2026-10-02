## intel-i7-9750h-avx2-tile

### Band tile shape (multi-threaded, band arm only)

| format | shape | batch | 3x3 ms/op | 4x4 ms/op | 4x4 speedup |
|---|---|---:|---:|---:|---:|
| Q4_K | 2560x8192 | 1 | 9.788 | 12.210 | 0.80x |
| Q4_K | 2560x8192 | 512 | 275.446 | 439.495 | 0.63x |
| Q4_K | 8192x2560 | 1 | 9.324 | 9.463 | 0.99x |
| Q4_K | 8192x2560 | 512 | 306.274 | 515.610 | 0.59x |
| Q8_0 | 2560x8192 | 1 | 5.717 | 19.616 | 0.29x |
| Q8_0 | 2560x8192 | 512 | 199.099 | 238.995 | 0.83x |
| Q8_0 | 8192x2560 | 1 | 6.266 | 10.154 | 0.62x |
| Q8_0 | 8192x2560 | 512 | 236.806 | 303.504 | 0.78x |

### Pre-registration 1 (single kernel replaces integer) on this run's data

- LLM-shape configurations with band >= 1.10x: 0 of 0
- configurations anywhere with band slower by > 5% (speedup < 0.952x): 0 of 0
- verdict for this ISA: **FAILS** (>=10% LLM-shape win AND no >5% loss elsewhere)
