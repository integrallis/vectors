# vectors-bench

[![MFCQI](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/integrallis/vectors/main/vectors-bench/.github/badges/mfcqi.json)](https://github.com/integrallis/mfcqi-java)

Benchmarks for the Vectors library.

## Responsibility

- JMH microbenchmarks for SIMD distance kernels
- ANN-Benchmarks-aligned macrobenchmark harness measuring recall@k, QPS, and latency
- Comparative benchmarks across index types and quantization strategies

## Running

```bash
./gradlew :vectors-bench:jmh
```

## Dependencies

- All Vectors library modules
- `org.openjdk.jmh:jmh-core` (benchmarking framework)
