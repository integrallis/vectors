# vectors-bench

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
