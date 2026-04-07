# Vectors

A pure Java SIMD-native vector search framework leveraging the [Vector API](https://docs.oracle.com/en/java/javase/25/docs/api/jdk.incubator.vector/jdk/incubator/vector/package-summary.html) (`jdk.incubator.vector`) for hardware-accelerated approximate nearest neighbor (ANN) search.

## Requirements

- **JDK 25+**
- **Gradle 9.4+**

## Modules

| Module | Description |
|--------|-------------|
| [vectors-core](vectors-core/) | SIMD distance kernels, vector types, distance metrics |
| [vectors-storage](vectors-storage/) | Off-heap memory, mmap, arena-based storage |
| [vectors-quantization](vectors-quantization/) | Scalar, product, binary, and non-uniform vector quantization |
| [vectors-hnsw](vectors-hnsw/) | HNSW graph index (pure Java replacement for hnswlib) |
| [vectors-vamana](vectors-vamana/) | Vamana/DiskANN graph index |
| [vectors-ivf](vectors-ivf/) | IVF family indexes (IVF_FLAT, IVF_PQ) |
| [vectors-bench](vectors-bench/) | JMH benchmarks and ANN-Benchmarks harness |

## Dependency Graph

```
vectors-core              ← foundation, no internal deps
vectors-storage           ← core
vectors-quantization      ← core, storage
vectors-hnsw              ← core, storage, quantization
vectors-vamana            ← core, storage, quantization
vectors-ivf               ← core, storage, quantization
vectors-bench             ← all above
```

## Building

```bash
# Build all library modules
./gradlew build -x :docs:build

# Run tests
./gradlew test

# Code formatting
./gradlew spotlessApply

# Check formatting
./gradlew spotlessCheck
```

The Vector API is added automatically via `--add-modules jdk.incubator.vector` on compile and test tasks.

## License

Apache License 2.0
