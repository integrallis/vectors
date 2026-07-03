# vectors-vamana

Vamana/DiskANN-style graph index implemented in Java.

## Responsibility

- Vamana flat graph index construction
- Robust pruning for graph quality
- Disk-optimized search for larger-than-memory datasets
- SSD-friendly access patterns with minimal random I/O

## Dependencies

- `vectors-core`
- `vectors-storage`
- `vectors-quantization`
- `org.slf4j:slf4j-api` (logging)
