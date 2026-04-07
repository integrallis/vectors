# vectors-storage

Off-heap memory management and on-disk storage for the Vectors library.

## Responsibility

- `MemorySegment`-based off-heap vector storage
- Memory-mapped file access via `Arena` for zero-copy reads
- Arena lifecycle management with spatial and temporal safety
- On-disk index and vector data formats

## Dependencies

- `vectors-core`
- `org.slf4j:slf4j-api` (logging)
