# vectors-ivf

Inverted File (IVF) family indexes implemented in Java.

## Responsibility

- IVF_FLAT: inverted file index with flat (uncompressed) posting lists
- IVF_PQ: inverted file index with product-quantized posting lists
- Coarse quantizer training (k-means clustering)
- Multi-probe search across posting lists

## Dependencies

- `vectors-core`
- `vectors-storage`
- `vectors-quantization`
- `org.slf4j:slf4j-api` (logging)
