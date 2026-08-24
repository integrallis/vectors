# Needle CQ compatibility fixtures

These four fixtures were generated from the Apache-2.0 Needle source at revision
`7bd8a63` by importing the reference functions directly:

- `needle.model.export._cq_pack`
- `needle.model.export._cq_unpack`
- `needle.model.quantize._cq_codebook_np`

The generator uses NumPy seed `20260824`, 3 rows, 133 logical columns, and groups of
128. The non-aligned logical width deliberately exercises final-group padding. The
input is a deterministic sine/cosine vector. Each binary fixture contains its geometry,
reference codebook, packed codes, FP16 norms, input vector, and the F32 result from
`_cq_unpack(...) @ input`.

| Resource | Decoded SHA-256 |
| --- | --- |
| `needle-cq2-v7bd8a63.b64` | `122e46e23f4fd44d7ecc29a8ccfda69389e5bc42134863b448ecbbcddf03ef1b` |
| `needle-cq3-v7bd8a63.b64` | `e405f999513b34b19f4b1988f80ee38e5caf1f7be811b1ce030eb0b0ad954926` |
| `needle-cq4-v7bd8a63.b64` | `ec16ec3ee81abb0cf3e31176ad7bbf43f607d54b161ad148ec85f0c9212ff3b5` |
| `needle-ternary-v7bd8a63.b64` | `6b53746785cfbbfee07ab5caf9bbbd999e3641251cf9d02f6731925624ce6cb3` |

The test verifies these checksums before reading expected values, so changing a fixture
cannot silently redefine compatibility.
