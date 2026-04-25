# Licensing

**java-vectors** uses a split licensing model. The vast majority of the project
is available under the permissive **Apache License 2.0**. A small number of
commercially sensitive modules are licensed under the **Functional Source
License 1.1 (FSL-1.1-ALv2)**, which automatically converts to Apache 2.0 after
two years.

---

## Quick summary

| License | What it means |
|---|---|
| **Apache 2.0** | Use, modify, and redistribute freely — including in commercial products. |
| **FSL-1.1-ALv2** | Same freedoms as Apache 2.0 *except* you may not offer the software as a competing commercial product or service. Converts to Apache 2.0 on the Change Date (April 25, 2028). |

---

## Module classification

### Apache 2.0 modules

All core libraries, adapters, caching, VCR test harness, benchmarks, demos,
and documentation are **Apache 2.0**:

| Module | Description |
|---|---|
| `vectors-core` | SIMD distance kernels, vector types, similarity functions |
| `vectors-storage` | Off-heap memory, mmap, arena-based storage |
| `vectors-quantization` | SQ, PQ, BQ, RaBitQ, Extended RaBitQ, TurboQuant, NVQ |
| `vectors-hnsw` | HNSW graph index, concurrent builder, Fused ADC |
| `vectors-vamana` | Vamana / DiskANN graph index |
| `vectors-ivf` | IVF family indexes |
| `vectors-db` | Embedded vector database |
| `vectors-spring-ai` | Spring AI VectorStore adapter |
| `vectors-spring-boot-starter` | Spring Boot auto-configuration |
| `vectors-langchain4j` | LangChain4j EmbeddingStore adapter |
| `vectors-bench` | JMH benchmarks + ANN-Benchmarks harness |
| `vectors-vcr-*` | VCR test harness modules (core, semantic-db, serde, JUnit 5, TestNG, Spring AI, LangChain4j) |
| `vectors-cache` | Semantic caching layer |
| `vectors-cache-jcache` | JCache (JSR-107) integration |
| `vectors-cache-semantic-db` | Semantic cache backed by vectors-db |
| `vectors-cache-spring-ai` | Spring AI cache integration |
| `vectors-cache-langchain4j` | LangChain4j cache integration |
| `docs` | Antora documentation site |
| `demos:*` | All demo applications |

### FSL-1.1-ALv2 modules

These modules are licensed under the Functional Source License to protect
against competing commercial SaaS offerings:

| Module | Description | Change Date |
|---|---|---|
| `vectors-distributed` | Embedded cluster — gossip, sharding, scatter-gather | April 25, 2028 |
| `vectors-server` | HTTP server (Helidon SE 4 / Nima) | April 25, 2028 |
| `vectors-gpu` | GPU backend via Panama-FFM + NVIDIA cuVS | April 25, 2028 |

After the Change Date, these modules automatically become available under the
Apache License 2.0.

---

## What this means for you

### Using java-vectors as a library in your application

**No restrictions.** All library modules (`vectors-core` through `vectors-db`,
all adapters, all caches) are Apache 2.0. Embed them in any application —
open source or commercial — without limitation.

### Using FSL modules internally

**Permitted.** The FSL explicitly allows internal use, non-commercial
education, non-commercial research, and professional services.

### Offering java-vectors as a competing SaaS product

**Not permitted** for FSL modules until the Change Date. If you want to build a
commercial vector-database-as-a-service using `vectors-distributed`,
`vectors-server`, or `vectors-gpu`, you need a commercial license from
Integrallis Software.

### Contributing

Contributions to all modules are welcome. By contributing, you agree that your
contributions will be licensed under the same license as the module you are
contributing to.

---

## Commercial licensing

For commercial licensing of FSL modules, or for any licensing questions,
contact:

**Integrallis Software, LLC**
Email: licensing@integrallis.com

---

## License files

| File | Contents |
|---|---|
| [`LICENSE`](LICENSE) | Apache License 2.0 (full text) |
| [`LICENSE-FSL`](LICENSE-FSL) | Functional Source License 1.1, ALv2 Future License (full text) |
| `vectors-distributed/LICENSE` | Copy of LICENSE-FSL |
| `vectors-server/LICENSE` | Copy of LICENSE-FSL |
| `vectors-gpu/LICENSE` | Copy of LICENSE-FSL |
