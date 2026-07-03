# JVM Community Launch Brief

This is the claims and messaging contract for `vectors` 0.1.0. Do not publish
the launch post until the external gates in `RELEASING.md` and the dated root
audit are complete.

## Position

`vectors` is an embedded Java 25 vector collection for applications that need
indexed local retrieval, metadata filters, and durable local state without
operating a separate database service.

It is not a distributed database, managed service, JVector replacement, Lucene
replacement, or universal performance leader. JVector and Lucene are mature
Java alternatives with different APIs and operational models.

The relevant ecosystem context is:

- Spring AI describes `SimpleVectorStore` as an in-memory implementation
  intended for testing:
  <https://docs.spring.io/spring-ai/reference/api/vectordbs.html>
- LangChain4j documents its simple in-memory store and its JVector integration:
  <https://docs.langchain4j.dev/integrations/embedding-stores/in-memory/> and
  <https://docs.langchain4j.dev/integrations/embedding-stores/jvector/>
- JVector provides mature pure-Java Vamana, disk-backed graphs, concurrent
  construction, and quantization: <https://github.com/datastax/jvector>
- Lucene provides Java HNSW and SIMD-aware vector utilities:
  <https://lucene.apache.org/core/10_3_1/core/org/apache/lucene/codecs/hnsw/package-summary.html>
  and
  <https://lucene.apache.org/core/10_3_1/core/org/apache/lucene/util/VectorUtil.html>
- The JDK 25 Vector API is still incubating:
  <https://docs.oracle.com/en/java/javase/25/docs/api/jdk.incubator.vector/jdk/incubator/vector/package-summary.html>

## Launch headline

> vectors 0.1.0: embedded vector search and generation-based persistence for
> Java 25

## Short description

> `vectors` 0.1.0 is an in-process Java 25 vector collection with FLAT, HNSW,
> Vamana, IVF-Flat, and IVF-PQ indexes; metadata filtering; generation-based
> mmap persistence; quantization; and Spring AI and LangChain4j adapters. The
> first release is single-process and CPU-only.

## Launch-post draft

We built `vectors` for Java applications that want indexed local retrieval and
durable local state without adding a separate vector-database service.

Release 0.1.0 provides FLAT, HNSW, Vamana, IVF-Flat, and IVF-PQ indexes behind
one collection API, with metadata filters, generation-based mmap persistence,
quantization, and Spring AI and LangChain4j adapters. It requires JDK 25 and the
incubating Vector API.

The boundaries are explicit: one process owns a persistent collection
directory; there is no cross-process writer lock or HA claim; distributed,
server, Studio, and GPU source modules are not part of the release.

The useful feedback is technical: persistence behavior, framework ergonomics,
index/quantizer behavior on real corpora, and reproducible benchmark results.

Links to include after release:

- repository and 0.1.0 release notes;
- Maven Central coordinates;
- a clean external quickstart that consumes Central artifacts;
- API documentation;
- raw benchmark methodology and output.

## Demonstration contract

The launch demo must start in a new directory and consume Maven Central rather
than Gradle project dependencies. It should demonstrate:

1. creating a persistent HNSW collection;
2. adding documents with metadata and committing;
3. closing and reopening the collection;
4. filtered search after reopen;
5. one Spring AI or LangChain4j adapter;
6. the required JDK flags and single-process boundary.

Run the demo on Linux and macOS before publication. Preserve its exact build
files and terminal output as release evidence.

## Claims allowed for 0.1.0

- embedded and in-process;
- Java 25;
- the indexes, quantizers, filters, persistence behavior, and adapters listed
  in the README;
- 15 Apache-2.0 artifacts published under the explicit allowlist;
- the exact release-workflow checks recorded in the dated audit.

## Claims not allowed

- “SOTA,” “best,” “fastest,” or “production-grade”;
- general sub-millisecond latency, QPS, vector-count, cost, or memory ceilings;
- distributed, HA, multi-process writer, GPU, or zero-native-call claims;
- numerical equivalence to FAISS, DiskANN, hnswlib, Lucene, or JVector;
- benchmark comparisons without committed raw output, hardware/JDK details,
  dataset hashes, parameters, warmup, and recall methodology.

## Release-day response checklist

- Reproduce bug reports from the tagged source and published coordinates.
- Ask benchmark reporters for dataset, dimension, metric, index parameters,
  JDK flags, hardware, warmup, and recall ground truth.
- Treat corruption, recovery, and wrong-result reports as release blockers.
- Correct inaccurate public claims in the same channel where they appeared.
- Route models questions separately; `models` remains a pre-alpha research
  runtime and is not part of the vectors reliability claim.
