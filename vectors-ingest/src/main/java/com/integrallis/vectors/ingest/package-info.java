/**
 * Streaming bulk ingestion pipeline: {@link com.integrallis.vectors.ingest.IngestSource source}
 * &rarr; {@link com.integrallis.vectors.ingest.Embedder embedder} &rarr; bounded queue &rarr;
 * {@link com.integrallis.vectors.ingest.BatchAccumulator batch} &rarr; {@link
 * com.integrallis.vectors.ingest.VectorSink vector} + {@link
 * com.integrallis.vectors.ingest.SidecartSink sidecart} commit, with a durable {@link
 * com.integrallis.vectors.ingest.IngestCursor cursor} for resumability.
 *
 * <p>The {@link com.integrallis.vectors.ingest.BulkIngestor} façade is the single public entry
 * point. It is reusable from demos, user code, integration tests, and future Studio "Import…"
 * flows. The pipeline is single-JVM, single-writer-per-source; multi-writer ingest, REST endpoints,
 * and 2PC across sinks are explicit non-goals (v2+).
 */
package com.integrallis.vectors.ingest;
