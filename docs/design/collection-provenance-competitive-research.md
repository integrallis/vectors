# Collection genesis and maintenance: what the vector databases do, and where our moat actually is

Written 2026-09-27. **Provenance: `read`** — vendor documentation and community sources, not measured
here. Nothing in this document has been reproduced on our own harness, and the competitive claims are
as of this date only.

## The question

If the genesis and maintenance of a collection should live *with* the collection, who already does
that, what do they leave out, and is what remains a feature an enterprise would pay for?

## What the field does

| system | is the recipe part of the collection? | re-embedding on change |
|---|---|---|
| **Vespa** | **Yes, most completely.** The embedder is an *indexing statement in the schema*: `field embeddings type tensor<bfloat16>(x[384]) { indexing { input title \| embed embedderId \| attribute \| index } }` | **Automatic.** A change to the indexing pipeline triggers reindexing, which re-feeds documents through the new chain and recomputes derived fields |
| **Weaviate** | **Yes.** `vectorizer` + `moduleConfig` live in the collection schema, per named vector. Deliberately **immutable**, because changing it invalidates the vector space | Manual. Add a new named vector (existing objects stay unpopulated) or migrate behind a **collection alias** |
| **Pinecone** | **Partly.** Indexes created "for a model" bind an embedding model to the index; otherwise it is metadata convention, with bulk update/delete by metadata to backfill or prune a `model_version` | Manual |
| **Qdrant** | **No first-class notion.** The documented pattern is convention: payload or collection metadata fields such as `embedding_model` and `pipeline_version`, compared against current config to decide staleness | DIY |
| **Milvus** | **No.** Scalar fields and `enable_dynamic_field` carry whatever you choose to put there | DIY |
| **pgvector** | **No.** It is a column type | DIY |

**The split is bring-your-own-vectors versus own-the-embedding.** Vespa and Weaviate own the
embedding, so they can and do record the recipe. The BYOV systems accept a `float[]` and cannot know
what made it, so provenance becomes a metadata convention the application must remember to follow.

**The best-practice literature independently converges on the same field list** we derived from first
principles — source document id, content hash, embedding model, model version, dimension, pipeline
revision. That is reassuring about the design and also means **the field list is not a differentiator.**
Anyone can store a model name.

## So the moat is not "we record the model"

Two systems already do, and the pattern is documented convention everywhere else. A recipe on its own
is table stakes we are currently *behind* on, not ahead.

What no one in that table can do follows from what is actually different about this stack:

### 1. Attested identity, not declared identity

Weaviate's vectorizer is a **module name plus config** pointing at a service. Vespa's embedder is a
**component id** in a deployment. Both record *what you told them you used*. Neither can demonstrate
which weights produced a given vector — swap the model behind the endpoint, or ship a new build of the
same component id, and the recorded recipe is unchanged while every vector afterwards is different.

We can do better because our models are **ModelJars artifacts with cryptographic identity**: a Maven
coordinate, a pinned upstream revision, and a SHA-256 of the actual GGUF, with adapter digests for a
composition. A recipe referencing that is checkable rather than merely stated.

### 2. Reproducibility — the claim nobody else can make

This is the one the last week earned and it is worth being precise about, because it is easy to
overclaim.

Embedding services are not bit-reproducible in general: their numerics vary with hardware, library
version and batch shape. We have just removed exactly that class of variation from our own path — the
attention reductions and the model-path float reductions are pinned to a host-independent shape, so
**same model + same recipe + same input ⇒ the same bits, on any machine.**

That converts a documentation claim into a testable property: **re-run the embedding and compare.** A
hosted vector database cannot offer it, because it does not control the numerics and its provider does
not promise them.

### 3. The audit question that is currently unanswerable

> *"Prove these embeddings were produced by the model you documented."*

In every system in the table above, the honest answer is "here is the configuration we recorded". With
attested identity plus reproducibility the answer becomes: **re-derive a sample and show the bits
match.** That is verification rather than attestation-by-assertion, and it is the shape of evidence a
regulated review actually accepts.

Concrete enterprise value, in the order a buyer would care:

* **detect a silent model swap** — a supply-chain and change-control concern, invisible today;
* **prove lineage on demand** — sampling re-derivation instead of trusting a config field;
* **reproducible re-embedding** — a migration that can be rolled back to bit-identical prior state;
* **no egress** — the embedding never leaves the JVM, which is often the reason a hosted vector
  database was refused in the first place.

**Stated honestly: 1 and 3 do not exist yet, and 2 exists in the model path but has not been
demonstrated end-to-end for embeddings.** The moat is available, not built. What is built is the
numerical foundation that makes it possible, which is the hard part and the part competitors cannot
retrofit without owning their own inference.

## Exposing it through Spring AI and LangChain4j

Read from our own adapters, so this part is fact rather than plan:

* **Spring AI** — `JavaVectorsVectorStore.builder(embeddingModel, collection)` holds the
  `EmbeddingModel` and calls `embeddingModel.embed(...)` itself. **The model is in scope at write time
  and is currently discarded.** The adapter can therefore record provenance with no application change.
* **LangChain4j** — `JavaVectorsEmbeddingStore.add(Embedding, TextSegment)` never sees a model: the
  framework embeds before calling the store. **Provenance cannot be inferred and must be supplied**, as
  a recipe on the store builder.

That asymmetry sets the design:

1. **Neither framework learns a new concept.** Our adapters record; the frameworks stay unaware. This
   matters because both already have *construction-time* recipes — LangChain4j's
   `EmbeddingStoreIngestor` chains transformer, splitter, model and store; Spring AI has the
   reader/transformer/writer ETL trio with `EmbeddingOptions` — and in both, that recipe lives in
   application wiring and is never persisted with the data. So both frameworks permit a model swap that
   the store accepts silently, and neither can answer "is this vector stale?".
2. **Two honest tiers, never conflated.** `ATTESTED` when the model is a ModelJars artifact and the
   digest is known; `DECLARED` when it is a hosted provider and only a name and dimension are
   available. A collection reports which tier it holds. Pretending an OpenAI endpoint gives attested
   provenance would be the same category error as quoting a vendor benchmark as our own measurement.
3. **Staleness is a collection query, not a framework feature.** "Which documents need re-embedding"
   answered against `(contentHash, recipeHash)`, plus a maintenance operation to act on it. Neither
   framework has anywhere to put this, which is precisely why it belongs at our layer.
4. **A mismatch fails loudly on open.** Weaviate's choice to make vectorizer config immutable is the
   right instinct: silently accepting incompatible vectors is worse than refusing to start.

## What to do before building

1. **Demonstrate the hazard rather than assert it.** Index the same corpus with two different
   768-dimension models, query with one, and show the ranking is confident nonsense. Every argument
   above rests on that being reachable in practice.
2. **Prove the reproducibility claim for embeddings specifically**, on two different host classes, the
   way the model path was just proved. Without that, item 2 of the moat is a belief.
3. **Settle the serialised recipe form before the API**, since it becomes a persisted identity that
   must survive upgrades — a recipe that cannot be compared across versions cannot answer the question
   it exists for.
4. **Check the alias pattern.** Weaviate's collection aliases for reversible migration look like the
   right operational primitive and we have no equivalent; worth evaluating before inventing one.

## Sources

Read 2026-09-27: Weaviate vectorizer migration and vector-config documentation and community threads;
Vespa reindexing, schema and embedding documentation; Qdrant points/payload and incremental-embedding
documentation; Pinecone integrated-embedding and bulk-metadata-operations documentation; Milvus schema
documentation; plus practitioner writing on embedding drift and staleness. URLs are in the pull request
discussion rather than inlined here, since documentation URLs move.
