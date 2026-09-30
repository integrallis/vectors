# Embedding recipes, staleness, and where the line sits between metadata and compound vectors

Written 2026-09-27. **The "what exists today" section is read from the source; everything after it is
proposal.** Nothing here is measured.

## What exists today

| concept | status |
|---|---|
| `Document` | `record Document(String id, float[] vector, String text, Map<String, MetadataValue> metadata)` |
| embedding model provenance | **absent** — `Embedder` has `name()` and `dimension()`, neither is persisted anywhere |
| timestamps (first embedded, last embedded, source modified) | **absent** |
| content hash for staleness | **absent** |
| ULID / time-sortable id segment | **absent.** `Segment` exists only as write-ahead-log segments in `vectors-storage`, which is unrelated |
| metadata types | `sealed MetadataValue`: `Str`, `Num(double)`, `Bool`, `Tags(List<String>)` |
| filters | `sealed Filter`: `All`, `Eq`, `NumericRange`, `In`, `And`, `Or`, `Not` |
| collection-level config | `VectorCollectionConfig(dimension, metric, …, normalizeCosineVectors, quantizedOnly)` — **the seam for a recipe already exists and is half-populated**: the geometry is recorded, the model is not |
| multiple / named vectors per document | **not supported** — one `float[]` per document |
| multi-signal retrieval | `vectors-hybrid` fuses *retrievers* (`RRFFusion`, `WeightedFusion`, `MaximalMarginalRelevance`) — not compound vectors |

## The hole this opens, stated plainly

**A collection's only tie between its vectors and the model that produced them is the dimension.**
Two unrelated 768-dimension models produce mutually meaningless vectors, and the store will mix them
without complaint. A query embedded with model A, searched against documents embedded with model B,
returns confidently ranked nonsense. There is no error, no warning, and no way to notice after the
fact, because nothing recorded which model wrote which vector.

**Dimension is not identity.** That is the actual defect; recipes are the fix, not a feature.

## What a recipe has to carry

Not a wish list — each of these changes the resulting vector, so omitting any of them makes the
recipe an incomplete identity:

| field | why it is load-bearing |
|---|---|
| model identity: name, version, and a content digest | the whole point. ModelJars already pins exactly this — a Maven coordinate plus SHA-256 — so a recipe should *reference* a ModelJars coordinate rather than invent a parallel notion of model identity |
| **role-specific instruction prefix** | instruction-tuned embedders are **asymmetric**: Nomic wants `search_document:` when indexing and `search_query:` when searching. Same model, same text, different vector. This is the field most often got wrong, and it belongs in the recipe rather than at each call site |
| pooling | mean / CLS / last-token change the vector entirely |
| normalisation | whether vectors are L2-normalised decides whether cosine and dot product agree |
| truncation policy and max tokens | silently decides what part of a long document was actually seen |
| tokenizer identity | usually implied by the model, not always |
| similarity function | `SimilarityFunction` exists (`COSINE`, `DOT_PRODUCT`, `EUCLIDEAN`) but does not appear to be recorded against a collection |
| stored dtype / quantisation | a quantised vector is a different vector |

## Staleness: the hash alone is not enough

The obvious design — hash the content, re-embed when it changes — is wrong on its own, and the reason
matters.

**Hash what was actually embedded**, after chunking, after the role prefix, after truncation — not the
raw source. Then a recipe change alters the embedded input even when the source is byte-identical.

So the re-embed rule takes two inputs:

```
stale  ⇔  contentHash(embedded input) changed   OR   recipeHash changed
```

Which means the useful unit is not the document but the **(document, recipe) pair**. Change the prefix
or bump the model, and every vector in the collection is stale by construction — that is correct, and a
system that only hashes content will confidently serve vectors from the previous model forever.

**Timestamps are operational, the hash is authoritative.** `firstEmbeddedAt` and `lastEmbeddedAt` answer
"re-embed anything older than X" and support audit; they should never be the correctness test.

**One trap to avoid, learned elsewhere in this stack:** a timestamp that a *read* mutates destroys
measurement reproducibility. The memory project already hit this — `recall` wrote back access times, so
earlier queries changed the ranking later ones saw and a sweep partly measured its own query order.
Embedding timestamps must be write-time only.

## The ULID idea, sharpened

Time-sortable identifiers are worth having, but not on the document.

What they buy: recency filtering without a metadata index, insert locality, range scans by time, and —
the real prize — **pruning or compacting a whole id range** without touching a graph.

The trap: if the document id encodes time, the id *is* the timestamp, and re-embedding in place either
rewrites the id (breaking every reference to it) or preserves a now-false creation time.

**So: the document id stays stable content identity, and the ULID belongs on the embedding version.**
A document has one id; each `(document, recipe)` pair has an embedding record with its own
time-sortable version. That composes with the staleness rule above instead of fighting it.

## Where the line sits between metadata and compound vectors

The sharpest formulation I can give:

> **Filters prune. Vectors rank.** If an attribute answers *"is this candidate admissible?"* it is
> metadata. If it answers *"how similar is this?"* it is a vector.

### Why packing engineered features into dimensions is a trap

1. **Concatenation fixes the weighting at index time, permanently.** Under cosine over two
   unit-normalised blocks, each block's contribution is decided by its norm when it was written. You
   cannot re-weight at query time without rebuilding. Two *separate* vectors fused at query time —
   which `vectors-hybrid` already does — can be re-weighted per query, per tenant, per experiment.
2. **Scalar features break the metric.** Cosine treats dimensions as directionally comparable. Packing
   "recency in days" into a dimension gives it a scale that silently dominates or vanishes, and its
   semantics are *monotonic* (larger is better), which is not what cosine computes. As a filter,
   "last 30 days" is exact and cheap; as a dimension it becomes approximate and inseparable from
   meaning.
3. **The ANN graph is built on the compound metric.** Re-weighting a feature means a full rebuild.
   Metadata filters never touch the graph.
4. **Exactness is lost for no gain.** Tenancy, ACLs and ranges must be exact. An approximate
   nearest-neighbour search over a dimension encoding a tenant id is a correctness bug waiting to
   happen.

### When a compound vector is genuinely right

The test that separates the two cases: **did a model define the joint space, or did you?**

* **Yes, a model did** — one vector. A jointly trained multi-modal encoder, ColBERT-style multi-vector
  late interaction, Matryoshka embeddings where truncation is principled because the model was trained
  for it.
* **No, you concatenated two independent models' outputs** — you have invented an untrained metric and
  fixed its weights by accident. Use two named vectors and fuse at query time.

### The resulting shape

| signal | where it goes |
|---|---|
| tenancy, ACL, language, source, mime, status | metadata + `Filter` |
| time bounds, numeric ranges, popularity thresholds | metadata + `NumericRange` |
| several semantic views you may want to weight differently (title vs body, summary vs full text) | **named vectors, each with its own recipe**, fused at query time |
| a jointly trained multi-modal or late-interaction space | one compound vector |
| engineered scalars (recency score, click rate) | **metadata, applied as re-ranking after retrieval** — never dimensions |

This makes "more complex recipes" tractable: complexity goes into *how many named vectors a document
has and what recipe produced each*, not into making one vector carry more meaning than a metric can
express.

## Gaps worth noting in passing

* `MetadataValue.Num` is a `double`, so time is an untyped double comparison. Millisecond epochs are
  exact to 2^53 (about 285,000 years), so this works — but a first-class temporal kind would make
  time-range filters legible and let a storage layer index them as time.
* `Filter` has no `Exists`, no prefix or glob match, and no field-to-field comparison. `Exists` in
  particular becomes necessary the moment documents have optional named vectors.
* ~~Nothing records `SimilarityFunction` per collection~~ — **corrected 2026-09-27**: it is recorded,
  in `VectorCollectionConfig.metric`, along with `normalizeCosineVectors` and `quantizedOnly`. So the
  geometry half of a recipe is already persisted per collection and only the model half is missing.
  That makes this a smaller change than first written, and it names where the recipe belongs.

## What I would want before building any of it

Research first, per the working agreement — none of the above is measured:

1. **Read how others draw this line**, specifically Qdrant's named vectors, Weaviate's named vectors
   and modules, Vespa's tensor fields and ranking expressions, and pgvector plus a relational filter.
   The question to answer is not what they *offer* but what their users get wrong.
2. **Establish whether the mixed-model hazard is reachable in practice** with a test that indexes with
   two different 768-dimension models and shows the ranking is confident nonsense. That converts the
   argument above from reasoning into a demonstration, which is the standard this project holds itself
   to.
3. **Decide the recipe's serialised form before its API**, because it becomes a persisted identity that
   must survive version upgrades — and a recipe that cannot be compared across versions cannot answer
   the staleness question it exists to answer.
