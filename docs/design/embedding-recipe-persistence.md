# Persisting an embedding recipe: proposal

Written 2026-09-27. **Proposal, not implemented.** Read from `Manifest`, `FileFormat` and
`GenerationDirectory`; the constraints below are facts about the current format, everything after
"Design" is a recommendation.

Breaking the on-disk format is acceptable — the library is newly released and has no known production
deployments — which materially improves what is possible here. The weaker sidecar-only design that
avoids a version bump is recorded at the end as the alternative, because that trade-off should be a
decision rather than an accident.

## What the format currently allows

| fact | consequence for a recipe |
|---|---|
| `Manifest` is a **fixed 164-byte little-endian header**, offsets hard-coded, with `164 — (future extension area — version bump required to grow)` | a recipe is variable length (model id, two prefixes, an `extra` map) and **cannot live in the header** |
| `version != FileFormat.VERSION_MANIFEST` **throws**; it is exact-match, not a floor | any header change is a hard break for existing collections. Currently at 4, bumped before (tombstones took it to 4) |
| self-CRC32 over bytes `[0, 160)` at offset 160 | anything inside the header is integrity-checked; anything outside it is not |
| generations are written tmp-dir → fsync contents → fsync dir → atomic rename (`CURRENT.tmp` convention) | a recipe must join that ordering, not invent its own |

The two requirements pull in opposite directions: the recipe **must** be variable length, and it
**must** be integrity-bound or "attested" means nothing. So it cannot be only in the header, and it
cannot be only outside it.

## Design

**Split the recipe from its binding.** The body lives in a sidecar; a 32-byte hash of it lives inside
the CRC-protected header.

### 1. Body: `recipe.json`, beside the manifest

JSON rather than a binary record, deliberately:

* it is **variable length**, which the header cannot be;
* it is **human-inspectable** — an auditor can read what model produced a collection without our
  tooling, and that is a large part of why the feature exists;
* it **evolves**. A binary layout would need a version bump per added field, and a recipe will grow.

It carries the recipe fields, plus `schemaVersion`, plus the `recipeHash` as computed when written.

### 2. Binding: 36 bytes in the manifest, `VERSION_MANIFEST` → 5

```
 160     4    recipe present flag         0 = none, 1 = recipe.json present
 164    32    recipe hash                 SHA-256 of the canonical recipe form
 196     4    self CRC32                  CRC32 over bytes [0, 196)
 ------ ----
 200     -    (future extension area)
```

`HEADER_SIZE` 164 → 200 and `SELF_CRC_OFFSET` 160 → 196. The hash is **inside** the CRC'd region, so
the binding is protected by the same integrity check as the dimension and the metric.

**Why the hash and not the recipe body:** 32 bytes is fixed-width, and the body then cannot be swapped
without the manifest disagreeing.

### 3. Write ordering — sidecar first, always

```
write recipe.json.tmp → fsync → atomic rename → fsync directory
  → write manifest containing its hash → fsync
```

Crash between the two leaves an **orphan sidecar**, which is inert because no manifest references it.
The reverse order would leave a manifest anchoring a recipe that does not exist — a collection that
refuses to open. Order the failure so the recoverable case is the one that happens.

### 4. Open: verify, and refuse rather than guess

1. `flag == 0` and no sidecar → `UNKNOWN`. Normal, not an error.
2. `flag == 1`, sidecar missing → **refuse to open.** The collection claims provenance it cannot produce.
3. `flag == 1`, sidecar present → recompute the canonical hash and compare with the manifest.
   Mismatch → **refuse to open**, naming both hashes.
4. Cross-check `recipe.dimension` and `recipe.metric` against the manifest's own fields. Disagreement
   → refuse. `withRecipe` already rejects this at construction; the check is repeated at open because
   a file can be edited and a constructor cannot.
5. `flag == 0` but a sidecar exists → ignore it and log. That is the orphan from a crashed write, and
   honouring an unreferenced recipe is exactly how an attacker or an accident would inject one.

### 5. Schema evolution: unknown fields mean *cannot attest*

The subtle failure, and the one worth getting right. If a later version adds a recipe field, the
canonical form changes and so does the hash. An older reader must not:

* compute a hash over the fields it understands and report **tampering** — a false alarm; nor
* ignore the unknown fields and report **ATTESTED** — a false assurance, which is worse.

**Rule: `schemaVersion` higher than the reader understands ⇒ the recipe is readable for display but
reports `DECLARED`, never `ATTESTED`, and the collection opens.** The model name is still useful; the
attestation claim is withheld because it cannot be verified. Unknown fields never silently vanish and
never silently pass.

### 6. Immutability

A recipe may be attached to an **empty** collection. Once vectors exist, changing it is refused — it
invalidates every stored vector, so it is a migration, not an update. Weaviate makes vectorizer config
immutable for the same reason. The immutability is also load-bearing for our own model: it is what lets
a `Document` carry one hash, since every vector in the collection came from the same recipe.

## Threat model, stated honestly

This is **drift and accident protection, and it is not a signature.**

It defends well against: a redeployed application pointing at a different model; a config change nobody
noticed; a collection whose origin has been forgotten; a sidecar edited or replaced by hand.

It does **not** defend against someone who can write both files: they can recompute the hash and the
self-CRC. Real tamper-evidence needs the manifest signed by a key the reader trusts. That is a separate
feature and should not be implied by the word "attested" in marketing copy — which is why
`CollectionProvenance` already separates `ATTESTED` (a weights digest is recorded and re-derivation is
possible) from any claim about *who wrote the file*.

## Migration

`VERSION_MANIFEST = 5` refuses version-4 collections. Given no production deployments that is
acceptable, but the error must be actionable rather than a bare mismatch:

> Collection at `<path>` uses manifest version 4; this build reads version 5. Version 5 adds embedding
> recipe provenance. Rebuild the collection, or open it with vectors 0.1.23 to export.

## Testing, before this is believed

1. **Round-trip** every recipe field, including empty prefixes and an `extra` map, and assert the hash
   survives a write/read cycle.
2. **Corrupt the sidecar** by one byte and assert the collection refuses to open, naming both hashes.
3. **Delete the sidecar** with the flag set and assert refusal.
4. **Orphan sidecar** with the flag clear and assert it is ignored, not honoured.
5. **Crash between the two writes** — simulate by writing the sidecar and then not the manifest — and
   assert the collection opens as `UNKNOWN`.
6. **Forward compatibility**: hand-write a sidecar with `schemaVersion` one higher and an unknown field,
   and assert the collection opens, displays the model, and reports `DECLARED` rather than `ATTESTED`.
7. **Immutability**: attach to an empty collection (allowed), add a vector, attempt to change (refused).
8. **Dimension and metric disagreement** between sidecar and manifest, hand-edited, and assert refusal.

Test 6 is the one most likely to be skipped and the most valuable: it is the only one that fails if a
future reader silently mis-attests.

## The alternative, recorded so the trade-off is deliberate

**Sidecar only, no version bump.** Existing collections keep opening; absence of the sidecar means
`UNKNOWN`, which is already the semantics. Nothing breaks and no migration is needed.

What it gives up: the sidecar is **unbound**. It can be swapped or deleted with no disagreement
anywhere, so the provenance is only as good as the filesystem's own integrity, and "attested" would
overstate it. That was the right choice while the format had to stay compatible; with breaking changes
acceptable it is strictly worse, and the anchored design should be preferred.
