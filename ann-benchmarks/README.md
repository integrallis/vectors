# `vectors` × ANN-Benchmarks

Adapter that runs the [`vectors`](https://github.com/integrallis/vectors) JVM
vector database inside the [erikbern/ann-benchmarks](https://github.com/erikbern/ann-benchmarks)
recall-vs-QPS harness, so `vectors` is measured **in the same containerized
framework, on the same machine** as FAISS, hnswlib, Qdrant, Milvus, pgvector,
etc. — true apples-to-apples (unlike a cross-machine overlay).

This is ROADMAP **P1.2**. The files here are the upstream contribution staged in
this repo for maintenance; they map onto the ann-benchmarks layout
`ann_benchmarks/algorithms/vectors/`.

| File | Purpose |
|------|---------|
| `module.py` | HTTP `BaseANN` subclass — drives `vectors-server` over localhost (DB-server bracket) |
| `config.yml` | Grid for HNSW + VAMANA + IVF-PQ (graph indexes sweep efSearch; IVF rebuilds nprobe) |
| `Dockerfile` | `FROM ann-benchmarks` + JDK 25 + the prebuilt `vectors-server` (HTTP variant) |
| `module_jpype.py` | In-process `BaseANN` via jpype — calls `vectors-db` directly, zero IPC (library bracket) |
| `config.jpype.yml` | Same grid for the jpype variant (register as a separate `vectors-jpype` algorithm) |
| `Dockerfile.jpype` | `FROM ann-benchmarks` + JDK 25 + `pip install JPype1` + the dist's classpath |
| `smoke_test.py` | Local HTTP end-to-end check (HNSW + VAMANA) **without** the full harness |
| `docker-compose.yml` + `docker/` | Self-contained container that runs the jpype bridge in-process to verify it (`validate_jpype.py`) |

Both adapters cover **HNSW + VAMANA + IVF-PQ**. Graph indexes sweep `efSearch` at
query time; IVF sweeps `nprobe` by rebuilding (it is a build-time setting in
`vectors`).

## Which bracket this measures

ANN-Benchmarks has two de-facto categories: in-process **libraries** (hnswlib,
FAISS, ScaNN — no IPC) and **database servers** (Qdrant, Milvus, Weaviate,
pgvector, Vespa — the wrapper talks to a server over localhost, and that
round-trip is part of the measured latency). This adapter runs `vectors` as a
**server**, so it compares fairly against the database bracket — which matches
how `vectors` is positioned ("the default JVM vector database").

For the *library-bracket* comparison vs hnswlib/FAISS with zero IPC, use
`module_jpype.py` (`Dockerfile.jpype`, `config.jpype.yml`): it loads `vectors-db`
in-process via [jpype](https://jpype.readthedocs.io) and calls `VectorCollection`
directly — no server, no round-trip — starting the JVM with
`--add-modules=jdk.incubator.vector` and the `vectors-server` distribution's
`lib/*.jar` as the classpath. Register it as a separate `vectors-jpype` algorithm
so both brackets plot side by side.

> **Verification:** both adapters are verified end-to-end. HTTP — `smoke_test.py`,
> recall@10 = 1.0 against a live server. jpype — `docker-compose.yml` builds a
> JDK 25 + JPype1 runtime and runs `docker/validate_jpype.py` in-process,
> HNSW + VAMANA at recall@10 = 1.0:
>
> ```bash
> ./gradlew :vectors-server:installDist
> docker compose -f ann-benchmarks/docker-compose.yml run --build --rm jpype-validate
> # -> HNSW: recall@10 = 1.0000 ... VAMANA: recall@10 = 1.0000 ... JPYPE VALIDATION PASSED
> ```

> **Scope:** the adapter sweeps **HNSW** (M, efConstruction, efSearch) plus FLAT —
> the universal ANN-Benchmarks algorithm. Tuning Vamana / IVF-PQ over HTTP needs
> a few extra fields on the server's `CreateCollectionRequest` (today it only
> accepts `hnswM` / `hnswEfConstruction`); that is the natural next increment.

## Run it inside ANN-Benchmarks

```bash
# 1. Build the server distribution in the vectors repo.
cd /path/to/vectors
./gradlew :vectors-server:installDist

# 2. Stage the adapter into an ann-benchmarks checkout.
git clone https://github.com/erikbern/ann-benchmarks
mkdir -p ann-benchmarks/ann_benchmarks/algorithms/vectors
cp ann-benchmarks/{module.py,config.yml,Dockerfile} \
   ann-benchmarks/ann_benchmarks/algorithms/vectors/        # this repo's ann-benchmarks/ dir
cp -r vectors-server/build/install/vectors-server \
   ann-benchmarks/ann_benchmarks/algorithms/vectors/vectors-server-dist

# 3. Build + run (Docker; single host — NO Kubernetes/Terraform needed).
cd ann-benchmarks
python install.py --algorithm vectors           # builds ann-benchmarks-vectors image
python run.py --algorithm vectors --dataset sift-128-euclidean
python plot.py --dataset sift-128-euclidean      # recall-vs-QPS curve vs all others
```

`run.py` also runs the competitors (`--algorithm faiss`, `hnswlib`, `qdrant`, …)
in their own containers on the same host; `plot.py` overlays every curve.

### Build-from-source alternative for the Dockerfile

The shipped `Dockerfile` copies a host-built distribution (no Gradle or private
repo access needed at image-build time). To build from source instead, replace
the `COPY` with a multi-stage build (`FROM eclipse-temurin:25-jdk AS builder`,
`git clone` the repo, `./gradlew :vectors-server:installDist`, then `COPY
--from=builder`).

## Running on AWS

The benchmark is **single-node, single-threaded by design** (that is what makes
it comparable), so no Kubernetes is warranted. Provision **one** fixed instance
type (pick one and keep it constant for reproducibility — e.g. a compute/
memory-optimized box with enough RAM for the dataset), install Docker + Python,
clone ann-benchmarks with this adapter staged in, run the three commands above,
and copy `results/` to S3. A ~30-line `user-data` script or a small Terraform
module is all the infra needed.

## Verify locally (no harness)

```bash
./gradlew :vectors-server:installDist
VECTORS_SERVER_CMD=$PWD/vectors-server/build/install/vectors-server/bin/vectors-server \
  python3 ann-benchmarks/smoke_test.py
# -> recall@10 = 0.99xx ... SMOKE TEST PASSED
```

## Config schema caveat

ann-benchmarks's `config.yml` schema has shifted across releases (older versions
used `base_args` / `run_groups` with underscores). The keys here match the
current layout; verify against the exact commit you target before submitting a PR.
