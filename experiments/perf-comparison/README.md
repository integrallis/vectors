# perf(comparison): benchmark v0.1.34 vs main

## Hypothesis
The 19 performance PRs merged since v0.1.34 should show measurable improvements
in:
1. HNSW search throughput (visited-set generation stamp, heap sizing, multi-start pooling, rescore heap)
2. Vamana search throughput (visited-set generation stamp, heap sizing, prefetch)
3. HNSW/Vamana build throughput (builder visited set, result reuse, RobustPruner scratch)
4. Commit pipeline latency (structural sharing in IdMapper/MetadataStore)
5. Cache hit latency (zero-allocation QvCache get, FilterHasher inline hash)

## Decision rule
Run the 5 benchmarks on both tags and compare. Accept the improvements if
search throughput improves by ≥ 3%, build throughput by ≥ 5%, or commit latency
reduces by ≥ 10%.

## Kill criterion
If any benchmark shows regression > 5% on the quiet VPS, investigate before
claiming victory.

## VPS provisioning (Hetzner)
```bash
# Create SSH key in Hetzner cloud (one-time)
hcloud ssh-key create --name benchmark-key --public-key-from-file ~/.ssh/id_ed25519.pub

# Spin up a CPX41 (8 vCPU, 16 GB) in US East
hcloud server create \
  --name vectors-benchmark \
  --type cpx41 \
  --image ubuntu-24.04 \
  --location ash \
  --ssh-key benchmark-key

# Get IP
IP=$(hcloud server ip vectors-benchmark)
echo "VPS IP: $IP"

# Wait for SSH, then install Java + Git
ssh -o StrictHostKeyChecking=no "root@$IP" <<'REMOTE'
  apt-get update && apt-get install -y openjdk-25-jdk git
  java -version
REMOTE

# Copy repo and run
tar czf - . | ssh "root@$IP" 'cd /root && mkdir vectors && tar xzf - -C vectors && cd vectors && ./experiments/benchmark-comparison.sh /root/results'

# Fetch results back
rsync -avz "root@$IP:/root/results/" ./benchmark-results/

# Destroy when done
hcloud server delete vectors-benchmark
```

## Local run (no VPS)
```bash
./experients/benchmark-comparison.sh results/$(date +%Y-%m-%d)-perf-comparison-local
```

## Benchmarks selected

| # | Benchmark | What it measures | Expected impact |
|---|---|---|---|
| 1 | `AnnSearchBenchmark.searchHnsw` | End-to-end HNSW search via facade | Visited-set O(1), heap sizing, multi-start dedup |
| 2 | `AnnSearchBenchmark.searchVamana` | End-to-end Vamana search via facade | Visited-set O(1), heap sizing |
| 3 | `IndexBuildBenchmark.buildHnsw` | HNSW graph construction | Builder visited set, result reuse, pruner scratch |
| 4 | `IndexBuildBenchmark.buildVamana` | Vamana graph construction | Builder visited set, result reuse, pruner scratch |
| 5 | `CommitPipelineBenchmark.copyOfAndStage` | In-memory commit + read | Structural sharing in IdMapper + MetadataStore |

## Machine spec for this comparison
*This section is filled in by the runner from lscpu / uname.*

## Results
*To be filled in after running on the quiet VPS.*
