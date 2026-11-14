#!/usr/bin/env bash
# Real-data IVF recall/QPS on both tags: RecallQpsBenchmark, fashion-mnist-784-euclidean (60k x 784,
# ground truth in the file), algorithm filter "ivf" (nlist in {sqrt n, 2 sqrt n, 4 sqrt n}, nprobe sweep).
# Each tag gets its own VECTORS_BENCH_DATA so the harness's checkpoint cache cannot skip the second
# tag's rows as "cached"; the hdf5 is symlinked from ~/data.
set -uo pipefail
export JAVA_HOME=/usr/lib/jvm/temurin-25-jdk-amd64; export PATH="$JAVA_HOME/bin:$PATH"
OUT=$HOME/out/realivf; mkdir -p $OUT
for t in v0.1.34 v0.1.35; do
  D=$HOME/benchdata-$t; mkdir -p $D/ann-benchmarks; ln -sf ~/data/fashion-mnist-784-euclidean.hdf5 $D/ann-benchmarks/
  (cd ~/$t && VECTORS_BENCH_DATA=$D ./gradlew :vectors-bench:recallQps -Pbench.dataset=fashion-mnist -Pbench.algo=ivf -Pbench.ivf.nprobe=1,2,4,8,16,32,64 --no-daemon -q > $OUT/$t.log 2>&1; echo "  $t exit=$? $(date -u +%T)"; mkdir -p $OUT/$t && cp $D/results/recall-qps.csv $D/results/recall-qps.json $OUT/$t/ 2>/dev/null; grep -cE "ivf_flat" $OUT/$t/recall-qps.csv 2>/dev/null | sed "s/^/  rows: /")
done
echo "REALIVF-DONE $(date -u +%FT%TZ)"
