#!/usr/bin/env bash
# IVF recall-vs-latency curves, both tags, identical synthetic corpus (TieredIvfRecallQpsBenchmark:
# mixture of 16 Gaussians, seed 0xC0FFEE, BuoyIndex routing = the path whose trainer changed from
# k-means(50 iters) to e-means(5 epochs) in 0.1.35). Latency at a fixed nprobe is not comparable
# across different partitions; recall at each nprobe together with its latency is.
set -uo pipefail
export JAVA_HOME=/usr/lib/jvm/temurin-25-jdk-amd64; export PATH="$JAVA_HOME/bin:$PATH"
OUT=$HOME/out/ivfcurve; mkdir -p $OUT
SHAPES=("128 200000 1000" "768 100000 500")
for r in 1 2; do for shape in "${SHAPES[@]}"; do set -- $shape; dim=$1; n=$2; q=$3
  for t in v0.1.34 v0.1.35; do
    name=$t-dim$dim-n$n-r$r
    (cd ~/$t && ./gradlew :vectors-bench:runTieredIvfRecallQps -Pbench.tieredIvf.dim=$dim -Pbench.tieredIvf.corpus=$n -Pbench.tieredIvf.queries=$q -Pbench.tieredIvf.nprobe=1,2,4,8,16,32,64 --no-daemon -q > $OUT/$name.log 2>&1; echo "  $name exit=$? rows=$(grep -cE '^[0-9]+ ' $OUT/$name.log) $(date -u +%T)")
  done
done; done
echo "IVFCURVE-DONE $(date -u +%FT%TZ)"
