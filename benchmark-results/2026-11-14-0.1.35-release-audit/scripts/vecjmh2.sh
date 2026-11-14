#!/usr/bin/env bash
set -uo pipefail
export JAVA_HOME=/usr/lib/jvm/temurin-25-jdk-amd64; export PATH="$JAVA_HOME/bin:$PATH"
export VECTORS_RESEARCH_DIR=$HOME/research
OUT=$HOME/out; mkdir -p $OUT/jmh2
# Phase 2: facade-level search (VectorDbBenchmark, n in {10k,100k} x dim in {128,768}) and IVF search at
# scale (IvfSearchBenchmark, same grid x nprobe in {1..32}); fork=1 because each param combo rebuilds its
# index in @Setup; 2 interleaved rounds per selector.
BENCH=("VectorDbBenchmark." "IvfSearchBenchmark.searchIvf")
echo "=== JMH phase 2: ${#BENCH[@]} selectors x 2 tags x 2 rounds, interleaved per selector; fork=1 warmup=3 iterations=5  $(date -u +%T)"
for r in 1 2; do for b in "${BENCH[@]}"; do
  if [[ $b == *. ]]; then re="${b%.}\\."; else re="${b%.*}\\.${b##*.}\$"; fi
  stem=${b%.}
  for t in v0.1.34 v0.1.35; do
    name=$t-${stem//./_}-r$r
    (cd ~/$t && rm -rf vectors-bench/build/results/jmh && ./gradlew :vectors-bench:jmh -Pjmh.includes="$re" -Pjmh.resultFormat=JSON -Pjmh.fork=1 -Pjmh.warmup=3 -Pjmh.iterations=5 --no-daemon > $OUT/jmh2/$name.log 2>&1; rc=$?; f=vectors-bench/build/results/jmh/results.json; [ -s $f ] && cp $f $OUT/jmh2/$name.json; echo "  $t $b r$r exit=$rc json=$([ -s $f ] && echo yes || echo NO) $(date -u +%T)")
  done
done; done
echo "VECJMH2-DONE $(date -u +%FT%TZ)"
